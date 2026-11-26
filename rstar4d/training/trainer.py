from __future__ import annotations
from collections import defaultdict, Counter
import contextlib
import copy
import csv
import json
import math
from pathlib import Path
import time
import traceback
import uuid
import torch
from torch.utils.data import DataLoader
from ..config import validate_train
from ..models import create_model, forward_volume, initialize_stage2
from ..models.model import attach_cycn
from ..data.contracts import validate_manifest, validate_split, snapshot
from ..data.sampling import PatchDataset, validation_samples
from ..utils.io import atomic_json, append_jsonl, read_json, canonical_hash, sha256, code_fingerprint, environment
from ..utils.seed import seed_all, rng_state, restore_rng
from .losses import compute_loss
from .checkpoint import atomic_save, load_checkpoint, cycn_settings
from .optimization import build_optimizer, learning_rates


def select_device(name):
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; select device=cpu explicitly')
    if device.type not in ('cpu','cuda'): raise ValueError('Only CPU/CUDA are supported')
    return device


def signature(config,data):
    algorithm=copy.deepcopy(config)
    for key in ['device','workers','pin_memory','log_every']: algorithm.pop(key,None)
    return canonical_hash({'config':algorithm,'data':data,'code':code_fingerprint()})


def _amp(device,enabled):
    return torch.cuda.amp.autocast(enabled=True) if device.type=='cuda' and enabled else contextlib.nullcontext()


def _loader(dataset,config,epoch,stage):
    generator=torch.Generator().manual_seed(config['seed']+100003*epoch+stage)
    return DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config['workers'],
        pin_memory=config['pin_memory'],generator=generator,persistent_workers=False)


def _batch_forward(model, x, batch):
    if getattr(model,'cycn_branch',None) is None:
        return forward_volume(model,x)
    return forward_volume(model,x,batch['temporal_topology'],batch['case_id'])


def _initial_model(stage, config, init, rows, data):
    profile = config['model_profile']
    settings = config['cycn']
    model = create_model(stage, profile)
    if init is None:
        if stage == 2:
            raise ValueError('Stage II needs init-from-stage1, init-from-stage2, or resume')
        return attach_cycn(model, settings), {'kind':'random','source_stage':None,'sha256':None}
    source_stage = int(init['source_stage'])
    if source_stage not in (1, 2) or (stage == 1 and source_stage != 1):
        raise ValueError('Invalid initialization stage')
    path = Path(init['path']).resolve()
    digest = sha256(path)
    value = torch.load(path, map_location='cpu', weights_only=True)
    if digest != sha256(path):
        raise ValueError('Initialization weights changed during loading')
    state = value.get('state_dict', value)
    if value.get('stage', source_stage) != source_stage:
        raise ValueError('Initialization stage metadata mismatch')
    meta = rows[0]['metadata']
    contracts = {'model_profile': profile, 'normalization': config['normalization'],
                 'intensity_domain': meta['intensity_domain'],
                 'calibration_application': meta['calibration_application']}
    if source_stage == 2 and settings['mode'] != 'off' and not all(key in value for key in contracts):
        raise ValueError('CycN Stage II warm start requires versioned weights with profile and data contracts')
    for key, expected in contracts.items():
        if key in value and value[key] != expected:
            raise ValueError(f'Initialization {key} mismatch')
    if source_stage == 2 and 'split' in value and value['split'] != data['split']:
        raise ValueError('Initialization animal split mismatch')
    if source_stage == 2 and 'data_snapshot' in value and value['data_snapshot'] != data:
        raise ValueError('Initialization data snapshot mismatch')
    source_settings = cycn_settings(value)
    transfer = stage == 2 and source_stage == 1
    if source_settings['mode'] != 'off':
        if not transfer or any(source_settings[key] != settings[key] for key in ('mode', 'fusion', 'boundary')):
            raise ValueError('Enhanced initialization requires matching CycN settings and Stage I to II transfer')
        attach_cycn(model, settings)
    if transfer:
        initialize_stage2(model, state)
    else:
        model.load_state_dict(state, strict=True)
    if source_settings['mode'] == 'off':
        attach_cycn(model, settings)
    return model, {'kind':'stage_transfer' if transfer else 'warm_start',
                   'source_stage':source_stage,'sha256':digest,'path':str(path)}


def validate_model(model,records,config,stage,samples,device):
    data=PatchDataset(records,config,stage,0,len(samples),samples)
    cases=defaultdict(lambda:{'sum':0.,'count':0,'animal_id':None})
    model.eval()
    with torch.no_grad():
        for batch in _loader(data,config,0,stage):
            x=batch['input'].to(device); target=batch['target'].to(device); valid=batch['valid'].to(device)
            with _amp(device,config['amp']):
                pred=_batch_forward(model,x,batch)
            if not torch.isfinite(pred).all(): raise FloatingPointError(f'Nonfinite validation output: {batch["case_id"]}')
            item=cases[batch['case_id'][0]]
            item['sum']+=float((pred.float()-target).abs()[valid].sum())
            item['count']+=int(valid.sum()); item['animal_id']=batch['animal_id'][0]
    if not cases or any(v['count']==0 for v in cases.values()): raise ValueError('No valid validation voxels')
    case_values={k:v['sum']/v['count'] for k,v in cases.items()}
    animals=defaultdict(list)
    for k,v in cases.items(): animals[v['animal_id']].append(case_values[k])
    animal_values={k:sum(v)/len(v) for k,v in animals.items()}
    return {'l1':sum(case_values.values())/len(case_values),
        'animal_l1':sum(animal_values.values())/len(animal_values),'cases':case_values,'animals':animal_values}


def _write_metrics_csv(path,row):
    keys=['epoch','stage','steps','train_l1','train_total','validation_l1','validation_animal_l1','lr','seconds','peak_allocated_gib','skipped_ssim']
    exists=path.exists()
    with path.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        if not exists: writer.writeheader()
        writer.writerow({k:row[k] for k in keys})


def train(config,manifest,split,run_dir,stage,resume=None,init=None,stop_after_epoch=None):
    """Train one stage; resume is guaranteed at completed epoch boundaries only."""
    config=validate_train(config); run_dir=Path(run_dir).resolve()
    if stage not in (1,2): raise ValueError('stage must be 1 or 2')
    if resume and init: raise ValueError('resume and init are mutually exclusive')
    saved=load_checkpoint(resume) if resume else None
    if saved and saved['stage']!=stage: raise ValueError('Checkpoint stage mismatch')
    if saved and (cycn_settings(saved)!=config['cycn'] or saved['model_profile']!=config['model_profile']):
        raise ValueError('Resume configuration disagrees with saved model structure or training policy')
    if saved:
        # Migration is supported by moving/copying the whole run, never by injecting
        # one checkpoint into an unrelated destination or losing its historical best.
        if Path(resume).resolve()!=run_dir/'checkpoints/last.pt':
            raise ValueError('Resume requires this run directory\'s checkpoints/last.pt; migrate the entire run directory or use a warm start')
        identity_path=run_dir/'run_identity.json'
        if not identity_path.is_file() or read_json(identity_path).get('run_id')!=saved.get('run_id'):
            raise ValueError('Resume run ownership mismatch/missing run identity')
        best_path=run_dir/'checkpoints/best.pt'
        if not best_path.is_file(): raise ValueError('Resume requires the historical best.pt from the same run')
        historical_best=load_checkpoint(best_path)
        if historical_best.get('run_id')!=saved.get('run_id') or historical_best['signature']!=saved['signature'] or historical_best['best']!=saved['best']:
            raise ValueError('Resume best/last checkpoint lineage mismatch')
        run_id=saved['run_id']
    else: run_id=str(uuid.uuid4())
    if not resume and run_dir.exists() and any(run_dir.iterdir()): raise FileExistsError('Training directory is nonempty; use resume or a new run')
    rows=validate_manifest(manifest,split,require_target=True)
    split=validate_split(rows,split)
    training=[r for r in rows if str(r['animal_id']) in split['train']]
    validation=[r for r in rows if str(r['animal_id']) in split['val']]
    if not training or not validation: raise ValueError('Training and validation animals/cases are both required')
    data=snapshot(rows,split); sig=signature(config,data)
    if saved and saved['signature']!=sig: raise ValueError('Resume rejected: data, split, algorithm/config or implementation changed; use an explicit warm start')
    device=select_device(config['device']); seed_all(config['seed'],config['deterministic'],config['threads'])
    if saved:
        model=create_model(stage,saved['model_profile'],cycn_settings(saved))
        model.load_state_dict(saved['state_dict'],strict=True); initialization=saved['initialization']
    else:
        if init is None and stage==1 and config['init_weights']: init={'path':config['init_weights'],'source_stage':1}
        model,initialization=_initial_model(stage,config,init,rows,data)
    model.to(device)
    optimizer=build_optimizer(model,config)
    group_names=list(learning_rates(optimizer))
    total_epochs=config['epochs'][str(stage)]
    scheduler=(torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['milestones'][str(stage)],gamma=config['gamma'])
        if config['scheduler']=='multistep' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_epochs,eta_min=config['min_lr']))
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and config['amp'])
    start_epoch=1; step=0; best=math.inf
    if saved:
        if saved.get('optimizer_group_names',['backbone'])!=group_names:
            raise ValueError('Optimizer parameter group order differs from the checkpoint')
        optimizer.load_state_dict(saved['optimizer'])
        for group,name in zip(optimizer.param_groups,group_names):
            if group.setdefault('name',name)!=name:
                raise ValueError('Optimizer state group name differs from the checkpoint')
        scheduler_state=copy.deepcopy(saved['scheduler'])
        if config['scheduler']=='multistep': scheduler_state['milestones']=Counter(scheduler_state['milestones'])
        scheduler.load_state_dict(scheduler_state); scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['next_epoch']; step=saved['step']; best=saved['best']; restore_rng(saved['rng'])
    run_dir.mkdir(parents=True,exist_ok=True); ckdir=run_dir/'checkpoints'; ckdir.mkdir(exist_ok=True)
    atomic_json(run_dir/'run_identity.json',{'run_id':run_id,'stage':stage,'signature':sig})
    samples=(saved['validation_samples'] if saved else validation_samples(validation,config,stage))
    # Validation sample identity is part of the checkpoint, including patch shapes/origins.
    atomic_json(run_dir/'resolved_config.json',config); atomic_json(run_dir/'data_snapshot.json',data)
    atomic_json(run_dir/'split.json',split); atomic_json(run_dir/'validation_samples.json',samples)
    atomic_json(run_dir/'environment.json',environment())
    if start_epoch>total_epochs:
        return {'stage':stage,'complete':True,'checkpoint':str(ckdir/'best.pt'),'last':str(ckdir/'last.pt'),'epochs':total_epochs}
    length=config['steps_per_epoch'][str(stage)] or (sum(r['shape'][1] for r in training) if stage==1 else 15*len(training))
    if length<1: raise ValueError('Empty training epoch')
    final_epoch=total_epochs if stop_after_epoch is None else min(total_epochs,int(stop_after_epoch))
    loss_config={**config['loss'],'normalization_scale':config['normalization']['scale'],'normalization_offset':config['normalization']['offset']}
    try:
        for epoch in range(start_epoch,final_epoch+1):
            begin=time.perf_counter(); model.train()
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            dataset=PatchDataset(training,config,stage,epoch,length)
            loader=_loader(dataset,config,epoch,stage)
            total_loss=0.; total_l1=0.; skipped=0; valid_ratios=[]; amp_overflows=0; epoch_updates=0; shape_counts=Counter(); animal_counts=Counter()
            optimizer.zero_grad(set_to_none=True)
            for index,batch in enumerate(loader):
                x=batch['input'].to(device); target=batch['target'].to(device); valid=batch['valid'].to(device)
                shape_counts['x'.join(map(str,x.shape[2:]))]+=1; animal_counts[batch['animal_id'][0]]+=1
                with _amp(device,config['amp']):
                    pred=_batch_forward(model,x,batch)
                loss=compute_loss(pred,target,valid,stage,loss_config)
                if not torch.isfinite(loss['total']): raise FloatingPointError(f'Nonfinite loss at {batch["case_id"]}, epoch={epoch}, draw={index}')
                # Average over the actual size of each accumulation group, including the final short group.
                group_start=(index//config['accumulation'])*config['accumulation']
                group_size=min(config['accumulation'],length-group_start)
                scaler.scale(loss['total']/group_size).backward()
                if (index+1)%config['accumulation']==0 or index+1==length:
                    scaler.unscale_(optimizer)
                    finite_gradients=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    if not finite_gradients and not scaler.is_enabled():
                        raise FloatingPointError(f'Nonfinite fp32 gradient at {batch["case_id"]}')
                    if finite_gradients and config['clip_grad'] is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'])
                    previous_scale=scaler.get_scale()
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if finite_gradients: step+=1; epoch_updates+=1
                    else:
                        # Match the public GradScaler behavior: skip the overflowing update,
                        # reduce the scale, and explicitly record it. Never save a zero-update epoch.
                        amp_overflows+=1
                        append_jsonl(run_dir/'train.jsonl',{'event':'amp_overflow','epoch':epoch,'draw':index,
                            'case_id':batch['case_id'][0],'scale_before':previous_scale,'scale_after':scaler.get_scale()})
                total_loss+=float(loss['total'].detach()); total_l1+=float(loss['l1'].detach()); skipped+=int(loss['skipped_ssim'])
                valid_ratios.append(float(valid.float().mean()))
                if (index+1)%config['log_every']==0 or index+1==length:
                    append_jsonl(run_dir/'train.jsonl',{'event':'batch','epoch':epoch,'stage':stage,'draw':index,'step':step,
                        'case_id':batch['case_id'][0],'shape':list(x.shape),'l1':float(loss['l1'].detach()),
                        'total':float(loss['total'].detach()),'valid_fraction':valid_ratios[-1],'skipped_ssim':loss['skipped_ssim']})
            if skipped==length and loss_config['kind']!='l1' and loss_config['ssim_weight']>0:
                raise ValueError('No valid SSIM windows during entire epoch; change explicit patch/mask/loss configuration')
            if epoch_updates==0: raise FloatingPointError('No optimizer updates in epoch: AMP overflow calibration exhausted the tiny epoch; use more draws or explicit fp32')
            evaluation=validate_model(model,validation,config,stage,samples,device)
            improved=evaluation['l1']<best
            if improved: best=evaluation['l1']
            rates_used=learning_rates(optimizer); lr_used=optimizer.param_groups[0]['lr']; scheduler.step()
            entry={'event':'epoch','epoch':epoch,'stage':stage,'steps':step,'train_l1':total_l1/length,
                'train_total':total_loss/length,'validation_l1':evaluation['l1'],'validation_animal_l1':evaluation['animal_l1'],
                'lr':lr_used,'next_lr':optimizer.param_groups[0]['lr'],'learning_rates':rates_used,
                'next_learning_rates':learning_rates(optimizer),'seconds':time.perf_counter()-begin,
                'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0.,
                'skipped_ssim':skipped,'amp_overflows':amp_overflows,'optimizer_updates':epoch_updates,
                'sample_count':length,'shape_counts_tzyx':dict(shape_counts),'animal_draw_counts':dict(animal_counts),
                'valid_fraction':sum(valid_ratios)/len(valid_ratios),'validation_cases':evaluation['cases']}
            payload={'schema_version':1,'checkpoint_kind':'training','run_id':run_id,'state_dict':model.state_dict(),'stage':stage,
                'model_profile':config['model_profile'],'completed_epoch':epoch,'next_epoch':epoch+1,'step':step,
                'cycn':{'version':1,**config['cycn']},'optimizer_group_names':group_names,
                'stage_complete':epoch==total_epochs,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'best':best,'rng':rng_state(),'config':config,'signature':sig,
                'data_snapshot':data,'validation_samples':samples,'initialization':initialization,
                'manifest':str(Path(manifest).resolve()),'split':split,'environment':environment(),
                'code_fingerprint':code_fingerprint(),'normalization':config['normalization'],
                'intensity_domain':rows[0]['metadata']['intensity_domain'],'calibration_application':rows[0]['metadata']['calibration_application']}
            if improved: atomic_save(ckdir/'best.pt',payload)
            atomic_save(ckdir/'last.pt',payload)
            append_jsonl(run_dir/'train.jsonl',entry); _write_metrics_csv(run_dir/'metrics.csv',entry)
            print(json.dumps(entry,ensure_ascii=False),flush=True)
        atomic_json(run_dir/'status.json',{'stage':stage,'completed_epoch':final_epoch,'complete':final_epoch==total_epochs})
    except Exception as exc:
        atomic_json(run_dir/'error.json',{'error':str(exc),'traceback':traceback.format_exc(),'stage':stage})
        raise
    return {'stage':stage,'complete':final_epoch==total_epochs,'checkpoint':str(ckdir/'best.pt'),'last':str(ckdir/'last.pt'),'epochs':final_epoch}
