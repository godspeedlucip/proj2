from __future__ import annotations
from collections import defaultdict, Counter
import contextlib
import copy
import hashlib
import json
from itertools import islice
import math
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..config import validate_train, resolve_stage2
from ..models import create_model, forward_volume, initialize_stage2, attach_stage2_extensions
from ..data.contracts import validate_manifest, validate_split, snapshot
from ..data.context import crop_core, plan_window
from ..data.sampling import PatchDataset, epoch_draw_budget, validation_samples
from ..utils.io import atomic_json, append_jsonl, canonical_hash, sha256, code_fingerprint, environment
from ..utils.seed import seed_all, rng_state, restore_rng
from .losses import compute_loss, boundary_terms, boundary_scale
from .motion_losses import motion_terms
from .tensorboard import (CardiacAverages, axial_preview, measured_preview_source,
                          recover_training_events, validation_preview, write_training_epoch)
from .checkpoint import append_metrics_csv, publish_epoch, read_resume, restore_history, run_lock


def select_device(name):
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; select device=cpu explicitly')
    if device.type not in ('cpu','cuda'): raise ValueError('Only CPU/CUDA are supported')
    return device


def signature(config,data):
    algorithm=copy.deepcopy(config)
    for key in ['device','workers','pin_memory','log_every','tensorboard']: algorithm.pop(key,None)
    return canonical_hash({'config':algorithm,'data':data,'code':code_fingerprint()})


def _amp(device,enabled):
    return torch.cuda.amp.autocast(enabled=True) if device.type=='cuda' and enabled else contextlib.nullcontext()


def _loader(dataset,config,epoch,stage,*,sampler=None):
    generator=torch.Generator().manual_seed(config['seed']+100003*epoch+stage)
    return DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=config['workers'],
        pin_memory=config['pin_memory'],generator=generator,persistent_workers=False)


def _matched_draw_order(dataset):
    """Pair matching proposed read shapes while preserving every seeded draw."""
    if dataset.resolved['crop_mode'] != 'core_with_context' or dataset.samples is not None:
        return range(len(dataset))
    buckets = defaultdict(list)
    for draw in range(len(dataset)):
        seed = np.random.SeedSequence([dataset.config['seed'],dataset.epoch,draw,dataset.stage])
        case, core, origin = dataset.proposal(draw,np.random.default_rng(seed))
        phases, *volume = dataset.shapes[case]
        window = plan_window(volume,origin,core,dataset.resolved['recipe'])
        shape = tuple(max(sl.stop,start-lo) for sl,start,lo in
                      zip(window.core,window.stop,window.origin))
        buckets[(phases,*shape)].append(draw)
    paired, singles = [], []
    for draws in buckets.values():
        count = len(draws) // 2 * 2
        paired.extend(draws[:count])
        singles.extend(draws[count:])
    return paired + singles


def _microbatches(loader, width):
    draws = enumerate(loader)
    while True:
        items = tuple(islice(draws,width))
        if not items:
            return
        shape = items[0][1]['input'].shape
        if all(batch['input'].shape == shape for _,batch in items):
            yield items
        else:
            for item in items:
                yield (item,)


def batch_context(batch, device):
    if 'context' in batch:
        return {name:value.to(device) for name,value in batch['context'].items()}
    return None


def _training_forward(model, x, context, device, resolved, amp):
    settings = resolved['motion']
    use_motion = any(settings[key] for key in ('struct_weight','smooth_weight','surface_weight'))
    with _amp(device,amp):
        if use_motion:
            return forward_volume(model,x,context=context,return_aux=True)
        if context is not None:
            return forward_volume(model,x,context=context),None
        return forward_volume(model,x),None


def compute_batch(model, batch, device, stage, resolved, loss_config, epoch, amp):
    context = batch_context(batch,device)
    pred,aux = _training_forward(model,batch['input'].to(device),context,device,resolved,amp)
    return _sample_loss(pred,aux,context,batch,device,stage,resolved,loss_config,epoch)


def _sample_loss(pred, aux, context, batch, device, stage, resolved, loss_config, epoch):
    target,valid = (batch[name].to(device) for name in ('target','valid'))
    if context is not None:
        core_pred,core_target,core_valid = (crop_core(value,context) for value in (pred,target,valid))
    else:
        core_pred,core_target,core_valid = pred,target,valid
    loss = compute_loss(core_pred,core_target,core_valid,stage,loss_config)
    diagnostics = {}
    if aux is not None:
        settings = resolved['motion']
        supervision = {'target':target,'source_path':batch['source_path']}
        if settings['surface_weight']:
            supervision.update({key:batch[key].to(device) for key in ('surface','distance','correspondence')})
        terms = motion_terms(aux,supervision,context,settings,resolved['recipe'])
        for key in ('struct','smooth','surface'):
            raw = terms.pop(key)
            weighted = settings[key+'_weight']*raw
            loss['total'] = loss['total'] + weighted
            diagnostics[key] = float(raw.detach())
            diagnostics[key+'_weighted'] = float(weighted.detach())
        diagnostics.update(terms)
    if resolved['boundary']['enabled']:
        cfg = resolved['boundary']
        bands = crop_core(batch['bands'].to(device),context)
        boundary = boundary_terms(core_pred,core_target,bands,cfg['structure_weights'],cfg['grad_weight_vox'])
        weight = boundary_scale(epoch,cfg['ramp_epochs'],cfg['weight'])
        loss['total'] = loss['total'] + weight*boundary['total']
        diagnostics.update(boundary_weight=weight,boundary=float(boundary['total'].detach()),
                           boundary_weighted=float(boundary['total'].detach())*weight,
                           boundary_overlap=boundary['overlap'])
        diagnostics['structures'] = {}
        for key,value in boundary['structures'].items():
            diagnostics['structures'][key] = {
                **{field:float(n.detach()) if torch.is_tensor(n) else n for field,n in value.items()
                   if field!='phase_voxels'},
                'phase_voxels':dict(zip(context['phase_labels'][0].tolist(),value['phase_voxels'][0].tolist())),
            }
        diagnostics['boundary_dual_hit'] = int(all(item['voxels']>0 for item in diagnostics['structures'].values()))
    return loss, diagnostics


def _compute_microbatch(model, batches, device, stage, resolved, loss_config, epoch, amp):
    if len(batches)==1:
        return [compute_batch(model,batches[0],device,stage,resolved,loss_config,epoch,amp)]
    images = torch.cat([batch['input'] for batch in batches]).to(device)
    context = None
    if 'context' in batches[0]:
        context = {name:torch.cat([batch['context'][name] for batch in batches]).to(device)
                   for name in batches[0]['context']}
    prediction,aux = _training_forward(model,images,context,device,resolved,amp)
    results = []
    for index,batch in enumerate(batches):
        item = slice(index,index+1)
        local = {name:value[item] for name,value in context.items()} if context is not None else None
        motion = None
        if aux is not None:
            motion = {**aux,'flow_vox':aux['flow_vox'][item],'valid_samples':aux['valid_samples'][item],
                      'origin':local.get('origin')}
        results.append(_sample_loss(prediction[item],motion,local,batch,device,stage,resolved,loss_config,epoch))
    return results


def _training_draws(loader, model, device, stage, resolved, loss_config, epoch, config, scaler, length):
    effective = config['accumulation']
    for items in _microbatches(loader, config['micro_batch_size']):
        results = _compute_microbatch(model,[batch for _,batch in items],device,stage,resolved,
                                      loss_config,epoch,config['amp'])
        for (index,batch),(loss,_) in zip(items,results):
            if not torch.isfinite(loss['total']):
                raise FloatingPointError(f'Nonfinite loss at {batch["case_id"]}, epoch={epoch}, draw={index}')
        group_start = items[0][0] // effective * effective
        divisor = min(effective,length-group_start)
        scaler.scale(sum(loss['total'] for loss,_ in results)/divisor).backward()
        for (index,batch),(loss,diagnostics) in zip(items,results):
            yield index,batch,loss,diagnostics,len(items)


def validate_model(model,records,config,stage,samples,device):
    data=PatchDataset(records,config,stage,0,len(samples),samples)
    preview = None
    by_sample = {row.get('sample_id', row['case_id']): row for row in records}
    cases=defaultdict(lambda:{'sum':0.,'count':0,'animal_id':None})
    model.eval()
    with torch.no_grad():
        for batch in _loader(data,config,0,stage):
            x=batch['input'].to(device); target=batch['target'].to(device); valid=batch['valid'].to(device)
            context = batch_context(batch,device)
            with _amp(device,config['amp']):
                pred=forward_volume(model,x,context=context) if context is not None else forward_volume(model,x)
            if context is not None:
                pred,target,valid = (crop_core(value,context) for value in (pred,target,valid))
            if not torch.isfinite(pred).all(): raise FloatingPointError(f'Nonfinite validation output: {batch["case_id"]}')
            if preview is None and config['tensorboard']:
                image = crop_core(x, context) if context is not None else x
                sample = batch.get('sample_id', batch['case_id'])[0]
                preview = validation_preview(image, pred, target, batch, by_sample[sample], config)
            item=cases[batch['case_id'][0]]
            item['sum']+=float((pred.float()-target).abs()[valid].sum())
            item['count']+=int(valid.sum()); item['animal_id']=batch['animal_id'][0]
    if not cases or any(v['count']==0 for v in cases.values()): raise ValueError('No valid validation voxels')
    case_values={k:v['sum']/v['count'] for k,v in cases.items()}
    animals=defaultdict(list)
    for k,v in cases.items(): animals[v['animal_id']].append(case_values[k])
    animal_values={k:sum(v)/len(v) for k,v in animals.items()}
    return {'l1':sum(case_values.values())/len(case_values),
        'animal_l1':sum(animal_values.values())/len(animal_values),'cases':case_values,'animals':animal_values,
        'preview':preview}


def train(config,manifest,split,run_dir,stage,resume=None,init=None,stop_after_epoch=None):
    """Train one stage; resume is guaranteed at completed epoch boundaries only."""
    run_dir = Path(run_dir).resolve()
    if resume and Path(resume).resolve().parent.parent in run_dir.parents:
        raise ValueError('Historical resume requires a new run outside the parent run directory')
    with run_lock(run_dir):
        return _train(config,manifest,split,run_dir,stage,resume,init,stop_after_epoch)


def _train(config,manifest,split,run_dir,stage,resume,init,stop_after_epoch):
    config=validate_train(config)
    if stage not in (1,2): raise ValueError('stage must be 1 or 2')
    resolved = resolve_stage2(config,stage)
    if resume and init: raise ValueError('resume and init are mutually exclusive')
    saved, identity, source = read_resume(resume,run_dir) if resume else (None, None, None)
    if identity and identity['stage']!=stage: raise ValueError('Checkpoint stage mismatch')
    if not resume and run_dir.exists() and any(run_dir.iterdir()): raise FileExistsError('Training directory is nonempty; use resume or a new run')
    sidecars = []
    if resolved['prior']['mode']=='geometry':
        sidecars.extend(('projections','angle_summary'))
    if resolved['boundary']['enabled'] or resolved['motion']['surface_weight']:
        sidecars.append('annotations')
    rows=validate_manifest(manifest,split,require_target=True,sidecars=sidecars,prefer_angle_summary=True)
    split=validate_split(rows,split)
    training=[r for r in rows if str(r['animal_id']) in split['train']]
    validation=[r for r in rows if str(r['animal_id']) in split['val']]
    if not training or not validation: raise ValueError('Training and validation animals/cases are both required')
    data=snapshot(rows,split)
    preview_options=config['tensorboard_real']
    measured=measured_preview_source(preview_options,resolved) if preview_options['manifest'] else None
    if measured is not None:
        data['tensorboard_measured']={key:measured.get(key) for key in
                                     ('case_id','sha256','projections_sha256','angle_summary_sha256')}
    sig=signature(config,data)
    if identity and identity['signature']!=sig:
        raise ValueError('Resume rejected: data, split, algorithm/config or implementation changed; use an explicit warm start')
    if saved and (saved['model_profile']!=config['model_profile'] or saved['data_snapshot']!=data or saved['split']!=split):
        raise ValueError('Checkpoint architecture, data snapshot or split mismatch; use an explicit warm start')
    if identity and not saved:
        original = identity['initialization']
        if original.get('path'):
            if sha256(original['path']) != original['sha256']:
                raise ValueError('First-epoch initialization weights changed; use a new run')
            init = {'path':original['path'], 'source_stage':original['source_stage']}
    device=select_device(config['device']); seed_all(config['seed'],config['deterministic'],config['threads'])
    model=create_model(stage,config['model_profile'])
    initialization={'kind':'random','source_stage':None,'sha256':None}
    if saved:
        if saved.get('stage2_resolved') is not None and saved['stage2_resolved']!=resolved:
            raise ValueError('Checkpoint Stage II recipe does not match the run')
        if stage==2:
            attach_stage2_extensions(model,resolved)
        model.load_state_dict(saved['state_dict'],strict=True); initialization=saved['initialization']
    else:
        if init is None and stage==1 and config['init_weights']: init={'path':config['init_weights'],'source_stage':1}
        if init:
            value=torch.load(init['path'],map_location='cpu',weights_only=True)
            state=value.get('state_dict',value)
            source_stage=int(init['source_stage'])
            if 'stage' in value and value['stage']!=source_stage: raise ValueError('Initialization stage metadata mismatch')
            if stage==1 and source_stage!=1: raise ValueError('Stage I requires Stage I initialization')
            if stage==2 and source_stage==1:
                initialize_stage2(model,state)
                attach_stage2_extensions(model,resolved)
            elif stage==2:
                load_report = warm_start_stage2(model,state,value.get('stage2_resolved'),resolved)
            else:
                model.load_state_dict(state,strict=True)
            initialization={'kind':'stage_transfer' if stage==2 and source_stage==1 else 'warm_start',
                'source_stage':source_stage,'sha256':sha256(init['path']),'path':str(Path(init['path']).resolve())}
            if stage==2 and source_stage==2:
                initialization['parameters'] = load_report
        elif stage==2: raise ValueError('Stage II needs init-from-stage1, init-from-stage2, or resume')
    model.to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],
        betas=tuple(config['betas']),weight_decay=config['weight_decay'])
    total_epochs=config['epochs'][str(stage)]
    scheduler=(torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['milestones'][str(stage)],gamma=config['gamma'])
        if config['scheduler']=='multistep' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_epochs,eta_min=config['min_lr']))
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and config['amp'])
    samples=saved['validation_samples'] if saved else validation_samples(validation,config,stage)
    start_epoch=1; step=0; best=math.inf; best_epoch=0
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        scheduler_state=copy.deepcopy(saved['scheduler'])
        if config['scheduler']=='multistep': scheduler_state['milestones']=Counter(scheduler_state['milestones'])
        scheduler.load_state_dict(scheduler_state); scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['next_epoch']; step=saved['step']; best=saved['best']; best_epoch=saved['best_epoch']
        identity=restore_history(source,run_dir,saved,identity)
        recover_training_events(run_dir, saved['completed_epoch'], config['tensorboard'])
        restore_rng(saved['rng'])
    elif identity:
        if canonical_hash(samples) != identity['validation_samples_sha256']:
            raise ValueError('First-epoch fixed validation sample mismatch')
        restore_history(source,run_dir,None,identity)
        recover_training_events(run_dir, 0, config['tensorboard'])
    else:
        identity={'run_id':str(uuid.uuid4()),'stage':stage,'signature':sig,
                  'manifest':str(Path(manifest).resolve()),'initialization':initialization,
                  'validation_samples_sha256':canonical_hash(samples)}
        atomic_json(run_dir/'run_identity.json',identity)
    run_dir.mkdir(parents=True,exist_ok=True); ckdir=run_dir/'checkpoints'; ckdir.mkdir(exist_ok=True)
    # Validation sample identity is part of the checkpoint, including patch shapes/origins.
    atomic_json(run_dir/'resolved_config.json',config); atomic_json(run_dir/'data_snapshot.json',data)
    atomic_json(run_dir/'initialization.json',initialization)
    if stage==2:
        atomic_json(run_dir/'stage2_resolved.json',resolved)
    atomic_json(run_dir/'split.json',split); atomic_json(run_dir/'validation_samples.json',samples)
    atomic_json(run_dir/'environment.json',environment())
    if start_epoch>total_epochs:
        return {'stage':stage,'complete':True,'checkpoint':str(ckdir/'best.pt'),'last':str(ckdir/'last.pt'),'epochs':total_epochs}
    length=epoch_draw_budget(training,config,stage)
    if length<1: raise ValueError('Empty training epoch')
    final_epoch=total_epochs if stop_after_epoch is None else min(total_epochs,int(stop_after_epoch))
    if final_epoch < max(1,start_epoch-1):
        raise ValueError('stop-after-epoch precedes the selected checkpoint or first epoch')
    loss_config={**config['loss'],'normalization_scale':config['normalization']['scale'],'normalization_offset':config['normalization']['offset']}
    try:
        for epoch in range(start_epoch,final_epoch+1):
            begin=time.perf_counter(); model.train()
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            dataset=PatchDataset(training,config,stage,epoch,length)
            order = _matched_draw_order(dataset) if config['micro_batch_size']>1 else None
            loader=_loader(dataset,config,epoch,stage,sampler=order)
            total_loss=0.; total_l1=0.; skipped=0; valid_ratios=[]; amp_overflows=0; epoch_updates=0; shape_counts=Counter(); animal_counts=Counter()
            cardiac_counts=Counter(); draws=hashlib.sha256()
            cardiac_means = CardiacAverages()
            micro_batch_samples = Counter()
            optimizer.zero_grad(set_to_none=True)
            for index,batch,loss,diagnostics,physical_size in _training_draws(
                    loader,model,device,stage,resolved,loss_config,epoch,config,scaler,length):
                x=batch['input']; valid=batch['valid']
                context = batch.get('context')
                micro_batch_samples[physical_size] += 1
                shape_counts['x'.join(map(str,x.shape[2:]))]+=1; animal_counts[batch['animal_id'][0]]+=1
                cardiac_means.update(diagnostics)
                window = {}
                if context is not None:
                    window = {'case_id':batch['case_id'][0],'sample_id':batch['sample_id'][0],'core_origin':batch['origin'][0].tolist(),
                              'core_shape':batch['patch_shape'][0].tolist(),'window_origin':context['origin'][0].tolist(),
                              'window_shape':list(x.shape[-3:]),'phase_labels':context['phase_labels'][0].tolist(),
                              'left_context_vox':context['core_bounds'][0,:,0].tolist(),
                              'right_context_vox':(torch.tensor(x.shape[-3:])-context['core_bounds'][0,:,1]).tolist()}
                    draws.update(json.dumps(window,sort_keys=True).encode())
                    for field,number in diagnostics.items():
                        if field!='structures': cardiac_counts[field] += number
                    for structure,values in diagnostics.get('structures',{}).items():
                        cardiac_counts[structure+'_voxels'] += values['voxels']
                        cardiac_counts[structure+'_pairs'] += values['pairs']
                        cardiac_counts[structure+'_hit_cores'] += int(values['voxels']>0)
                        cardiac_counts[structure+'_l1'] += values['l1']
                        cardiac_counts[structure+'_gradient'] += values['gradient']
                    append_jsonl(run_dir/'cardiac_samples.jsonl',{'epoch':epoch,'draw':index,'animal_id':batch['animal_id'][0],
                                                                  **window,**diagnostics})
                else:
                    draw = [batch['sample_id'][0], batch['origin'][0].tolist(), batch['patch_shape'][0].tolist()]
                    draws.update(json.dumps(draw).encode())
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
                            'case_id':batch['case_id'][0],'sample_id':batch['sample_id'][0],'scale_before':previous_scale,'scale_after':scaler.get_scale()})
                total_loss+=float(loss['total'].detach()); total_l1+=float(loss['l1'].detach()); skipped+=int(loss['skipped_ssim'])
                valid_ratios.append(float(crop_core(valid,context).float().mean() if context is not None else valid.float().mean()))
                if (index+1)%config['log_every']==0 or index+1==length:
                    append_jsonl(run_dir/'train.jsonl',{'event':'batch','epoch':epoch,'stage':stage,'draw':index,'step':step,
                        'case_id':batch['case_id'][0],'sample_id':batch['sample_id'][0],'shape':list(x.shape),
                        'physical_batch_size':physical_size,'l1':float(loss['l1'].detach()),
                        'total':float(loss['total'].detach()),'valid_fraction':valid_ratios[-1],'skipped_ssim':loss['skipped_ssim']})
            if skipped==length and loss_config['kind']!='l1' and loss_config['ssim_weight']>0:
                raise ValueError('No valid SSIM windows during entire epoch; change explicit patch/mask/loss configuration')
            if epoch_updates==0: raise FloatingPointError('No optimizer updates in epoch: AMP overflow calibration exhausted the tiny epoch; use more draws or explicit fp32')
            evaluation=validate_model(model,validation,config,stage,samples,device)
            slice_previews={}
            if measured is not None and config['tensorboard']:
                sample=evaluation['preview']['sample_id']
                validation_record=next(row for row in validation if row['sample_id']==sample)
                slice_previews['measured']=axial_preview(model,measured,config,resolved,device,
                    phase_label=preview_options['phase_label'],axial_slice=preview_options['axial_slice'])
                slice_previews['validation_slice']=axial_preview(model,validation_record,config,resolved,device)
            improved=evaluation['l1']<best
            if improved: best=evaluation['l1']; best_epoch=epoch
            lr_used=optimizer.param_groups[0]['lr']; scheduler.step()
            entry={'event':'epoch','epoch':epoch,'stage':stage,'steps':step,'train_l1':total_l1/length,
                'train_total':total_loss/length,'validation_l1':evaluation['l1'],'validation_animal_l1':evaluation['animal_l1'],
                'lr':lr_used,'next_lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-begin,
                'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0.,
                'skipped_ssim':skipped,'amp_overflows':amp_overflows,'optimizer_updates':epoch_updates,
                'sample_count':length,'sampling_sha256':draws.hexdigest(),
                'effective_batch_size':config['accumulation'],'samples_by_physical_batch':dict(micro_batch_samples),
                'shape_counts_tzyx':dict(shape_counts),'animal_draw_counts':dict(animal_counts),
                'valid_fraction':sum(valid_ratios)/len(valid_ratios),'validation_cases':evaluation['cases']}
            if getattr(dataset,'contextual',False):
                entry.update(cardiac=dict(cardiac_counts), cardiac_means=cardiac_means.values(),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            payload={'schema_version':1,'checkpoint_kind':'training','run_id':identity['run_id'],'state_dict':model.state_dict(),'stage':stage,
                'model_profile':config['model_profile'],'completed_epoch':epoch,'next_epoch':epoch+1,'step':step,
                'stage_complete':epoch==total_epochs,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'best':best,'best_epoch':best_epoch,'epoch_metrics':entry,
                'rng':rng_state(),'config':config,'signature':sig,
                'data_snapshot':data,'validation_samples':samples,'initialization':initialization,
                'manifest':str(Path(manifest).resolve()),'split':split,'environment':environment(),
                'code_fingerprint':code_fingerprint(),'normalization':config['normalization'],
                'intensity_domain':rows[0]['metadata']['intensity_domain'],'calibration_application':rows[0]['metadata']['calibration_application']}
            if evaluation['preview'] is not None:
                payload['tensorboard_preview'] = evaluation['preview']
            if slice_previews:
                payload['tensorboard_slices'] = slice_previews
            if stage==2:
                payload['stage2_resolved']=resolved
            publish_epoch(ckdir,payload)
            write_training_epoch(run_dir, payload)
            append_jsonl(run_dir/'train.jsonl',entry); append_metrics_csv(run_dir/'metrics.csv',entry)
            print(json.dumps(entry,ensure_ascii=False),flush=True)
        atomic_json(run_dir/'status.json',{'stage':stage,'completed_epoch':final_epoch,'complete':final_epoch==total_epochs})
    except Exception as exc:
        atomic_json(run_dir/'error.json',{'error':str(exc),'traceback':traceback.format_exc(),'stage':stage})
        raise
    return {'stage':stage,'complete':final_epoch==total_epochs,'checkpoint':str(ckdir/'best.pt'),'last':str(ckdir/'last.pt'),'epochs':final_epoch}


def warm_start_stage2(model,state,source_recipe,target_recipe):
    backbone = {name:value for name,value in state.items() if not name.startswith(('motion.','prior_fusion.'))}
    model.load_state_dict(backbone,strict=True)
    attach_stage2_extensions(model,target_recipe)
    full = model.state_dict()
    compatible = source_recipe and source_recipe['version']==target_recipe['version']
    loaded = dict(backbone)
    if compatible:
        for name,value in state.items():
            if name in full and value.shape==full[name].shape and name not in backbone:
                loaded[name]=value
    model.load_state_dict({**full,**loaded},strict=True)
    return {'loaded':sorted(loaded),'new':sorted(set(full)-set(loaded)),'skipped':sorted(set(state)-set(loaded))}
