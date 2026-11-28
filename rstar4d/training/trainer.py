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
from ..models import attach_stam, attach_ttl, create_model, forward_volume, initialize_stage2, model_from_checkpoint
from ..data.contracts import validate_manifest, validate_split, snapshot
from ..data.sampling import PatchDataset, validation_samples
from ..utils.io import atomic_json, append_jsonl, read_json, canonical_hash, sha256, code_fingerprint, environment
from ..utils.seed import seed_all, rng_state, restore_rng
from .losses import compute_loss
from .checkpoint import atomic_save, load_checkpoint


def select_device(name):
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; select device=cpu explicitly')
    if device.type not in ('cpu','cuda'): raise ValueError('Only CPU/CUDA are supported')
    return device


def signature(config,data):
    algorithm=copy.deepcopy(config)
    for key in ['device','workers','pin_memory','log_every']: algorithm.pop(key,None)
    # The loaded SAM block travels with the checkpoint. Its old local path is provenance.
    algorithm['stam'].pop('sam_checkpoint', None)
    return canonical_hash({'config':algorithm,'data':data,'code':code_fingerprint()})


def _amp(device,enabled):
    return torch.cuda.amp.autocast(enabled=True) if device.type=='cuda' and enabled else contextlib.nullcontext()


def _loader(dataset,config,epoch,stage):
    generator=torch.Generator().manual_seed(config['seed']+100003*epoch+stage)
    return DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config['workers'],
        pin_memory=config['pin_memory'],generator=generator,persistent_workers=False)


def _forward_patch(model, x, batch):
    if getattr(model, 'ttl', None) is not None:
        return forward_volume(model, x, temporal_topology=batch['temporal_topology'])
    return forward_volume(model, x)


def validate_model(model,records,config,stage,samples,device):
    data=PatchDataset(records,config,stage,0,len(samples),samples)
    cases=defaultdict(lambda:{'sum':0.,'count':0,'animal_id':None})
    model.eval()
    with torch.no_grad():
        for batch in _loader(data,config,0,stage):
            x=batch['input'].to(device); target=batch['target'].to(device); valid=batch['valid'].to(device)
            with _amp(device,config['amp']): pred=_forward_patch(model,x,batch)
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


def configure_ttl_training(model):
    from ..models.ttl import TRAINABLE_PARAMETERS

    if getattr(model, 'ttl', None) is None:
        raise ValueError('TTL-only training requires an attached TTL')
    model.requires_grad_(False)
    model.eval()
    model.ttl.requires_grad_(True)
    model.ttl.train()
    parameters = list(model.ttl.parameters())
    if sum(p.numel() for p in parameters) != TRAINABLE_PARAMETERS:
        raise RuntimeError('Unexpected TTL parameter count')
    return parameters


def _check_ttl_optimizer(model, optimizer):
    expected = {id(p) for p in model.ttl.parameters()}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    groups = optimizer.param_groups
    if (len(groups) != 1 or groups[0].get('name') != 'ttl'
            or len(groups[0]['params']) != len(expected)
            or {id(p) for p in groups[0]['params']} != expected or trainable != expected):
        raise RuntimeError('Optimizer and trainable parameters must exactly match TTL')


def make_optimizer(model, config, backbone_frozen=False):
    block = getattr(model, 'stam', None)
    if getattr(model, 'ttl', None) is not None:
        parameters = [{'name': 'ttl', 'params': configure_ttl_training(model)}]
    elif block is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        backbone = list(_backbone(model))
        for param in backbone:
            param.requires_grad_(not backbone_frozen)
        parameters = [{'name': 'stam', 'params': [p for p in block.parameters() if p.requires_grad]}]
        if not backbone_frozen:
            parameters.append({'name': 'backbone', 'params': backbone,
                               'lr': config['learning_rate'] * config['stam']['backbone_lr_ratio']})
    return torch.optim.Adam(parameters, lr=config['learning_rate'], betas=tuple(config['betas']),
                            weight_decay=config['weight_decay'])


def _backbone(model):
    return (p for name, p in model.named_parameters() if not name.startswith('stam.'))


def _unfreeze_backbone(model, optimizer, scheduler, config):
    parameters = list(_backbone(model))
    for param in parameters:
        param.requires_grad_(True)
    ratio = config['stam']['backbone_lr_ratio']
    base_lr = config['learning_rate'] * ratio
    lr = optimizer.param_groups[0]['lr'] * ratio
    optimizer.add_param_group({'name': 'backbone', 'params': parameters, 'lr': lr, 'initial_lr': base_lr})
    scheduler.base_lrs.append(base_lr)
    scheduler._last_lr.append(lr)


def _group_names(optimizer):
    return [group['name'] for group in optimizer.param_groups]


def _stam_mode(value, requested):
    mode = (value.get('stam') or {}).get('architecture', {}).get('mode')
    if mode not in ('joint', 'temporal_only'):
        raise ValueError('Expected an existing Stage II STAM checkpoint with architecture metadata')
    if requested not in ('from_parent', mode):
        raise ValueError(f'Configured STAM mode {requested!r} differs from checkpoint mode {mode!r}')
    return mode


def load_stam_parent(path, config):
    value = load_checkpoint(path, require_training=False)
    if value.get('ttl') is not None:
        raise ValueError('TTL initialization requires a STAM parent without TTL')
    if value.get('stage') != 2:
        raise ValueError('TTL initialization requires a Stage II STAM checkpoint')
    if value.get('model_profile') != config['model_profile']:
        raise ValueError('STAM parent model_profile differs from the configuration')
    _stam_mode(value, config['stam']['mode'])
    return model_from_checkpoint(value), value, sha256(path)


def _validate_stage2_source(value, config, rows, data, stam_parent=False):
    source = 'STAM parent' if stam_parent else 'Original Stage II initialization'
    expected = {'stage': 2, 'model_profile': config['model_profile'], 'normalization': config['normalization']}
    for key in ('intensity_domain', 'calibration_application'):
        expected[key] = rows[0]['metadata'][key]
    for key, target in expected.items():
        if value.get(key) != target:
            raise ValueError(f'{source} {key} missing or incompatible')
    if value.get('split') is None or validate_split(rows, value['split']) != data['split']:
        raise ValueError(f'{source} split missing or incompatible')
    if value.get('data_snapshot', None if stam_parent else data) != data:
        raise ValueError(f'{source} data snapshot missing or differs')
    topologies = sorted({row['metadata']['temporal_topology'] for row in rows})
    known_topologies = value.get('temporal_topologies', topologies if 'data_snapshot' in value else None)
    if known_topologies != topologies:
        raise ValueError(f'{source} temporal topology differs')
    if stam_parent:
        parent_config = value.get('config', {})
        for field in ('loss', 'temporal_roll'):
            if parent_config.get(field) != config[field]:
                raise ValueError(f'STAM parent {field} missing or differs from the configuration')
    elif value.get('stam') is not None:
        raise ValueError('STAM requires original Stage II initialization weights')


def _historical_best(path, saved):
    if not path.is_file():
        raise ValueError('Training checkpoint requires its historical best.pt')
    best = load_checkpoint(path)
    matches = best['best'] == saved['best']
    pending_ttl_best = False
    if saved.get('ttl') is not None:
        matches = matches and best['completed_epoch'] == saved.get('best_epoch')
        # An improving TTL epoch publishes last first, including the winning weights.
        pending_ttl_best = (saved.get('best_epoch') == saved['completed_epoch']
                            and best['completed_epoch'] < saved['completed_epoch']
                            and saved['best'] < best['best'])
    if (best.get('run_id') != saved.get('run_id') or best['signature'] != saved['signature']
            or best['completed_epoch'] > saved['completed_epoch'] or not (matches or pending_ttl_best)):
        raise ValueError('best/last checkpoint lineage mismatch')
    return best


def _validate_ttl_resume(saved, best, config, data, sig):
    from ..models.ttl import architecture

    if saved['data_snapshot'] != data or signature(validate_train(saved['config'], 2), data) != sig:
        raise ValueError('TTL checkpoint data snapshot or resolved configuration differs')
    digest = saved.get('parent_checkpoint_sha256')
    expected = {'architecture': architecture(config['ttl']['mode']),
                'freeze_policy': 'ttl_only', 'param_groups': ['ttl']}
    for value in (saved, best):
        if value.get('ttl') != expected:
            raise ValueError('TTL checkpoint architecture, freeze policy or parameter groups differ')
        source = value.get('initialization', {})
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)
                or value.get('parent_checkpoint_sha256') != digest
                or source.get('sha256') != digest or source.get('kind') != 'stam_parent'):
            raise ValueError('TTL best/last checkpoint parent lineage mismatch')
        groups = value['optimizer']['param_groups']
        if len(groups) != 1 or groups[0].get('name') != 'ttl':
            raise ValueError('TTL checkpoint requires one TTL optimizer group')
    if (saved['next_epoch'] != saved['completed_epoch'] + 1 or not 0 <= best['step'] <= saved['step']
            or not 0 <= best['completed_epoch'] <= saved['completed_epoch']
            or best.get('best_epoch') != best['completed_epoch']
            or (best['completed_epoch'] == 0 and best['step'] != 0)):
        raise ValueError('TTL best/last checkpoint progress mismatch')


def _fork_prefix(path, saved, config, data, stage):
    path = Path(path).resolve()
    if path.name != 'last.pt' or stage != 2 or saved['stage'] != 2:
        raise ValueError('Fork requires a Stage II last.pt and its adjacent best.pt')
    source_stam = saved.get('stam', {})
    target_stam = config['stam']
    if (source_stam.get('architecture', {}).get('mode') != target_stam['mode']
            or target_stam['mode'] == 'off'):
        raise ValueError('Fork source and target STAM modes must match')
    if (not source_stam.get('backbone_frozen') or source_stam.get('param_groups') != ['stam']
            or target_stam['train_policy'] not in ('frozen', 'joint_after_warmup')):
        raise ValueError('Fork needs a frozen STAM prefix and a frozen or warmup target policy')
    boundary = saved['completed_epoch']
    if not 1 <= boundary < config['epochs']['2'] or saved['next_epoch'] != boundary + 1:
        raise ValueError('Fork checkpoint is outside the target training budget')
    if target_stam['train_policy'] == 'joint_after_warmup' and boundary != target_stam['warmup_epochs']:
        raise ValueError('Fork must start at the end of the target warmup')
    if saved['data_snapshot'] != data:
        raise ValueError('Fork parent data snapshot differs')
    source_config = validate_train(saved['config'], stage)
    if signature(source_config, data) != saved['signature']:
        raise ValueError('Fork parent configuration or implementation changed')
    source_config['stam'].update(train_policy=target_stam['train_policy'],
                                 warmup_epochs=target_stam['warmup_epochs'])
    if signature(source_config, data) != signature(config, data):
        raise ValueError('Fork target algorithm or STAM setup differs from the parent')
    return _historical_best(path.with_name('best.pt'), saved), sha256(path)


def train(config, manifest, split, run_dir, stage, resume=None, init=None, stop_after_epoch=None, fork_from=None):
    """Train one stage; resume is guaranteed at completed epoch boundaries only."""
    config=validate_train(config, stage); run_dir=Path(run_dir).resolve()
    if stage not in (1,2): raise ValueError('stage must be 1 or 2')
    if sum(value is not None for value in (resume, init, fork_from)) > 1:
        raise ValueError('resume, fork-from and initialization are mutually exclusive')
    stam_config = config['stam']
    ttl_config = config['ttl']
    ttl_enabled = ttl_config['mode'] != 'off'
    parent_init = init is not None and init.get('kind') == 'stam_parent'
    saved = load_checkpoint(resume or fork_from) if resume or fork_from else None
    if saved and saved['stage']!=stage: raise ValueError('Checkpoint stage mismatch')
    if fork_from and (ttl_enabled or saved.get('ttl') is not None):
        raise ValueError('TTL training supports resume, not --fork-from')
    if parent_init and (not ttl_enabled or stage != 2 or init.get('source_stage') != 2):
        raise ValueError('--init-from-stam requires Stage II and ttl.mode=full or local3')
    if ttl_enabled:
        if saved:
            if saved.get('ttl', {}).get('architecture', {}).get('mode') != ttl_config['mode']:
                raise ValueError('Training checkpoint TTL architecture mismatch')
            stam_config['mode'] = _stam_mode(saved, stam_config['mode'])
            for key in ('z_chunk_size', 'activation_checkpointing'):
                stam_config[key] = saved['config']['stam'][key]
        elif not parent_init:
            raise ValueError('TTL training requires --init-from-stam or --resume')
        else:
            parent_model, parent, parent_digest = load_stam_parent(init['path'], config)
            stam_config.update(mode=parent_model.stam.mode, z_chunk_size=parent_model.stam.z_chunk_size,
                               activation_checkpointing=parent_model.stam.activation_checkpointing)
    elif saved and saved.get('ttl') is not None:
        raise ValueError('Training checkpoint TTL architecture mismatch')
    enabled = stam_config['mode'] != 'off'
    if resume:
        # Migration is supported by moving/copying the whole run, never by injecting
        # one checkpoint into an unrelated destination or losing its historical best.
        if Path(resume).resolve()!=run_dir/'checkpoints/last.pt':
            raise ValueError('Resume requires this run directory\'s checkpoints/last.pt; migrate the entire run directory or use a warm start')
        identity_path=run_dir/'run_identity.json'
        if not identity_path.is_file() or read_json(identity_path).get('run_id')!=saved.get('run_id'):
            raise ValueError('Resume run ownership mismatch/missing run identity')
        historical_best = _historical_best(run_dir/'checkpoints/best.pt', saved)
        run_id=saved['run_id']
    else: run_id=str(uuid.uuid4())
    if not resume and run_dir.exists() and any(run_dir.iterdir()): raise FileExistsError('Training directory is nonempty; use resume or a new run')
    rows=validate_manifest(manifest,split,require_target=True)
    split=validate_split(rows,split)
    training=[r for r in rows if str(r['animal_id']) in split['train']]
    validation=[r for r in rows if str(r['animal_id']) in split['val']]
    if not training or not validation: raise ValueError('Training and validation animals/cases are both required')
    data=snapshot(rows,split); sig=signature(config,data)
    if resume and saved['signature']!=sig: raise ValueError('Resume rejected: data, split, algorithm/config or implementation changed; use an explicit warm start')
    if resume and ttl_enabled:
        _validate_ttl_resume(saved, historical_best, config, data, sig)
    parent_hash = saved.get('parent_checkpoint_sha256') if saved else None
    if fork_from:
        parent_best, parent_hash = _fork_prefix(fork_from, saved, config, data, stage)
    if enabled and rows[0]['metadata']['normalization'] != config['normalization']:
        raise ValueError('STAM training config and data normalization differ')
    device=select_device(config['device']); seed_all(config['seed'],config['deterministic'],config['threads'])
    initialization={'kind':'random','source_stage':None,'sha256':None}
    sam_source = None
    if saved:
        saved_stam = saved.get('stam')
        if bool(saved_stam) != enabled or (enabled and saved_stam['architecture']['mode'] != stam_config['mode']):
            raise ValueError('Training checkpoint STAM architecture mismatch')
        if enabled:
            sam_source = saved_stam.get('sam_source')
        model = model_from_checkpoint(saved)
        initialization=saved['initialization']
    elif ttl_enabled:
        _validate_stage2_source(parent, config, rows, data, stam_parent=True)
        model = parent_model
        attach_ttl(model, ttl_config)
        sam_source = parent['stam'].get('sam_source')
        parent_hash = parent_digest
        initialization = {'kind': 'stam_parent', 'source_stage': 2, 'sha256': parent_hash,
                          'path': str(Path(init['path']).resolve())}
        del parent
    else:
        model=create_model(stage,config['model_profile'])
        if (enabled or (stage == 2 and (config['stage2_start_epoch'] != 1 or config['stage2_include_initial']))) and (
                not init or init['source_stage'] != 2):
            raise ValueError('This configuration requires --init-from-stage2 with original Stage II weights')
        if init is None and stage==1 and config['init_weights']: init={'path':config['init_weights'],'source_stage':1}
        if init:
            value=torch.load(init['path'],map_location='cpu',weights_only=True)
            state=value.get('state_dict',value)
            source_stage=int(init['source_stage'])
            if 'stage' in value and value['stage']!=source_stage: raise ValueError('Initialization stage metadata mismatch')
            if stage==1 and source_stage!=1: raise ValueError('Stage I requires Stage I initialization')
            if enabled or (stage == 2 and (config['stage2_start_epoch'] != 1 or config['stage2_include_initial'])):
                _validate_stage2_source(value, config, rows, data)
            if stage==2 and source_stage==1: initialize_stage2(model,state)
            else: model.load_state_dict(state,strict=True)
            initialization={'kind':'stage_transfer' if stage==2 and source_stage==1 else 'warm_start',
                'source_stage':source_stage,'sha256':sha256(init['path']),'path':str(Path(init['path']).resolve())}
        elif stage==2: raise ValueError('Stage II needs init-from-stage1, init-from-stage2, or resume')
        if enabled:
            sam_source = attach_stam(model, stam_config, load_sam=True)
    model.to(device)
    policy = stam_config['train_policy']
    warmup = stam_config['warmup_epochs']
    prefix_epoch = warmup if policy == 'joint_after_warmup' else 10
    backbone_frozen = enabled and (policy == 'frozen' or (
        policy == 'joint_after_warmup' and (not saved or saved['completed_epoch'] <= warmup)))
    optimizer = make_optimizer(model, config, backbone_frozen)
    if ttl_enabled:
        _check_ttl_optimizer(model, optimizer)
    if enabled and saved:
        names = _group_names(optimizer)
        if (saved_stam['backbone_frozen'] != backbone_frozen or saved_stam['param_groups'] != names
                or [group.get('name') for group in saved['optimizer']['param_groups']] != names
                or (resume and saved_stam['train_policy'] != policy)):
            raise ValueError('Training checkpoint has incompatible STAM optimizer groups or freeze state')
    total_epochs=config['epochs'][str(stage)]
    if config['scheduler'] == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_epochs, power=0.9)
    elif config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config['milestones'][str(stage)], gamma=config['gamma'])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=config['min_lr'])
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and config['amp'])
    start_epoch=config['stage2_start_epoch'] if stage == 2 else 1
    step=0; best=math.inf
    best_epoch = None
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        if ttl_enabled:
            _check_ttl_optimizer(model, optimizer)
        scheduler_state=copy.deepcopy(saved['scheduler'])
        if config['scheduler']=='multistep': scheduler_state['milestones']=Counter(scheduler_state['milestones'])
        scheduler.load_state_dict(scheduler_state); scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['next_epoch']; step=saved['step']; best=saved['best']; restore_rng(saved['rng'])
        best_epoch = saved.get('best_epoch')
    elif start_epoch > 1:
        # Position a fresh optimizer at the matching absolute epoch without fictitious updates.
        scheduler.last_epoch = start_epoch - 1
        rates = scheduler._get_closed_form_lr()
        scheduler._last_lr = rates
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate
    run_dir.mkdir(parents=True,exist_ok=True); ckdir=run_dir/'checkpoints'; ckdir.mkdir(exist_ok=True)
    atomic_json(run_dir/'run_identity.json',{'run_id':run_id,'stage':stage,'signature':sig})
    samples=(saved['validation_samples'] if saved else validation_samples(validation,config,stage))
    # Validation sample identity is part of the checkpoint, including patch shapes/origins.
    atomic_json(run_dir/'resolved_config.json',config); atomic_json(run_dir/'data_snapshot.json',data)
    atomic_json(run_dir/'split.json',split); atomic_json(run_dir/'validation_samples.json',samples)
    atomic_json(run_dir/'environment.json',environment())
    if fork_from:
        for payload, name in ((parent_best, 'best.pt'), (saved, 'last.pt')):
            updated = dict(payload, run_id=run_id, signature=sig, config=config, stage_complete=False,
                           manifest=str(Path(manifest).resolve()), parent_checkpoint_sha256=parent_hash,
                           stam=dict(payload['stam'], train_policy=policy, warmup_epochs=warmup))
            atomic_save(ckdir/name, updated)
    if resume and ttl_enabled and historical_best['best'] != saved['best']:
        atomic_save(ckdir/'best.pt', saved)
        append_jsonl(run_dir/'train.jsonl', {'event': 'recovered_best', 'epoch': best_epoch,
                                            'step': step, 'validation_l1': best})
        if start_epoch > total_epochs:
            atomic_json(run_dir/'status.json', {'stage': stage, 'completed_epoch': total_epochs, 'complete': True})
    if start_epoch>total_epochs:
        return {'stage':stage,'complete':True,'checkpoint':str(ckdir/'best.pt'),'last':str(ckdir/'last.pt'),'epochs':total_epochs}
    length=config['steps_per_epoch'][str(stage)] or (sum(r['shape'][1] for r in training) if stage==1 else 15*len(training))
    if length<1: raise ValueError('Empty training epoch')
    final_epoch=total_epochs if stop_after_epoch is None else min(total_epochs,int(stop_after_epoch))
    if final_epoch < start_epoch:
        raise ValueError('stop-after-epoch is before the next scheduled epoch')
    loss_config={**config['loss'],'normalization_scale':config['normalization']['scale'],'normalization_offset':config['normalization']['offset']}
    if enabled:
        from ..models.stam import architecture
    if ttl_enabled:
        from ..models.ttl import architecture as ttl_architecture

    def checkpoint_payload(epoch):
        payload={'schema_version':1,'checkpoint_kind':'training','run_id':run_id,'state_dict':model.state_dict(),'stage':stage,
            'model_profile':config['model_profile'],'completed_epoch':epoch,'next_epoch':epoch+1,'step':step,
            'stage_complete':epoch==total_epochs,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'best':best,'rng':rng_state(),'config':config,'signature':sig,
            'data_snapshot':data,'validation_samples':samples,'initialization':initialization,
            'manifest':str(Path(manifest).resolve()),'split':split,'environment':environment(),
            'code_fingerprint':code_fingerprint(),'normalization':config['normalization'],
            'intensity_domain':rows[0]['metadata']['intensity_domain'],'calibration_application':rows[0]['metadata']['calibration_application'],
            'temporal_topologies':sorted({row['metadata']['temporal_topology'] for row in rows})}
        if enabled:
            payload['stam'] = {'architecture': architecture(stam_config['mode']), 'train_policy': policy,
                               'warmup_epochs': warmup, 'backbone_frozen': backbone_frozen, 'sam_frozen': True,
                               'param_groups': _group_names(optimizer), 'sam_source': sam_source}
            payload['parent_checkpoint_sha256'] = parent_hash
        if ttl_enabled:
            payload['ttl'] = {'architecture': ttl_architecture(ttl_config['mode']),
                              'freeze_policy': 'ttl_only', 'param_groups': _group_names(optimizer)}
            payload['best_epoch'] = best_epoch
        return payload

    try:
        if not saved and stage == 2 and config['stage2_include_initial']:
            best = validate_model(model,validation,config,stage,samples,device)['l1']
            best_epoch = start_epoch - 1
            atomic_save(ckdir/'best.pt', checkpoint_payload(best_epoch))
            append_jsonl(run_dir/'train.jsonl', {'event': 'initial_validation', 'epoch': start_epoch - 1, 'validation_l1': best})
            if ttl_enabled:
                configure_ttl_training(model)
        for epoch in range(start_epoch,final_epoch+1):
            begin = time.perf_counter()
            if ttl_enabled:
                configure_ttl_training(model)
                _check_ttl_optimizer(model, optimizer)
            else:
                model.train()
            if enabled and not ttl_enabled and backbone_frozen and policy == 'joint_after_warmup' and epoch == warmup + 1:
                _unfreeze_backbone(model, optimizer, scheduler, config)
                backbone_frozen = False
            rates_used = {group['name']: group['lr'] for group in optimizer.param_groups} if enabled else None
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            dataset=PatchDataset(training,config,stage,epoch,length)
            loader=_loader(dataset,config,epoch,stage)
            total_loss=0.; total_l1=0.; skipped=0; valid_ratios=[]; amp_overflows=0; epoch_updates=0; shape_counts=Counter(); animal_counts=Counter()
            optimizer.zero_grad(set_to_none=True)
            for index,batch in enumerate(loader):
                x=batch['input'].to(device); target=batch['target'].to(device); valid=batch['valid'].to(device)
                shape_counts['x'.join(map(str,x.shape[2:]))]+=1; animal_counts[batch['animal_id'][0]]+=1
                with _amp(device,config['amp']): pred=_forward_patch(model,x,batch)
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
            if ttl_enabled:
                configure_ttl_training(model)
            improved=evaluation['l1']<best
            if improved:
                best = evaluation['l1']
                best_epoch = epoch
            lr_used=optimizer.param_groups[0]['lr']; scheduler.step()
            entry={'event':'epoch','epoch':epoch,'stage':stage,'steps':step,'train_l1':total_l1/length,
                'train_total':total_loss/length,'validation_l1':evaluation['l1'],'validation_animal_l1':evaluation['animal_l1'],
                'lr':lr_used,'next_lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-begin,
                'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0.,
                'skipped_ssim':skipped,'amp_overflows':amp_overflows,'optimizer_updates':epoch_updates,
                'sample_count':length,'shape_counts_tzyx':dict(shape_counts),'animal_draw_counts':dict(animal_counts),
                'valid_fraction':sum(valid_ratios)/len(valid_ratios),'validation_cases':evaluation['cases']}
            if enabled:
                entry.update(group_lrs=rates_used, next_group_lrs={group['name']: group['lr'] for group in optimizer.param_groups},
                             backbone_frozen=backbone_frozen)
            payload = checkpoint_payload(epoch)
            if improved and not ttl_enabled:
                atomic_save(ckdir/'best.pt', payload)
            atomic_save(ckdir/'last.pt', payload)
            if improved and ttl_enabled:
                atomic_save(ckdir/'best.pt', payload)
            if enabled and not ttl_enabled and backbone_frozen and epoch == prefix_epoch:
                warmup_dir = ckdir/'warmup'
                atomic_save(warmup_dir/'best.pt', _historical_best(ckdir/'best.pt', payload))
                atomic_save(warmup_dir/'last.pt', payload)
            append_jsonl(run_dir/'train.jsonl',entry); _write_metrics_csv(run_dir/'metrics.csv',entry)
            print(json.dumps(entry,ensure_ascii=False),flush=True)
        atomic_json(run_dir/'status.json',{'stage':stage,'completed_epoch':final_epoch,'complete':final_epoch==total_epochs})
    except Exception as exc:
        atomic_json(run_dir/'error.json',{'error':str(exc),'traceback':traceback.format_exc(),'stage':stage})
        raise
    return {'stage':stage,'complete':final_epoch==total_epochs,'checkpoint':str(ckdir/'best.pt'),'last':str(ckdir/'last.pt'),'epochs':final_epoch}
