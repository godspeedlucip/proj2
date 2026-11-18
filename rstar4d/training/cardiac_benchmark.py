"""Measure Stage II with the real dataset, objective and Adam updates."""
import gc
import json
import time

import torch

from ..config import resolve_stage2
from ..data.contracts import validate_manifest, validate_split
from ..data.context import plan_window
from ..data.sampling import InvalidCore, PatchDataset, validation_samples
from ..models import create_model, attach_stage2_extensions, initialize_stage2
from ..utils.io import environment
from ..utils.seed import seed_all
from .trainer import _loader, compute_batch, select_device, warm_start_stage2


def _measure(config, resolved, batch, device, steps, weights):
    model = create_model(2,config['model_profile'])
    if weights is not None:
        saved = torch.load(weights,map_location='cpu',weights_only=True)
        state = saved.get('state_dict',saved)
        if saved.get('stage',1)==1:
            initialize_stage2(model,state)
            attach_stage2_extensions(model,resolved)
        else:
            warm_start_stage2(model,state,saved.get('stage2_resolved'),resolved)
    else:
        initialize_stage2(model,create_model(1,config['model_profile']).state_dict())
        attach_stage2_extensions(model,resolved)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],betas=config['betas'],
                                 weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and config['amp'])
    updates = measured_updates = 0
    loss_config = {**config['loss'],'normalization_scale':config['normalization']['scale'],
                   'normalization_offset':config['normalization']['offset']}
    timings = []
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss,counts = compute_batch(model,batch,device,2,resolved,loss_config,
                                    resolved['boundary']['ramp_epochs'],config['amp'])
        scaler.scale(loss['total']).backward()
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = int(scaler.get_scale()>=before)
        updates += updated
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        if step>0:
            timings.append(time.perf_counter()-start)
            measured_updates += updated
    return {'status':'ok' if updates else 'no_update','optimizer_updates':updates,
            'measured_updates':measured_updates,'seconds_per_step':sum(timings)/len(timings),
            'parameters':sum(p.numel() for p in model.parameters()),
            'sample_diagnostics':counts}


def benchmark_manifest(config, manifest, split, steps, weights=None, memory_limit_gib=None):
    if steps<1:
        raise ValueError('Benchmark needs at least one measured step')
    resolved = resolve_stage2(config,2)
    if resolved['crop_mode']!='core_with_context':
        raise ValueError('Real Stage II benchmark requires core_with_context')
    sidecars = []
    if resolved['prior']['mode']=='geometry':
        sidecars.extend(('projections','angle_summary'))
    if resolved['boundary']['enabled'] or resolved['motion']['surface_weight']:
        sidecars.append('annotations')
    rows = validate_manifest(manifest,require_target=True,full=False,sidecars=sidecars,prefer_angle_summary=True)
    if split is not None:
        train_animals = validate_split(rows,split)['train']
        rows = [row for row in rows if str(row['animal_id']) in train_animals]
    if not rows:
        raise ValueError('No available benchmark training case')
    device = select_device(config['device'])
    if memory_limit_gib is not None:
        if device.type!='cuda' or not 0<memory_limit_gib*2**30<=torch.cuda.get_device_properties(device).total_memory:
            raise ValueError('Invalid per-process CUDA memory limit')
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(memory_limit_gib*2**30/torch.cuda.get_device_properties(device).total_memory,index)
    results = []
    for shape in config['stage2_patches']:
        local = {**config,'stage2_patches':[shape]}
        seed_all(config['seed'],config['deterministic'],config['threads'])
        candidates = [r for r in rows if all(a>=b for a,b in zip(r['shape'][1:],shape))]
        candidates.sort(key=lambda row:(-row['shape'][0],-int(bool(row.get('annotations'))),row['case_id']))
        if not candidates:
            results.append({'core_shape':shape,'status':'no_eligible_case'})
            continue
        largest = candidates[:1]
        samples = validation_samples(largest,{**local,'validation_samples_per_case':4},2)
        dataset = PatchDataset(largest,local,2,0,len(samples),samples)
        batch = None
        for candidate in _loader(dataset,local,0,2):
            if batch is None or candidate['input'].numel()>batch['input'].numel():
                batch = candidate
        volume = largest[0]['shape'][1:]
        mid = [(n-k)//2 for n,k in zip(volume,shape)]
        largest_origin = list(mid)
        for axis,a in enumerate(resolved['recipe']['alignment']):
            def extent(index):
                origin = list(mid)
                origin[axis] = index
                window = plan_window(volume,origin,shape,resolved['recipe'])
                return window.stop[axis]-window.origin[axis]
            positions = range(max(0,mid[axis]-a),min(volume[axis]-shape[axis],mid[axis]+a)+1)
            largest_origin[axis] = max(positions,key=extent)
        interior = [{'case_id':largest[0]['case_id'],'origin':largest_origin,'shape':shape}]
        try:
            candidate = next(iter(_loader(PatchDataset(largest,local,2,0,1,interior),local,0,2)))
        except InvalidCore:
            candidate = None
        if candidate is not None and candidate['input'].numel()>batch['input'].numel():
            batch = candidate
        context = batch['context']
        info = {'case_id':batch['case_id'][0],'core_shape':shape,'phases':batch['input'].shape[2],
                'core_origin':batch['origin'][0].tolist(),'window_origin':context['origin'][0].tolist(),
                'window_shape':list(batch['input'].shape[-3:]),'amp':config['amp'],
                'left_context_vox':context['core_bounds'][0,:,0].tolist(),
                'right_context_vox':(torch.tensor(batch['input'].shape[-3:])-context['core_bounds'][0,:,1]).tolist()}
        if device.type=='cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            info.update(_measure(local,resolved,batch,device,steps,weights))
        except torch.cuda.OutOfMemoryError as exc:
            info.update(status='oom',error=str(exc))
        if device.type=='cuda':
            info['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device)/2**30
            info['peak_reserved_gib'] = torch.cuda.max_memory_reserved(device)/2**30
            torch.cuda.empty_cache()
        results.append(info)
        print(json.dumps(info,ensure_ascii=False,allow_nan=False),flush=True)
        gc.collect()
    return {'environment':environment(),'stage':2,'stage2_resolved':resolved,
            'memory_limit_gib':memory_limit_gib,'results':results}
