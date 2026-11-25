from __future__ import annotations
import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
import time
import torch
from .config import load_config
from .utils.io import atomic_json, environment, read_json, sha256


def parser():
    p=argparse.ArgumentParser(prog='rstar4d',description='RSTAR4D public-code training, TIGRE simulation, inference and evaluation')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor'); d.add_argument('--skip-tigre',action='store_true'); d.add_argument('--out')
    d=sub.add_parser('inventory-pig'); d.add_argument('--data-root',required=True); d.add_argument('--settings-root'); d.add_argument('--identity-map'); d.add_argument('--out',required=True)
    d=sub.add_parser('validate-settings'); d.add_argument('--config',required=True); d.add_argument('--manifest','--inventory',dest='manifest',required=True)
    d=sub.add_parser('make-phantom'); d.add_argument('--out',required=True); d.add_argument('--views',type=int,default=240); d.add_argument('--phase-counts',nargs=3,type=int,default=[10,11,10]); d.add_argument('--shape',nargs=3,type=int,default=[16,32,32])
    d=sub.add_parser('simulate-pig'); d.add_argument('--config',required=True); d.add_argument('--manifest',required=True); d.add_argument('--out',required=True); d.add_argument('--limit',type=int); d.add_argument('--case-id',action='append')
    d=sub.add_parser('import-paired'); d.add_argument('--manifest',required=True); d.add_argument('--out',required=True)
    d=sub.add_parser('validate-data'); d.add_argument('--manifest',required=True); d.add_argument('--split'); d.add_argument('--allow-no-target',action='store_true')
    d=sub.add_parser('convert-weights'); d.add_argument('--input',required=True); d.add_argument('--out',required=True); d.add_argument('--stage',type=int,choices=[1,2],required=True)
    d.add_argument('--profile',choices=['official_v1','pig_ordered_v1'],default='official_v1'); d.add_argument('--intensity-domain',choices=['mu','calibrated_mu','hu','legacy'],required=True)
    d.add_argument('--normalization-scale',type=float,required=True); d.add_argument('--normalization-offset',type=float,default=0); d.add_argument('--calibration-application',choices=['none','post_fdk','legacy_pre_fdk'],default='none')
    d.add_argument('--split', help='Animal split used to train the original Stage II weights')
    d.add_argument('--temporal-topology', choices=['ordered_states', 'cyclic'], action='append')
    for command in ['train','pipeline']:
        d=sub.add_parser(command); d.add_argument('--config'); d.add_argument('--manifest'); d.add_argument('--split'); d.add_argument('--run-dir')
        if command=='train':
            d.add_argument('--stage',type=int,choices=[1,2]); d.add_argument('--stop-after-epoch',type=int)
            group=d.add_mutually_exclusive_group(); group.add_argument('--resume'); group.add_argument('--init-from-stage1'); group.add_argument('--init-from-stage2'); group.add_argument('--fork-from')
    d=sub.add_parser('infer'); d.add_argument('--checkpoint',required=True); d.add_argument('--manifest',required=True); d.add_argument('--out',required=True)
    d.add_argument('--split'); d.add_argument('--split-name',choices=['train','val','test','all'],default='test'); d.add_argument('--mode',choices=['full','tiled'],default='tiled')
    d.add_argument('--tile',nargs=3,type=int,default=[32,128,128]); d.add_argument('--overlap',nargs=3,type=int,default=[16,64,64]); d.add_argument('--device',default='cuda'); d.add_argument('--fp32',action='store_true'); d.add_argument('--export',nargs='+',choices=['h5','tif','nii'],default=['h5'])
    d=sub.add_parser('evaluate'); d.add_argument('--predictions',required=True); d.add_argument('--manifest',required=True); d.add_argument('--metric-config'); d.add_argument('--projection',action='store_true'); d.add_argument('--out',required=True)
    d.add_argument('--lv-annotations', help='JSONL with trusted, phase-labelled LV masks')
    d.add_argument('--lv-extractor', help='Frozen TorchScript model returning a binary LV mask')
    d.add_argument('--lv-device', default='cpu')
    d=sub.add_parser('compare-stam')
    d.add_argument('--candidate', action='append', required=True)
    d.add_argument('--reference', action='append', required=True)
    d.add_argument('--seed', type=int, action='append', required=True)
    d.add_argument('--motion-tolerance-ml', type=float, default=0.)
    d.add_argument('--out', required=True)
    d=sub.add_parser('benchmark'); d.add_argument('--config'); d.add_argument('--stage',type=int,choices=[1,2],default=2); d.add_argument('--phase-count',type=int,default=11); d.add_argument('--steps',type=int,default=2); d.add_argument('--weights'); d.add_argument('--out')
    return p


def doctor(skip_tigre=False):
    result=environment(); result['torch_smoke']='cpu'
    if torch.cuda.is_available():
        x=torch.ones(8,device='cuda',requires_grad=True); x.square().sum().backward()
        assert torch.isfinite(x.grad).all(); result['torch_smoke']='cuda forward/backward passed'
        result['gpu_total_gib']=torch.cuda.get_device_properties(0).total_memory/1024**3
    if not skip_tigre:
        from .simulation.reference import TigreTool, SeriesSettings
        from .simulation.backend import geometry, project, reconstruct, backend_fingerprint
        import numpy as np
        settings=SeriesSettings(voxel_spacing_multiplier=1,n_detector_rows=64,n_detector_cols=64,d_detector_row_mm=1,d_detector_col_mm=1,
            padding_row_pixels=0,padding_col_pixels=0,fov_output_pixels=32,fov_source_pixels=32,fov_mask_shrink_pixels=0)
        z,y,x=np.indices((32,32,32))-15.5; mu=((x*x+y*y+z*z)<49).astype(np.float32)*.02
        geo=geometry(mu.shape,(1,1,1),settings); angles=TigreTool.make_angles(120,0,360)
        projections=project(mu,geo,angles); reconstructed,_=reconstruct(projections,geo,angles,settings)
        center=float(reconstructed[14:18,14:18,14:18].mean())
        if abs(center/.02-1)>.05: raise RuntimeError('TIGRE water-sphere unit check failed')
        result['tigre']={'status':'actual CUDA Ax/FDK passed','center_mu':center,'backend':backend_fingerprint()}
    return result


def benchmark(config, stage, t, steps, weights=None):
    from .config import validate_train
    from .models import attach_stam, create_model, forward_volume, initialize_stage2, model_from_checkpoint
    from .training.trainer import make_optimizer, select_device
    from .training.losses import compute_loss
    from .utils.seed import seed_all

    if steps < 1 or t < 2:
        raise ValueError('Positive steps and at least two phases required')
    config = validate_train(config, stage)
    device = select_device(config['device'])
    patches = [config['stage1_patch']] if stage == 1 else config['stage2_patches']
    value = torch.load(weights, map_location='cpu', weights_only=True) if weights else None
    if value is not None and (value.get('stage', stage) != stage or
                              value.get('model_profile', config['model_profile']) != config['model_profile']):
        raise ValueError('Benchmark checkpoint stage or model profile differs from the configuration')
    if value is not None and value.get('stam') and config['stam']['mode'] == 'off':
        from_checkpoint = value['config']['stam']
        config = validate_train(dict(config, stam=from_checkpoint), stage)
    stam = config['stam']
    enabled = stam['mode'] != 'off'
    backbone_frozen = enabled and stam['train_policy'] != 'joint'
    results = []
    sam_source = None
    norm = config['normalization']
    loss_config = dict(config['loss'], normalization_scale=norm['scale'], normalization_offset=norm['offset'])

    for patch in patches:
        seed_all(config['seed'], config['deterministic'], config['threads'])
        if value is not None and value.get('stam'):
            if value['stam']['architecture']['mode'] != stam['mode']:
                raise ValueError('Benchmark checkpoint STAM mode differs from the configuration')
            model = model_from_checkpoint(value, stage, config['model_profile'])
            model.stam.z_chunk_size = stam['z_chunk_size']
            model.stam.activation_checkpointing = stam['activation_checkpointing']
            sam_source = value['stam']['sam_source']
        else:
            model = create_model(stage, config['model_profile'])
            if value is not None:
                model.load_state_dict(value.get('state_dict', value), strict=True)
            elif stage == 2:
                initialize_stage2(model, create_model(1, config['model_profile']).state_dict())
            if enabled:
                sam_source = attach_stam(model, stam, load_sam=True)
        model.to(device).train()
        optimizer = make_optimizer(model, config, backbone_frozen)
        use_amp = device.type == 'cuda' and config['amp']
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        x = torch.rand(1, 2, t, *patch, device=device)
        target = torch.rand(1, 1, t, *patch, device=device)
        valid = torch.ones_like(target, dtype=torch.bool)
        before = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device.index)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        updates = overflows = 0
        while updates < steps and overflows < 16:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext():
                pred = forward_volume(model, x)
            loss = compute_loss(pred, target, valid, stage, loss_config)['total']
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite benchmark loss')
            scaler.scale(loss).backward()
            if not use_amp and not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite benchmark gradient')
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                overflows += 1
            else:
                updates += 1
            loss_value = float(loss.detach())
            del pred, loss
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == 'cuda' else None
        changed = {name for name, param in model.named_parameters() if not torch.equal(before[name], param.detach().cpu())}
        backbone_changed = any(not name.startswith('stam.') for name in changed)
        projection_changed = 'stam.output_projection.weight' in changed
        sam_changed = any(name.startswith('stam.sam.') for name in changed)
        expected_update = (projection_changed and not sam_changed and backbone_changed != backbone_frozen
                           if enabled else backbone_changed)
        if updates != steps or not expected_update:
            raise RuntimeError(f'Benchmark update failed: useful={updates}/{steps}, overflows={overflows}, '
                               f'backbone={backbone_changed}, output_projection={projection_changed}, SAM={sam_changed}')
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        parameters = sum(param.numel() for param in model.parameters())
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler, before
        model.eval()
        infer_start = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext():
            pred = forward_volume(model, x)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Nonfinite benchmark inference')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        results.append({
            'shape': [1, 2, t, *patch], 'steps': updates, 'amp_overflows': overflows,
            'seconds': elapsed, 'seconds_per_update': elapsed / updates,
            'inference_seconds': time.perf_counter() - infer_start,
            'parameter_update': expected_update, 'loss': loss_value,
            'parameters': parameters, 'trainable_parameters': trainable,
            'backbone_changed': backbone_changed, 'output_projection_changed': projection_changed,
            'sam_changed': sam_changed, 'peak_allocated_gib': peak_allocated, 'peak_reserved_gib': peak_reserved,
            'gpu_free_at_start_gib': free / 1024**3 if device.type == 'cuda' else None,
            'gpu_total_gib': total / 1024**3 if device.type == 'cuda' else None,
        })
        del model, x, target, valid, pred
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return {'environment': environment(), 'stage': stage, 'model_profile': config['model_profile'],
            'config': config, 'weights_sha256': sha256(weights) if weights else None,
            'sam_source': sam_source, 'results': results}

def dispatch(args):
    command=args.command
    if command=='doctor':
        result=doctor(args.skip_tigre)
        if args.out: atomic_json(args.out,result)
        return result
    if command=='inventory-pig':
        from .data.inventory import inventory_pig
        return inventory_pig(args.data_root,args.out,args.settings_root,args.identity_map)
    if command=='validate-settings':
        from .simulation.pipeline import validate_settings
        return {'cases':validate_settings(load_config(args.config,'simulation'),args.manifest)}
    if command=='make-phantom':
        from .simulation.phantom import make_phantom
        return make_phantom(args.out,args.views,args.phase_counts,tuple(args.shape))
    if command=='simulate-pig':
        from .simulation.pipeline import simulate_manifest
        cfg=load_config(args.config,'simulation')
        if args.limit is not None: cfg['limit']=args.limit
        if args.case_id: cfg['case_ids']=args.case_id
        return {'manifest':str(simulate_manifest(args.manifest,cfg,args.out))}
    if command=='import-paired':
        from .data.paired import import_paired
        return {'manifest':str(import_paired(args.manifest,args.out))}
    if command=='validate-data':
        from .data.contracts import validate_manifest
        rows=validate_manifest(args.manifest,args.split,not args.allow_no_target)
        return {'status':'valid','cases':len(rows),'phase_counts':sorted({r['shape'][0] for r in rows})}
    if command=='convert-weights':
        from .models import create_model
        from .training.checkpoint import atomic_save
        if Path(args.out).exists(): raise FileExistsError('Converted weights already exist')
        if args.normalization_scale<=0: raise ValueError('Normalization scale must be positive')
        value=torch.load(args.input,map_location='cpu',weights_only=True); state=value.get('state_dict',value)
        model=create_model(args.stage,args.profile); model.load_state_dict(state,strict=True)
        payload={'schema_version':1,'checkpoint_kind':'weights','state_dict':model.state_dict(),'stage':args.stage,
            'model_profile':args.profile,'normalization':{'scale':args.normalization_scale,'offset':args.normalization_offset},
            'intensity_domain':args.intensity_domain,'calibration_application':args.calibration_application,
            'source_sha256':sha256(args.input),'domain_assumption':'explicitly supplied during conversion'}
        if args.split:
            from .data.contracts import validate_split
            payload['split'] = validate_split([], read_json(args.split))
        if args.temporal_topology:
            payload['temporal_topologies'] = sorted(set(args.temporal_topology))
        atomic_save(args.out,payload)
        return {'checkpoint':args.out,'parameters':sum(p.numel() for p in model.parameters())}
    if command in ('train','pipeline'):
        from .training.trainer import train
        from .training.pipeline import pipeline
        from .training.checkpoint import load_checkpoint
        if command=='train' and args.resume:
            ck=load_checkpoint(args.resume); cfg=load_config(args.config) if args.config else ck['config']
            return train(cfg,args.manifest or ck['manifest'],args.split or ck['split'],args.run_dir or Path(args.resume).resolve().parents[1],
                args.stage or ck['stage'],resume=args.resume,stop_after_epoch=args.stop_after_epoch)
        if not all([args.manifest,args.split,args.run_dir]): raise ValueError('manifest, split and run-dir are required for new training')
        cfg=load_config(args.config)
        if command=='pipeline': return pipeline(cfg,args.manifest,args.split,args.run_dir)
        if args.stage is None: raise ValueError('stage is required for new single-stage training')
        init=None
        if args.init_from_stage1: init={'path':args.init_from_stage1,'source_stage':1}
        if args.init_from_stage2: init={'path':args.init_from_stage2,'source_stage':2}
        return train(cfg,args.manifest,args.split,args.run_dir,args.stage,init=init,stop_after_epoch=args.stop_after_epoch,fork_from=args.fork_from)
    if command=='infer':
        from .inference.predict import predict_manifest
        return {'manifest':str(predict_manifest(args.checkpoint,args.manifest,args.out,args.split_name,args.mode,args.tile,args.overlap,
            args.device,not args.fp32,args.export,args.split))}
    if command=='evaluate':
        from .evaluation.report import evaluate_predictions
        cfg=load_config(args.metric_config,'metrics')
        if args.projection: cfg['projection']=True
        return evaluate_predictions(args.predictions,args.manifest,cfg,args.out,
                                    args.lv_annotations,args.lv_extractor,args.lv_device)
    if command=='compare-stam':
        from .evaluation.stam_comparison import compare_stam
        result = compare_stam(args.candidate, args.reference, args.seed, args.motion_tolerance_ml)
        atomic_json(args.out, result)
        return result
    if command=='benchmark':
        result=benchmark(load_config(args.config),args.stage,args.phase_count,args.steps,args.weights)
        if args.out: atomic_json(args.out,result)
        return result
    raise ValueError('Unhandled command')


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        result=dispatch(args); print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True); return 0
    except Exception as exc:
        print(json.dumps({'status':'error','type':type(exc).__name__,'message':str(exc)},ensure_ascii=False),file=sys.stderr,flush=True)
        return 2


if __name__=='__main__': raise SystemExit(main())
