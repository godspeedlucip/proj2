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
from .utils.io import atomic_json, environment, sha256


def parser():
    p=argparse.ArgumentParser(prog='rstar4d',description='RSTAR4D public-code training, TIGRE simulation, inference and evaluation')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor'); d.add_argument('--skip-tigre',action='store_true'); d.add_argument('--out')
    d=sub.add_parser('inventory-pig'); d.add_argument('--data-root',required=True); d.add_argument('--settings-root'); d.add_argument('--identity-map'); d.add_argument('--out',required=True)
    d=sub.add_parser('validate-settings'); d.add_argument('--config',required=True); d.add_argument('--manifest','--inventory',dest='manifest',required=True)
    d=sub.add_parser('make-phantom'); d.add_argument('--out',required=True); d.add_argument('--views',type=int,default=240); d.add_argument('--phase-counts',nargs=3,type=int,default=[10,11,10]); d.add_argument('--shape',nargs=3,type=int,default=[16,32,32])
    d=sub.add_parser('simulate-pig'); d.add_argument('--config',required=True); d.add_argument('--manifest',required=True); d.add_argument('--out',required=True); d.add_argument('--limit',type=int); d.add_argument('--case-id',action='append')
    d=sub.add_parser('import-paired'); d.add_argument('--manifest',required=True); d.add_argument('--out',required=True)
    d=sub.add_parser('prepare-cardiac-annotations')
    d.add_argument('--inventory',required=True); d.add_argument('--manifest',required=True)
    d.add_argument('--annotations',required=True); d.add_argument('--out',required=True)
    d.add_argument('--half-width-vox',type=float,default=2.); d.add_argument('--split')
    d=sub.add_parser('validate-data'); d.add_argument('--manifest',required=True); d.add_argument('--split'); d.add_argument('--allow-no-target',action='store_true')
    d=sub.add_parser('convert-weights'); d.add_argument('--input',required=True); d.add_argument('--out',required=True); d.add_argument('--stage',type=int,choices=[1,2],required=True)
    d.add_argument('--profile',choices=['official_v1','pig_ordered_v1'],default='official_v1'); d.add_argument('--intensity-domain',choices=['mu','calibrated_mu','hu','legacy'],required=True)
    d.add_argument('--normalization-scale',type=float,required=True); d.add_argument('--normalization-offset',type=float,default=0); d.add_argument('--calibration-application',choices=['none','post_fdk','legacy_pre_fdk'],default='none')
    for command in ['train','pipeline']:
        d=sub.add_parser(command); d.add_argument('--config'); d.add_argument('--manifest'); d.add_argument('--split'); d.add_argument('--run-dir')
        if command=='train':
            d.add_argument('--stage',type=int,choices=[1,2]); d.add_argument('--stop-after-epoch',type=int)
            group=d.add_mutually_exclusive_group()
            group.add_argument('--resume',metavar='CHECKPOINT',
                               help='Resume last.pt or the latest epoch in place. An older epoch-NNNN.pt requires --run-dir for a new run.')
            group.add_argument('--init-from-stage1'); group.add_argument('--init-from-stage2')
    d=sub.add_parser('infer'); d.add_argument('--checkpoint',required=True); d.add_argument('--manifest',required=True); d.add_argument('--out',required=True)
    d.add_argument('--split'); d.add_argument('--split-name',choices=['train','val','test','all'],default='test'); d.add_argument('--mode',choices=['full','tiled'],default='tiled')
    d.add_argument('--tile',nargs=3,type=int); d.add_argument('--overlap',nargs=3,type=int); d.add_argument('--device',default='cuda'); d.add_argument('--fp32',action='store_true'); d.add_argument('--export',nargs='+',choices=['h5','tif','nii'],default=['h5'])
    d=sub.add_parser('evaluate'); d.add_argument('--predictions',required=True); d.add_argument('--manifest',required=True); d.add_argument('--metric-config'); d.add_argument('--projection',action='store_true'); d.add_argument('--out',required=True); d.add_argument('--contours')
    d=sub.add_parser('benchmark'); d.add_argument('--config'); d.add_argument('--stage',type=int,choices=[1,2],default=2); d.add_argument('--phase-count',type=int,default=11); d.add_argument('--steps',type=int,default=2); d.add_argument('--weights'); d.add_argument('--out'); d.add_argument('--manifest'); d.add_argument('--split'); d.add_argument('--memory-limit-gib',type=float)
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


def benchmark(config,stage,t,steps,weights=None):
    from .models import create_model, forward_volume, initialize_stage2
    from .training.trainer import select_device
    from .training.losses import compute_loss
    from .utils.seed import seed_all
    if steps<1 or t<2: raise ValueError('Positive steps and at least two phases required')
    device=select_device(config['device']); results=[]
    patches=[config['stage1_patch']] if stage==1 else config['stage2_patches']
    for patch in patches:
        seed_all(config['seed'],config['deterministic'],config['threads'])
        model=create_model(stage,config['model_profile']).to(device)
        if weights:
            value=torch.load(weights,map_location='cpu',weights_only=True); model.load_state_dict(value.get('state_dict',value),strict=True)
        elif stage==2: initialize_stage2(model,create_model(1,config['model_profile']).state_dict())
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
        scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and config['amp'])
        x=torch.rand(1,2,t,*patch,device=device); target=torch.rand(1,1,t,*patch,device=device); valid=torch.ones_like(target,dtype=torch.bool)
        before=model.conv1.conv1.weight.detach().clone()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter(); loss_value=None
        for i in range(steps):
            optimizer.zero_grad(set_to_none=True)
            ctx=torch.cuda.amp.autocast() if device.type=='cuda' and config['amp'] else contextlib.nullcontext()
            with ctx: pred=forward_volume(model,x)
            loss=compute_loss(pred,target,valid,stage,config['loss'])['total']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite benchmark loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); loss_value=float(loss.detach())
        if device.type=='cuda': torch.cuda.synchronize(device)
        changed=not torch.equal(before,model.conv1.conv1.weight)
        if not changed: raise RuntimeError('Benchmark made no parameter update (AMP overflow); use --weights or explicit fp32 config')
        results.append({'shape':[1,2,t,*patch],'steps':steps,'seconds':time.perf_counter()-start,'parameter_update':changed,
            'loss':loss_value,'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None,
            'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3 if device.type=='cuda' else None})
        del model,optimizer,scaler,x,target,valid,before,pred,loss
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    return {'environment':environment(),'stage':stage,'model_profile':config['model_profile'],'config':config,'results':results}


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
    if command=='prepare-cardiac-annotations':
        from .data.cardiac_annotations import prepare_manifest
        return {'manifest':str(prepare_manifest(args.inventory,args.manifest,args.annotations,args.out,args.half_width_vox,args.split))}
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
        atomic_save(args.out,{'schema_version':1,'checkpoint_kind':'weights','state_dict':model.state_dict(),'stage':args.stage,
            'model_profile':args.profile,'normalization':{'scale':args.normalization_scale,'offset':args.normalization_offset},
            'intensity_domain':args.intensity_domain,'calibration_application':args.calibration_application,
            'source_sha256':sha256(args.input),'domain_assumption':'explicitly supplied during conversion'})
        return {'checkpoint':args.out,'parameters':sum(p.numel() for p in model.parameters())}
    if command in ('train','pipeline'):
        from .training.trainer import train
        from .training.pipeline import pipeline
        from .training.checkpoint import resume_defaults
        if command=='train' and args.resume:
            ck=resume_defaults(args.resume); cfg=load_config(args.config) if args.config else ck['config']
            return train(cfg,args.manifest or ck['manifest'],args.split or ck['split'],args.run_dir or Path(args.resume).resolve().parents[1],
                args.stage or ck['stage'],resume=args.resume,stop_after_epoch=args.stop_after_epoch)
        if not all([args.manifest,args.split,args.run_dir]): raise ValueError('manifest, split and run-dir are required for new training')
        cfg=load_config(args.config)
        if command=='pipeline': return pipeline(cfg,args.manifest,args.split,args.run_dir)
        if args.stage is None: raise ValueError('stage is required for new single-stage training')
        init=None
        if args.init_from_stage1: init={'path':args.init_from_stage1,'source_stage':1}
        if args.init_from_stage2: init={'path':args.init_from_stage2,'source_stage':2}
        return train(cfg,args.manifest,args.split,args.run_dir,args.stage,init=init,stop_after_epoch=args.stop_after_epoch)
    if command=='infer':
        from .inference.predict import predict_manifest
        return {'manifest':str(predict_manifest(args.checkpoint,args.manifest,args.out,args.split_name,args.mode,args.tile,args.overlap,
            args.device,not args.fp32,args.export,args.split))}
    if command=='evaluate':
        from .evaluation.report import evaluate_predictions
        cfg=load_config(args.metric_config,'metrics')
        if args.projection: cfg['projection']=True
        return evaluate_predictions(args.predictions,args.manifest,cfg,args.out,contours=args.contours)
    if command=='benchmark':
        config=load_config(args.config)
        if args.manifest:
            if args.stage!=2:
                raise ValueError('Real manifest benchmark is only available for Stage II')
            from .training.cardiac_benchmark import benchmark_manifest
            result=benchmark_manifest(config,args.manifest,args.split,args.steps,args.weights,args.memory_limit_gib)
        else:
            result=benchmark(config,args.stage,args.phase_count,args.steps,args.weights)
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
