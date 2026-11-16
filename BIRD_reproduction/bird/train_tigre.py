"""View-based BIRD adaptation: TIGRE projection + streaming INR VJP.

This entrypoint changes the paper's ray-query discretization. Both image heads
are projected with TIGRE; Atb(matched) provides approximate image derivatives.
"""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .model import BIRD
from .losses import data_fidelity,spatial_regularizer,temporal_regularizer,registration_loss
from .train import sample_coordinates
from .tigre_backend import TigreCase,make_geometry,tigre_ax,materialize,volume_vjp


def train_tigre(case_path,cfg,outdir,shape_zyx=None,views_per_step=1,voxel_chunk=32768,
                device='cuda',gpu=0,resume=False,stop_after=None):
    case=TigreCase(case_path)
    shape=list(shape_zyx or cfg.spatial_resolution[::-1])
    if len(shape)!=3 or any(not isinstance(n,int) or n<3 for n in shape):
        raise ValueError('volume shape must be three integers >=3, in Z Y X order')
    cfg=Config(**cfg.to_dict())
    cfg.spatial_resolution=shape[::-1]
    cfg.temporal_resolution=case.metadata['acquisition']['n_views']+1
    cfg.validate()
    if not 1<=views_per_step<=min(len(case),cfg.rays_per_batch) or voxel_chunk<1:
        raise ValueError('invalid views_per_step or voxel_chunk')
    if cfg.regularization_points is None:
        raise ValueError('TIGRE view adaptation requires explicit regularization_points in config')
    end=cfg.iterations if stop_after is None else stop_after
    if not 1<=end<=cfg.iterations:
        raise ValueError('stop_after outside configured iteration count')
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch unavailable; use the installed TIGRE/CUDA environment')
    # Fail before creating run outputs if TIGRE is unavailable.
    geo=make_geometry(case.metadata['geometry'],shape)
    from .tigre_backend import gpu_ids
    gpu_ids(gpu)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    checkpoint_path=outdir/'checkpoint.pt'
    if checkpoint_path.exists() and not resume:
        raise FileExistsError('checkpoint exists; use --resume or a new output directory')
    if resume and not checkpoint_path.exists():raise FileNotFoundError(checkpoint_path)
    torch.manual_seed(cfg.seed);generator=torch.Generator().manual_seed(cfg.seed+1)
    model=BIRD(cfg).to(device)
    optimizer=torch.optim.Adam(model.optimizer_groups(),betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.iterations,eta_min=0.)
    options=dict(backend='TIGRE Siddon / matched approximate adjoint',shape_zyx=shape,
                 views_per_step=views_per_step,voxel_chunk=voxel_chunk,gpu=gpu)
    start=0
    if resume:
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if state['config']!=cfg.to_dict() or state.get('tigre_options')!=options:
            raise ValueError('resume config/options mismatch')
        if state.get('case_sha256')!=case.signature or state.get('data_sha256')!=case.metadata['projections']['sha256']:
            raise ValueError('resume case/data mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);start=state['step']
        torch.set_rng_state(state['torch_rng']);generator.set_state(state['sampling_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    if end<start:raise ValueError('stop_after precedes checkpoint step')
    bounds=case.metadata['geometry']['grid']['bounds_xyz_mm']
    resolved=dict(config=cfg.to_dict(),case_path=str(case.path),case_sha256=case.signature,
                  data_sha256=case.metadata['projections']['sha256'],tigre_options=options,
                  bounds_mm=bounds,time_bounds_s=[0.,case.duration],device=str(device),torch_version=str(torch.__version__),
                  uses_prior_image=False,uses_gt_supervision=False)
    (outdir/'resolved.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2),encoding='utf-8')
    history=outdir/'history.jsonl'
    if resume and history.exists():
        original=history.read_text(encoding='utf-8');retained=[]
        for line in original.splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(row,dict) and isinstance(row.get('step'),int) and row['step']<=start:retained.append(line)
        repaired=''.join(line+'\n' for line in retained)
        if repaired!=original:
            shutil.copyfile(history,outdir/f'history.before-recovery-{time.time_ns()}.jsonl')
            history.write_text(repaired,encoding='utf-8')
    def save(step):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   config=cfg.to_dict(),case_sha256=case.signature,data_sha256=case.metadata['projections']['sha256'],
                   tigre_options=options,step=step,bounds_mm=bounds,time_bounds_s=[0.,case.duration],
                   torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        tmp=outdir/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(checkpoint_path)
    h=[1/n for n in cfg.spatial_resolution];dt=1/(cfg.temporal_resolution-1)
    detector_count=int(np.prod(case.projections.shape[1:]))
    with history.open('a' if resume else 'w',encoding='utf-8') as log:
        for step in range(start,end):
            began=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            selected=torch.randperm(len(case),generator=generator)[:views_per_step].tolist()
            outputs=[];tp_outputs=[];targets=[];ids=[];leaves=[]
            for local,view in enumerate(selected):
                row=case.views[view];t=row['time_s']/case.duration
                volume=materialize(model,t,shape,voxel_chunk,device).requires_grad_()
                angles=np.array([np.deg2rad(row['angle_deg'])],np.float32)
                final=tigre_ax(volume[0],geo,angles,gpu).reshape(-1)
                topology=tigre_ax(volume[1],geo,angles,gpu).reshape(-1)
                count=cfg.rays_per_batch//views_per_step+(local<cfg.rays_per_batch%views_per_step)
                pixel_cpu=torch.randint(detector_count,(count,),generator=generator)
                pixel=pixel_cpu.to(device)
                outputs.append(final[pixel]);tp_outputs.append(topology[pixel])
                targets.append(torch.tensor(case.projections[view].reshape(-1)[pixel_cpu.numpy()],device=device))
                ids.append(row['view_index']*detector_count+pixel)
                leaves.append((t,volume))
            loss=data_fidelity(torch.cat(outputs),torch.cat(tp_outputs),torch.cat(targets),torch.cat(ids),cfg.lambda_tp)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite TIGRE data fidelity')
            loss.backward()
            for t,volume in leaves:
                if volume.grad is None or not bool(torch.isfinite(volume.grad).all()):
                    raise FloatingPointError('missing/nonfinite TIGRE image gradient')
                volume_vjp(model,t,shape,volume.grad,voxel_chunk,device)
            del leaves,outputs,tp_outputs,targets,ids,volume
            record=dict(fdl=loss.item(),dm=0.,ts=0.,rgt=0.)
            for name,weight in [('dm',cfg.lambda_dm),('ts',cfg.lambda_ts),('rgt',cfg.lambda_rgt)]:
                if weight==0:continue
                for offset in range(0,cfg.regularization_points,cfg.reg_chunk):
                    count=min(cfg.reg_chunk,cfg.regularization_points-offset)
                    x,t=sample_coordinates(count,generator,device,h if name=='dm' else None,dt if name=='ts' else 0.)
                    if name=='dm':value=spatial_regularizer(model.forward_dvf,model.backward_dvf,x,t,h)
                    elif name=='ts':value=temporal_regularizer(model.forward_dvf,x,t,dt)
                    else:value=registration_loss(model,x,t)
                    if not bool(torch.isfinite(value)):raise FloatingPointError(f'nonfinite {name}')
                    fraction=count/cfg.regularization_points
                    (weight*fraction*value).backward();record[name]+=fraction*value.item()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite model gradient')
            record.update(total=record['fdl']+cfg.lambda_dm*record['dm']+cfg.lambda_ts*record['ts']+cfg.lambda_rgt*record['rgt'],
                          step=step+1,lr_hash=optimizer.param_groups[0]['lr'],lr_network=optimizer.param_groups[1]['lr'],
                          selected_global_views=[case.views[i]['view_index'] for i in selected])
            optimizer.step();scheduler.step()
            if device.type=='cuda':torch.cuda.synchronize(device)
            record['seconds']=time.perf_counter()-began
            log.write(json.dumps(record)+'\n');log.flush()
            if step==start or (step+1)%cfg.log_every==0:
                print(f"step={step+1} total={record['total']:.6g} fdl={record['fdl']:.6g} views={record['selected_global_views']}",flush=True)
            if (step+1)%cfg.checkpoint_every==0 or step+1==end:save(step+1)
    return checkpoint_path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True);parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--volume-shape',type=int,nargs=3,metavar=('Z','Y','X'))
    parser.add_argument('--views-per-step',type=int,default=1);parser.add_argument('--voxel-chunk',type=int,default=32768)
    parser.add_argument('--device',default='cuda');parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int)
    args=parser.parse_args();torch.set_num_threads(2)
    print(train_tigre(args.case,Config.load(args.config),args.out,args.volume_shape,args.views_per_step,
                     args.voxel_chunk,args.device,args.gpu,args.resume,args.stop_after))
