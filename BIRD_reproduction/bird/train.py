"""Single-scan BIRD optimization; registration gradients only reach forward DVF."""
import argparse
import json
import time
import shutil
from pathlib import Path
import torch
from .config import Config
from .data import ProjectionDataset
from .model import BIRD
from .geometry import render_rays
from .losses import data_fidelity, spatial_regularizer, temporal_regularizer, registration_loss


def sample_coordinates(count,generator,device,spatial_margin=None,temporal_margin=0.):
    q=torch.rand(count,4,generator=generator).to(device)
    if spatial_margin is not None:
        h=torch.tensor(spatial_margin,device=device)
        q[:,:3]=h+q[:,:3]*(1-2*h)
    q[:,3:]=temporal_margin+q[:,3:]*(1-2*temporal_margin)
    return q[:,:3],q[:,3:]


def train(data_path,cfg,outdir,device='cpu',resume=False,stop_after=None):
    cfg.validate()
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if stop_after is not None and not 1<=stop_after<=cfg.iterations:
        raise ValueError('stop_after must lie within configured iteration count')
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    checkpoint_path=outdir/'checkpoint.pt'
    if checkpoint_path.exists() and not resume:
        raise FileExistsError('checkpoint exists; use resume or a new output directory')
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    dataset=ProjectionDataset(data_path)
    torch.manual_seed(cfg.seed)
    generator=torch.Generator().manual_seed(cfg.seed+1)
    model=BIRD(cfg).to(device)
    optimizer=torch.optim.Adam(model.optimizer_groups(),betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.iterations,eta_min=0.)
    start=0
    if resume:
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if state['config']!=cfg.to_dict():
            raise ValueError('resume config differs from checkpoint')
        if state['data_sha256']!=dataset.sha256:
            raise ValueError('resume data SHA-256 differs from checkpoint')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start=state['step']
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['sampling_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    metadata=dict(config=cfg.to_dict(),data_path=str(Path(data_path).resolve()),data_sha256=dataset.sha256,
                  bounds_mm=dataset.bounds.tolist(),time_bounds_s=dataset.time_bounds.tolist(),
                  torch_version=str(torch.__version__),device=str(device))
    (outdir/'resolved.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    bounds=torch.tensor(dataset.bounds,device=device)
    steps=[1/n for n in cfg.spatial_resolution]
    dt=1/(cfg.temporal_resolution-1)
    points=cfg.regularization_points or max(1,cfg.rays_per_batch*cfg.samples_per_ray//4)
    end=cfg.iterations if stop_after is None else stop_after

    def save(step):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   config=cfg.to_dict(),step=step,data_sha256=dataset.sha256,
                   bounds_mm=dataset.bounds.tolist(),time_bounds_s=dataset.time_bounds.tolist(),
                   torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                   torch_version=str(torch.__version__))
        temporary=outdir/'checkpoint.tmp'
        torch.save(state,temporary)
        temporary.replace(checkpoint_path)

    model.train()
    history_path=outdir/'history.jsonl'
    if resume and history_path.exists():
        # A crash can leave log rows newer than the last committed checkpoint,
        # including a partial last JSON line. Keep an audit copy before repair.
        original=history_path.read_text(encoding='utf-8')
        retained=[]
        for line in original.splitlines():
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row,dict) and isinstance(row.get('step'),int) and row['step']<=start:
                retained.append(line)
        repaired=''.join(line+'\n' for line in retained)
        if repaired!=original:
            shutil.copyfile(history_path,outdir/f'history.before-recovery-{time.time_ns()}.jsonl')
            history_path.write_text(repaired,encoding='utf-8')
    with history_path.open('a' if resume else 'w',encoding='utf-8') as log:
        for step in range(start,end):
            begin=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            indices=torch.randint(len(dataset),(cfg.rays_per_batch,),generator=generator).numpy()
            batch=dataset.batch(indices,device)
            pred,tp=render_rays(model,batch['origins'],batch['directions'],batch['times'],bounds,
                                cfg.samples_per_ray,cfg.ray_chunk,checkpoint_chunks=True)
            loss=data_fidelity(pred,tp,batch['measured'],batch['ids'],cfg.lambda_tp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite fidelity loss at step {step+1}')
            loss.backward()
            record=dict(fdl=loss.item(),dm=0.,ts=0.,rgt=0.)
            # Each loss has an independent batch. Chunk sums implement a point mean.
            for name,weight in [('dm',cfg.lambda_dm),('ts',cfg.lambda_ts),('rgt',cfg.lambda_rgt)]:
                if weight==0:
                    continue
                for offset in range(0,points,cfg.reg_chunk):
                    count=min(cfg.reg_chunk,points-offset)
                    x,t=sample_coordinates(count,generator,device,steps if name=='dm' else None,dt if name=='ts' else 0.)
                    if name=='dm':
                        value=spatial_regularizer(model.forward_dvf,model.backward_dvf,x,t,steps)
                    elif name=='ts':
                        value=temporal_regularizer(model.forward_dvf,x,t,dt)
                    else:
                        value=registration_loss(model,x,t)
                    if not torch.isfinite(value):
                        raise FloatingPointError(f'nonfinite {name} loss at step {step+1}')
                    fraction=count/points
                    (weight*fraction*value).backward()
                    record[name]+=fraction*value.item()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError(f'nonfinite gradient at step {step+1}')
            record.update(total=record['fdl']+cfg.lambda_dm*record['dm']+cfg.lambda_ts*record['ts']+cfg.lambda_rgt*record['rgt'],
                          step=step+1,lr_hash=optimizer.param_groups[0]['lr'],lr_network=optimizer.param_groups[1]['lr'],
                          seconds=time.perf_counter()-begin)
            optimizer.step();scheduler.step()
            log.write(json.dumps(record)+'\n');log.flush()
            if (step+1)%cfg.log_every==0 or step==start:
                print(f"step={step+1} total={record['total']:.6g} fdl={record['fdl']:.6g} dm={record['dm']:.3g} ts={record['ts']:.3g} rgt={record['rgt']:.3g}",flush=True)
            if (step+1)%cfg.checkpoint_every==0 or step+1==end:
                save(step+1)
    return checkpoint_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    print(train(args.data,Config.load(args.config),args.out,args.device,args.resume,args.stop_after))


if __name__=='__main__':
    main()
