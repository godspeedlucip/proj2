"""Query mu, TP and backward DVF at arbitrary in-scan times (seconds)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .model import BIRD
from .io import npz_output_path


def reconstruct(checkpoint_path,output,times_s,resolution=None,chunk=65536,device='cpu'):
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    cfg=Config(**state['config']).validate()
    times=np.asarray(times_s,dtype=np.float64)
    tb=np.array(state['time_bounds_s'])
    if times.ndim!=1 or len(times)==0 or not np.isfinite(times).all() or (times<tb[0]).any() or (times>tb[1]).any():
        raise ValueError('inference times must lie within recorded time bounds')
    if 'tigre_options' in state and (times>=tb[1]).any():
        raise ValueError('TIGRE acquisition time excludes the scan endpoint')
    resolution=list(resolution or cfg.spatial_resolution)
    if len(resolution)!=3 or any(not isinstance(n,int) or n<1 for n in resolution) or chunk<1:
        raise ValueError('positive resolution and chunk required')
    output=npz_output_path(output)
    if output.exists():
        raise FileExistsError(output)
    model=BIRD(cfg).to(device);model.load_state_dict(state['model']);model.eval()
    nx,ny,nz=resolution;count=nx*ny*nz
    extent=np.diff(np.array(state['bounds_mm']),axis=0)[0].astype(np.float32)
    # Allocate only outputs; generate coordinates per chunk, not a complete 4D grid.
    mu=np.empty((len(times),nz,ny,nx),dtype=np.float32)
    tp=np.empty_like(mu)
    dvf=np.empty((*mu.shape,3),dtype=np.float32)
    with torch.no_grad():
        for phase,seconds in enumerate(times):
            time=float((seconds-tb[0])/(tb[1]-tb[0]))
            for start in range(0,count,chunk):
                stop=min(count,start+chunk)
                index=torch.arange(start,stop,device=device)
                x=(index%nx+.5)/nx;y=((index//nx)%ny+.5)/ny;z=(index//(nx*ny)+.5)/nz
                xyz=torch.stack((x,y,z),-1).float()
                result=model(xyz,time)
                mu[phase].reshape(-1)[start:stop]=result['mu'][:,0].cpu().numpy()
                tp[phase].reshape(-1)[start:stop]=result['mu_tp'][:,0].cpu().numpy()
                dvf[phase].reshape(-1,3)[start:stop]=result['backward'].cpu().numpy()*extent
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,mu=mu,mu_tp=tp,backward_dvf_mm=dvf,times_s=times,
                        bounds_mm=np.array(state['bounds_mm']),voxel_size_mm=extent/np.array(resolution),
                        axis_order=np.array('T,Z,Y,X'),config_json=np.array(json.dumps(cfg.to_dict())))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--times',type=float,nargs='+',required=True)
    parser.add_argument('--resolution',type=int,nargs=3);parser.add_argument('--chunk',type=int,default=65536)
    parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    print(reconstruct(args.checkpoint,args.out,args.times,args.resolution,args.chunk,args.device))
