"""Export raw-mu BIRD predictions on the exact pre-crop GT FDK grid.

No GT image values are read. A user's GT validity mask and post-FDK calibration
must be applied explicitly during evaluation, in the same domain for both images.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .model import BIRD
from .data import file_sha256
from .tigre_backend import validate_case_metadata


def postprocess_gt(volume,grid):
    from scipy.ndimage import zoom
    spacing=np.array(grid['spacing_zyx_mm']);nz,ny,nx=volume.shape
    object_fov=np.array(grid['object_fov_vu_mm'])
    shrink=grid['mask_shrink_pixels']
    radii=object_fov[1]/2-shrink/2*spacing[1:]
    if (radii<=0).any():raise ValueError('nonpositive FOV mask radius')
    z=(np.arange(nz)-(nz-1)/2)*spacing[0]
    y=(np.arange(ny)-(ny-1)/2)*spacing[1]
    x=(np.arange(nx)-(nx-1)/2)*spacing[2]
    mask=(np.abs(z)[:,None,None]<=object_fov[0]/2)&((y[None,:,None]/radii[0])**2+(x[None,None,:]/radii[1])**2<=1)
    output=np.where(mask,volume,0).astype(np.float32)
    width=grid['output_shape_zyx'][1]
    if grid['postprocess']=='scipy_zoom_order1_grid_mode_false':
        output=zoom(output,(1.,width/ny,width/nx),order=1,mode='nearest',prefilter=False)
    else:
        ys,xs=(ny-width)//2,(nx-width)//2
        output=output[:,ys:ys+width,xs:xs+width]
    if list(output.shape)!=grid['output_shape_zyx']:
        raise ValueError('export grid differs from GT shape')
    return output


def export_gt_grid(checkpoint_path,case_path,outdir,times_s,chunk=65536,device='cuda'):
    case=json.loads(Path(case_path).read_text(encoding='utf-8'))
    validate_case_metadata(case)
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if state.get('case_sha256')!=file_sha256(case_path):
        raise ValueError('checkpoint/case mismatch')
    duration=case['acquisition']['duration_s'];times=np.asarray(times_s,dtype=np.float64)
    if times.ndim!=1 or len(times)<1 or not np.isfinite(times).all() or (times<0).any() or (times>=duration).any():
        raise ValueError('times must be within [0,scan_duration) seconds')
    if chunk<1:raise ValueError('chunk must be positive')
    outdir=Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):raise FileExistsError(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    model=BIRD(Config(**state['config'])).to(device);model.load_state_dict(state['model']);model.eval()
    grid=case['evaluation_grid'];nz,ny,nx=grid['intermediate_shape_zyx'];total=nz*ny*nx
    spacing=torch.tensor(grid['spacing_zyx_mm'][::-1],device=device,dtype=torch.float32)
    bounds=torch.tensor(state['bounds_mm'],device=device,dtype=torch.float32)
    records=[]
    with torch.no_grad():
        for phase,seconds in enumerate(times):
            volume=np.empty((nz,ny,nx),np.float32)
            for start in range(0,total,chunk):
                stop=min(start+chunk,total);idx=torch.arange(start,stop,device=device)
                coords=torch.stack((idx%nx-(nx-1)/2,(idx//nx)%ny-(ny-1)/2,idx//(nx*ny)-(nz-1)/2),-1).float()
                world=coords*spacing
                xyz=(world-bounds[0])/(bounds[1]-bounds[0])
                inside=((xyz>=0)&(xyz<=1)).all(-1)
                mu=model(xyz,float(seconds/duration))['mu'][:,0]
                volume.reshape(-1)[start:stop]=torch.where(inside,mu,torch.zeros_like(mu)).cpu().numpy()
            output=postprocess_gt(volume,grid)
            path=outdir/f'mu_{phase:03d}.npy';np.save(path,output,allow_pickle=False)
            records.append(dict(path=path.name,time_s=float(seconds),shape_zyx=list(output.shape),finite=bool(np.isfinite(output).all())))
            print(f'exported {path.name}: time={seconds:g}s shape={output.shape}',flush=True)
    metadata=dict(case_sha256=state['case_sha256'],grid=grid,records=records,
                  intensity='raw_mu_mm^-1',gt_validity_mask_applied=False,polynomial_calibration_applied=False,
                  note='Apply the same validity mask and explicitly matched intensity calibration for GT evaluation.')
    (outdir/'export.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return outdir/'export.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--case',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--times',nargs='+',type=float,required=True)
    parser.add_argument('--chunk',type=int,default=65536);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(2)
    print(export_gt_grid(args.checkpoint,args.case,args.out,args.times,args.chunk,args.device))
