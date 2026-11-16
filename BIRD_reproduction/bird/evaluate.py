"""Projection-fit diagnostics and explicit-range volume metrics, not paper scores."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import ProjectionDataset
from .model import BIRD
from .geometry import render_rays


def volume_metrics(prediction,target,data_range,ssim=True):
    prediction,target=np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if (prediction.shape!=target.shape or prediction.ndim!=3 or prediction.size==0
        or not np.isfinite(prediction).all() or not np.isfinite(target).all()
        or not math.isfinite(data_range) or data_range<=0):
        raise ValueError('matching finite [Z,Y,X] volumes and positive explicit data_range required')
    mse=float(np.mean((prediction-target)**2))
    result=dict(mae=float(np.mean(np.abs(prediction-target))),rmse=math.sqrt(mse),
                psnr=float('inf') if mse==0 else 10*math.log10(data_range**2/mse))
    if ssim:
        from skimage.metrics import structural_similarity
        if min(target.shape[1:])<7:
            raise ValueError('SSIM uses an explicit 7x7 window; axial dimensions must be >=7')
        result['ssim_axial_mean']=float(np.mean([structural_similarity(a,b,data_range=data_range,win_size=7)
                                               for a,b in zip(prediction,target)]))
    return result


def projection_fit(checkpoint_path,data_path,device='cpu',batch_size=256):
    if batch_size<1:
        raise ValueError('positive batch_size required')
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    cfg=Config(**state['config']).validate()
    data=ProjectionDataset(data_path)
    if data.sha256!=state['data_sha256']:
        raise ValueError('projection-fit data differs from training data')
    bounds=torch.tensor(data.bounds,device=device)
    report=dict(rays=len(data),step=state['step'],data_sha256=data.sha256,
                note='Training-scan projection fit only; not held-out reconstruction quality.')
    for label in ('initial','trained'):
        torch.manual_seed(cfg.seed)
        model=BIRD(cfg).to(device)
        if label=='trained':
            model.load_state_dict(state['model'])
        model.eval();sse=sae=0.
        with torch.no_grad():
            for start in range(0,len(data),batch_size):
                batch=data.batch(np.arange(start,min(len(data),start+batch_size)),device)
                prediction,_=render_rays(model,batch['origins'],batch['directions'],batch['times'],bounds,
                                         cfg.samples_per_ray,cfg.ray_chunk)
                error=(prediction-batch['measured']).double()
                sse+=error.square().sum().item();sae+=error.abs().sum().item()
        report[label]=dict(rmse=math.sqrt(sse/len(data)),mae=sae/len(data))
        del model
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    output=Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    report=projection_fit(args.checkpoint,args.data,args.device)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
