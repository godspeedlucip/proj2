from __future__ import annotations
from itertools import product
import math
import contextlib
import numpy as np
import torch
from ..models import forward_volume


def starts(length,tile,overlap,alignment):
    if tile<1 or overlap<0 or overlap>=tile or tile%alignment or (tile-overlap)%alignment:
        raise ValueError('Tile and stride must be positive and aligned (z=2,xy=8 for Stage II)')
    padded=max(tile,math.ceil(length/alignment)*alignment)
    positions=list(range(0,padded-tile+1,tile-overlap))
    if positions[-1]!=padded-tile: positions.append(padded-tile)
    return positions


def gaussian_weight(shape):
    axes=[np.exp(-.5*(np.linspace(-1,1,n,dtype=np.float32)/.4)**2) for n in shape]
    weight=axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:]
    return np.maximum(weight,1e-4).astype(np.float32)


def predict_volume(model,cbct,prior,normalization,mode='tiled',tile=(32,128,128),overlap=(16,64,64),device='cpu',amp=True):
    if len(cbct.shape)!=4 or prior.shape!=cbct.shape[1:]: raise ValueError('Expected matching TZYX CBCT and ZYX prior')
    model.eval(); device=torch.device(device)
    t,z,y,x=cbct.shape; scale=normalization['scale']; offset=normalization['offset']
    if mode not in ('tiled','full'): raise ValueError('mode must be tiled or full')
    stage=getattr(model,'stage',2)
    def run(c,p):
        inp=np.stack([(c+offset)/scale,np.broadcast_to((p+offset)/scale,c.shape)])
        tensor=torch.from_numpy(np.ascontiguousarray(inp)).unsqueeze(0).to(device)
        autocast=torch.cuda.amp.autocast() if device.type=='cuda' and amp else contextlib.nullcontext()
        with torch.no_grad(),autocast: output=forward_volume(model,tensor)
        if not torch.isfinite(output).all(): raise FloatingPointError('Nonfinite prediction')
        return output[0,0].float().cpu().numpy()
    if mode=='full':
        if stage==1:
            result=np.empty(cbct.shape,np.float32)
            for zi in range(z): result[:,zi:zi+1]=run(np.asarray(cbct[:,zi:zi+1]),np.asarray(prior[zi:zi+1]))
        else: result=run(np.asarray(cbct,dtype=np.float32),np.asarray(prior,dtype=np.float32))
    else:
        tile=list(tile); overlap=list(overlap)
        if stage==1: tile[0]=1; overlap[0]=0
        alignment=(1 if stage==1 else 2,8,8)
        positions=[starts(n,k,o,a) for n,k,o,a in zip((z,y,x),tile,overlap,alignment)]
        result=np.zeros(cbct.shape,np.float32); denominator=np.zeros((z,y,x),np.float32)
        weight=gaussian_weight(tile)
        for origin in product(*positions):
            slices=tuple(slice(a,min(a+k,n)) for a,k,n in zip(origin,tile,(z,y,x)))
            real=tuple(s.stop-s.start for s in slices)
            c=np.asarray(cbct[(slice(None),)+slices],dtype=np.float32)
            p=np.asarray(prior[slices],dtype=np.float32)
            pads=[(0,k-n) for k,n in zip(tile,real)]
            c=np.pad(c,[(0,0)]+pads,mode='edge'); p=np.pad(p,pads,mode='edge')
            output=run(c,p)
            clipped=tuple(slice(0,n) for n in real); w=weight[clipped]
            result[(slice(None),)+slices]+=output[(slice(None),)+clipped]*w
            denominator[slices]+=w
        if not np.all(denominator>0): raise RuntimeError('Uncovered inference voxels')
        np.divide(result,denominator[None],out=result)
    result*=scale; result-=offset
    return result
