from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


def image_metrics(pred,target,valid,window=(0.,.05),roi=None,win_size=11,sigma=1.5):
    if pred.shape!=target.shape or valid.shape!=target.shape: raise ValueError('Metric shape mismatch')
    lo,hi=window
    if hi<=lo or win_size%2!=1: raise ValueError('Invalid metric window')
    absolute=0.; squared=0.; count=0; ssim_sum=0.; centers=0
    for t in range(target.shape[0]):
        for z in range(target.shape[1]):
            p=np.asarray(pred[t,z],dtype=np.float64); q=np.asarray(target[t,z],dtype=np.float64)
            m=np.asarray(valid[t,z],dtype=bool)
            region=np.ones_like(m) if roi is None else np.asarray(roi[t,z],dtype=bool)
            if not np.isfinite(p).all() or not np.isfinite(q).all(): raise ValueError('Nonfinite image in evaluation')
            selected=m&region; diff=(p-q)[selected]
            absolute+=float(np.abs(diff).sum()); squared+=float(np.square(diff).sum()); count+=int(selected.sum())
            if min(p.shape)<win_size: continue
            eligible=binary_erosion(m,structure=np.ones((win_size,win_size),bool),border_value=0)&region
            if not eligible.any(): continue
            _,ssim_map=structural_similarity(np.clip(p,lo,hi),np.clip(q,lo,hi),data_range=hi-lo,
                gaussian_weights=True,sigma=sigma,use_sample_covariance=False,win_size=win_size,full=True)
            ssim_sum+=float(ssim_map[eligible].sum()); centers+=int(eligible.sum())
    return {'mae':absolute/count if count else None,'rmse':float(np.sqrt(squared/count)) if count else None,
        'ssim':ssim_sum/centers if centers else None,'valid_voxels':count,'ssim_centers':centers,
        'reason':None if count and centers else ('empty valid ROI' if not count else 'no complete SSIM window')}
