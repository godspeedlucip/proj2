"""Public loss exactly on fully valid inputs; explicit masked extension."""
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from pytorch_msssim.ssim import _fspecial_gauss_1d


def _squeeze_spatial(x):
    for d in range(x.ndim-1,1,-1): x=x.squeeze(d)
    if x.ndim not in (4,5): raise ValueError(f'Unsupported SSIM input {tuple(x.shape)}')
    return x


def masked_ssim(x,y,valid,data_range=255.,require_xy=False):
    x=_squeeze_spatial(x); y=_squeeze_spatial(y); valid=_squeeze_spatial(valid).bool()
    if require_xy and min(x.shape[-2:])<11: return x.sum()*0,0
    conv=F.conv2d if x.ndim==4 else F.conv3d
    win=_fspecial_gauss_1d(11,1.5).repeat([x.shape[1]]+[1]*(x.ndim-1)).to(x)
    def blur(a):
        for i,size in enumerate(x.shape[2:]):
            if size>=11: a=conv(a,win.transpose(i+2,-1),groups=x.shape[1])
        return a
    support=valid.float()
    for i,size in enumerate(x.shape[2:]):
        if size>=11:
            support=conv(support,torch.ones_like(win).transpose(i+2,-1),groups=x.shape[1])
            support=(support>=11-1e-6).float()
    eligible=support.bool()
    count=int(eligible.sum())
    if count==0: return x.sum()*0,0
    # Invalid pixels cannot receive gradients even in skipped smoothing dimensions.
    x=torch.where(valid,x,y.detach())
    mu_x=blur(x); mu_y=blur(y)
    vx=blur(x*x)-mu_x.square(); vy=blur(y*y)-mu_y.square()
    cov=blur(x*y)-mu_x*mu_y
    c1=(.01*data_range)**2; c2=(.03*data_range)**2
    values=((2*mu_x*mu_y+c1)/(mu_x.square()+mu_y.square()+c1))*((2*cov+c2)/(vx+vy+c2))
    return values[eligible].mean(),count


def compute_loss(pred,target,valid,stage,config):
    pred=pred.float(); target=target.float(); valid=valid.bool()
    if pred.shape!=target.shape or valid.shape!=target.shape: raise ValueError('Loss shape mismatch')
    count=int(valid.sum())
    if count==0: raise ValueError('No valid voxels in loss')
    l1=(pred-target).abs()[valid].mean()
    kind=config.get('kind','public_code'); weight=float(config.get('ssim_weight',.1))
    value=pred.sum()*0; windows=0
    if kind!='l1' and weight:
        if kind=='public_code':
            p=pred if stage==1 else pred.squeeze(1)
            q=target if stage==1 else target.squeeze(1)
            m=valid if stage==1 else valid.squeeze(1)
            if bool(m.all()):
                value=ssim(p*1000,q*1000,data_range=255)
                windows=1
            else: value,windows=masked_ssim(p*1000,q*1000,m)
        elif kind=='physical_ssim':
            # Network units -> original intensity units using the same affine transform.
            scale=config.get('normalization_scale',.02); offset=config.get('normalization_offset',0.)
            lo,hi=config.get('window',[0.,.05])
            if not hi>lo: raise ValueError('Invalid SSIM window')
            p=(pred*scale-offset).clamp(lo,hi).reshape(-1,1,*pred.shape[-2:])
            q=(target*scale-offset).clamp(lo,hi).reshape_as(p)
            m=valid.reshape_as(p)
            value,windows=masked_ssim(p,q,m,data_range=hi-lo,require_xy=True)
        else: raise ValueError(f'Unknown loss {kind}')
    total=l1+(weight*(1-value) if windows else 0.)
    return {'total':total,'l1':l1,'ssim':value if windows else None,'ssim_windows':windows,
            'skipped_ssim':bool(kind!='l1' and weight and not windows),'valid_count':count}
