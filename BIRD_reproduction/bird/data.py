"""Validated NPZ projections: explicit rays or per-view flat detector geometry."""
from pathlib import Path
import hashlib
import numpy as np
import torch


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


class ProjectionDataset:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path,allow_pickle=False) as file:
            self.arrays={name:np.asarray(file[name]) for name in file.files}
        a=self.arrays
        def require(name,shape=None):
            if name not in a:
                raise ValueError(f'missing {name}')
            value=a[name]
            if not np.issubdtype(value.dtype,np.number) or np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError(f'{name} must be finite real numeric data')
            if shape is not None and value.shape!=shape:
                raise ValueError(f'{name} shape must be {shape}, got {value.shape}')
            return value
        self.bounds=require('bounds_mm',(2,3)).astype(np.float32)
        self.time_bounds=require('time_bounds_s',(2,)).astype(np.float64)
        if np.any(self.bounds[1]<=self.bounds[0]):
            raise ValueError('bounds must have positive axis extents')
        if self.time_bounds[1]<=self.time_bounds[0]:
            raise ValueError('time bounds must increase')
        p=require('projections')
        self.geometry=p.ndim==3
        if self.geometry:
            self.views,self.height,self.width=p.shape
            if min(p.shape)<1:
                raise ValueError('projections must be nonempty')
            for name in ['sources_mm','detector_centers_mm','detector_u_mm','detector_v_mm']:
                require(name,(self.views,3))
            u,v=a['detector_u_mm'],a['detector_v_mm']
            normal=np.cross(u,v)
            if np.any(np.linalg.norm(normal,axis=-1)<=0):
                raise ValueError('detector basis must be noncollinear')
            if np.any(np.abs(np.sum((a['detector_centers_mm']-a['sources_mm'])*normal,-1))<1e-8):
                raise ValueError('source cannot lie in detector plane')
            times=require('times_s',(self.views,))
        elif p.ndim==1 and len(p)>0:
            n=len(p)
            require('origins_mm',(n,3))
            direction=require('directions',(n,3))
            if np.any(np.linalg.norm(direction,axis=-1)<=0):
                raise ValueError('directions cannot have zero length')
            ids=require('ray_ids',(n,))
            if not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or len(np.unique(ids))!=n:
                raise ValueError('ray_ids must be unique nonnegative integers')
            times=require('times_s',(n,))
        else:
            raise ValueError('projections must have shape [N] or [V,H,W]')
        if np.any(times<self.time_bounds[0]) or np.any(times>self.time_bounds[1]):
            raise ValueError('times_s outside time_bounds_s')
        self.projections=p.reshape(-1).astype(np.float32)
        if not np.isfinite(self.projections).all() or not np.isfinite(self.bounds).all():
            raise ValueError('data not finite after float32 conversion')
        self.sha256=file_sha256(path)

    def __len__(self):
        return len(self.projections)

    def batch(self,indices,device):
        indices=np.asarray(indices,dtype=np.int64)
        if indices.ndim!=1 or len(indices)==0 or np.any(indices<0) or np.any(indices>=len(self)):
            raise ValueError('batch indices out of range or empty')
        a=self.arrays
        if self.geometry:
            views=indices//(self.height*self.width)
            pixel=indices%(self.height*self.width)
            row,col=pixel//self.width,pixel%self.width
            origins=a['sources_mm'][views]
            targets=(a['detector_centers_mm'][views]+(col-(self.width-1)/2)[:,None]*a['detector_u_mm'][views]
                     +(row-(self.height-1)/2)[:,None]*a['detector_v_mm'][views])
            directions=targets-origins
            times=a['times_s'][views]
            ids=indices
        else:
            origins=a['origins_mm'][indices]
            directions=a['directions'][indices]
            times=a['times_s'][indices]
            ids=a['ray_ids'][indices]
        # Scale first so a finite float64 vector cannot overflow its squared norm.
        scale=np.abs(directions).max(axis=-1,keepdims=True)
        if not np.isfinite(scale).all() or np.any(scale<=0):
            raise ValueError('invalid direction after geometry evaluation')
        directions=directions/scale
        directions=directions/np.linalg.norm(directions,axis=-1,keepdims=True)
        times=(times-self.time_bounds[0])/(self.time_bounds[1]-self.time_bounds[0])
        def tensor(x):
            with np.errstate(over='ignore',invalid='ignore'):
                array=np.asarray(x,dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError('geometry/time data not finite after float32 conversion')
            return torch.as_tensor(array,device=device)
        return dict(origins=tensor(origins),directions=tensor(directions),times=tensor(times[:,None]),
                    measured=tensor(self.projections[indices]),ids=torch.as_tensor(ids,device=device,dtype=torch.long))
