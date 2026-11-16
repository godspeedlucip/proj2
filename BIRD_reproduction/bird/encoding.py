"""Differentiable 1D/3D hashgrids, independently implemented in PyTorch."""
import itertools
import math
import torch
from torch import nn


class HashGrid(nn.Module):
    def __init__(self, dimensions, levels=16, features=2, base=16, maximum=768, log2_size=19):
        super().__init__()
        if dimensions not in (1,3):
            raise ValueError('HashGrid supports 1D or 3D')
        if min(levels,features,base,maximum,log2_size)<1:
            raise ValueError('grid sizes must be positive')
        self.dimensions=dimensions
        self.output_dim=levels*features
        scale=(max(base,maximum)/base)**(1/max(levels-1,1))
        self.resolutions=[max(1,int(math.floor(base*scale**i+1e-6))) for i in range(levels)]
        self.tables=nn.ParameterList()
        for resolution in self.resolutions:
            count=min(2**log2_size,(resolution+1)**dimensions)
            p=nn.Parameter(torch.empty(count,features))
            nn.init.uniform_(p,-1e-4,1e-4)
            self.tables.append(p)
        self.register_buffer('corners',torch.tensor(list(itertools.product((0,1),repeat=dimensions)),dtype=torch.long),persistent=False)

    def _index(self, vertices, resolution, count):
        if count == (resolution+1)**self.dimensions:
            v=vertices.remainder(resolution+1)
            index=v[...,0]
            for axis in range(1,self.dimensions):
                index=index+v[...,axis]*(resolution+1)**axis
            return index
        primes=(1,2654435761,805459861)
        index=vertices[...,0]*primes[0]
        for axis in range(1,self.dimensions):
            index=torch.bitwise_xor(index,vertices[...,axis]*primes[axis])
        return index.remainder(count)

    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]!=self.dimensions:
            raise ValueError(f'expected [N,{self.dimensions}] coordinates')
        result=[]
        for resolution,table in zip(self.resolutions,self.tables):
            position=x*resolution
            cell=torch.floor(position).long()
            frac=position-cell.to(x.dtype)
            vertices=cell[:,None,:]+self.corners[None,:,:]
            weights=torch.where(self.corners[None,:,:].bool(),frac[:,None,:],1-frac[:,None,:]).prod(-1)
            values=table[self._index(vertices,resolution,len(table))]
            result.append((weights[:,:,None]*values).sum(1))
        return torch.cat(result,-1)


class SmallMLP(nn.Sequential):
    def __init__(self,n_in,n_out,width):
        super().__init__(nn.Linear(n_in,width),nn.ReLU(),nn.Linear(width,n_out))


class SpaceTimeEncoder(nn.Module):
    """Eq.24-26: refine, spatial/temporal/product pathways, fusion."""
    def __init__(self,cfg):
        super().__init__()
        args=dict(levels=cfg.levels,features=cfg.features,base=cfg.base_resolution,log2_size=cfg.log2_hashmap_size)
        self.space=HashGrid(3,maximum=2*max(cfg.spatial_resolution),**args)
        self.time=HashGrid(1,maximum=2*cfg.temporal_resolution,**args)
        q=cfg.levels*cfg.features
        h,w=cfg.latent,cfg.width
        self.refine_space=SmallMLP(q,h,w)
        self.refine_time=SmallMLP(q,h,w)
        self.space_path=SmallMLP(h,h,w)
        self.time_path=SmallMLP(h,h,w)
        self.product_path=SmallMLP(h,h,w)
        self.fusion=SmallMLP(3*h,h,w)
        self.output_dim=h

    def forward(self,x,t):
        hx=self.refine_space(self.space(x))
        ht=self.refine_time(self.time(t))
        return self.fusion(torch.cat((self.space_path(hx),self.time_path(ht),self.product_path(hx*ht)),-1))
