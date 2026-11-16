"""BIRD journal Eq.8-14,17,23-26."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .encoding import HashGrid, SmallMLP, SpaceTimeEncoder


class ShortcutPredictor(nn.Module):
    def __init__(self,n_in,n_out,width,latent):
        super().__init__()
        self.first=SmallMLP(n_in,latent,width)
        self.second=SmallMLP(n_in+latent,n_out,width)

    def forward(self,z):
        return self.second(torch.cat((z,self.first(z)),-1))


class DVF(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder=SpaceTimeEncoder(cfg)
        self.head=ShortcutPredictor(cfg.latent,3,cfg.width,cfg.latent)
        nn.init.uniform_(self.head.second[-1].weight,-1e-4,1e-4)
        nn.init.zeros_(self.head.second[-1].bias)

    def forward(self,x,t):
        return self.head(self.encoder(x,t))


class BIRD(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg.validate()
        self.backward_dvf=DVF(cfg)
        self.forward_dvf=DVF(cfg)
        self.tp_encoder=HashGrid(3,cfg.levels,cfg.features,cfg.base_resolution,
                                 2*max(cfg.spatial_resolution),cfg.log2_hashmap_size)
        self.ff_encoder=SpaceTimeEncoder(cfg)
        self.tp_head=ShortcutPredictor(self.tp_encoder.output_dim,1,cfg.width,cfg.latent)
        self.final_head=ShortcutPredictor(self.tp_encoder.output_dim+cfg.latent,1,cfg.width,cfg.latent)
        bias=math.log(math.expm1(cfg.initial_attenuation)) if cfg.attenuation_activation=='softplus' else cfg.initial_attenuation
        for head in (self.tp_head,self.final_head):
            nn.init.normal_(head.second[-1].weight,std=.01)
            nn.init.constant_(head.second[-1].bias,bias)

    @staticmethod
    def _time(x,t):
        t=torch.as_tensor(t,device=x.device,dtype=x.dtype)
        if t.numel()==1:
            return t.reshape(1,1).expand(len(x),1)
        if t.shape==(len(x),):
            t=t[:,None]
        if t.shape!=(len(x),1):
            raise ValueError('time must be scalar, [N], or [N,1]')
        return t

    def _attenuation(self,y):
        return F.softplus(y) if self.cfg.attenuation_activation=='softplus' else y

    def topology(self,x,t):
        t=self._time(x,t)
        xs=x+self.backward_dvf(x,t)
        return self._attenuation(self.tp_head(self.tp_encoder(xs)))

    def forward(self,x,t):
        if x.ndim!=2 or x.shape[-1]!=3:
            raise ValueError('x must have shape [N,3]')
        t=self._time(x,t)
        backward=self.backward_dvf(x,t)
        xs=x+backward
        ftp=self.tp_encoder(xs)
        fff=self.ff_encoder(x,t)
        return dict(mu=self._attenuation(self.final_head(torch.cat((ftp,fff),-1))),
                    mu_tp=self._attenuation(self.tp_head(ftp)),backward=backward,static=xs)

    def registration_parameters(self):
        return [p for module in (self.backward_dvf,self.tp_encoder,self.tp_head) for p in module.parameters()]

    def optimizer_groups(self):
        grid=[p for module in self.modules() if isinstance(module,HashGrid) for p in module.tables]
        ids={id(p) for p in grid}
        rest=[p for p in self.parameters() if id(p) not in ids]
        return [dict(params=grid,lr=self.cfg.lr_hash,name='hashgrid'),dict(params=rest,lr=self.cfg.lr_network,name='network')]
