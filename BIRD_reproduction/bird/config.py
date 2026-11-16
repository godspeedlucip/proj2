"""Reported parameters and explicit assumptions for unpublished settings."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import math
from pathlib import Path


@dataclass
class Config:
    levels: int = 16
    features: int = 2
    base_resolution: int = 16
    log2_hashmap_size: int = 19
    spatial_resolution: list[int] = field(default_factory=lambda: [384,384,320])
    temporal_resolution: int = 540
    width: int = 32
    latent: int = 32
    attenuation_activation: str = 'softplus'
    initial_attenuation: float = .02
    lambda_tp: float = .5
    lambda_dm: float = 10.
    lambda_ts: float = .1
    lambda_rgt: float = 10.
    iterations: int = 64000
    rays_per_batch: int = 2048
    samples_per_ray: int = 512
    ray_chunk: int = 16
    reg_chunk: int = 1024
    regularization_points: int | None = None
    lr_hash: float = .03
    lr_network: float = .003
    seed: int = 2026
    checkpoint_every: int = 1000
    log_every: int = 100

    def validate(self):
        for name in ['levels','features','base_resolution','log2_hashmap_size','width','latent',
                     'iterations','rays_per_batch','samples_per_ray','ray_chunk','reg_chunk',
                     'checkpoint_every','log_every']:
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if len(self.spatial_resolution)!=3 or any(not isinstance(n,int) or n<3 for n in self.spatial_resolution):
            raise ValueError('spatial_resolution must contain three integers >= 3')
        if not isinstance(self.temporal_resolution,int) or self.temporal_resolution < 4:
            raise ValueError('temporal_resolution must be >= 4')
        if self.regularization_points is not None and (not isinstance(self.regularization_points,int) or self.regularization_points<1):
            raise ValueError('regularization_points must be null or positive integer')
        for name in ['lambda_tp','lambda_dm','lambda_ts','lambda_rgt','lr_hash','lr_network','initial_attenuation']:
            if not math.isfinite(getattr(self,name)):
                raise ValueError(f'{name} must be finite')
        if not 0 <= self.lambda_tp <= 1 or min(self.lambda_dm,self.lambda_ts,self.lambda_rgt)<0:
            raise ValueError('invalid loss weights')
        if min(self.lr_hash,self.lr_network,self.initial_attenuation)<=0:
            raise ValueError('learning rates and initial_attenuation must be positive')
        if self.attenuation_activation not in ('identity','softplus'):
            raise ValueError('attenuation_activation must be identity or softplus')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig'))).validate()


def small_config():
    return Config(levels=3, features=2, base_resolution=4, log2_hashmap_size=9,
                  spatial_resolution=[16,16,16], temporal_resolution=16, width=16, latent=16,
                  iterations=100, rays_per_batch=64, samples_per_ray=24, ray_chunk=32,
                  reg_chunk=32, regularization_points=32, checkpoint_every=50, log_every=10)
