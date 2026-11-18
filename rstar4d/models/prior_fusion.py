"""Local static-prior retrieval with optional projection-geometry conditioning."""
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .motion import spatial_encoder
from .sampling_grid import feature_mask, grid_zyx, pull3d


COARSE = tuple(product((-1,0,1),repeat=3))
FINE = ((0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))


def flat_indices(points, shape):
    return (points[...,0]*shape[1] + points[...,1])*shape[2] + points[...,2]


def match_candidates(query, key, valid, locations):
    shape = valid.shape[-3:]
    last = locations.new_tensor(shape) - 1
    offsets = locations.new_tensor(COARSE)
    scores, masks, indices = [], [], []
    for delta in offsets:
        candidate = locations + delta
        eligible = ((candidate >= 0) & (candidate <= last)).all(dim=-1)
        index = flat_indices(candidate.clamp_min(0).minimum(last),shape)
        cosine = (query * key[:,:,index]).sum(dim=1).clamp(-1,1)
        scores.append(cosine)
        masks.append(eligible[None] & valid.flatten(1)[:,index])
        indices.append(index)
    mask = torch.stack(masks,dim=1)
    raw = torch.stack(scores,dim=1)
    present = mask.any(dim=1)
    masked = raw.masked_fill(~mask,-torch.inf)
    choice = masked.argmax(dim=1,keepdim=True)
    normalized = torch.where(present[:,None],masked/.2,0.).softmax(dim=1)
    winner = torch.stack(indices)[None].expand(query.shape[0],-1,-1).gather(1,choice).squeeze(1)
    return (winner, torch.where(present,raw.gather(1,choice).squeeze(1),0.),
            torch.where(present,normalized.gather(1,choice).squeeze(1),0.),present)


class GeometryPriorFusion(nn.Module):
    def __init__(self, mode, chunk_size):
        super().__init__()
        self.mode = mode
        self.chunk_size = chunk_size
        self.encoder = spatial_encoder()
        self.query = nn.Conv3d(16,8,1)
        self.key = nn.Conv3d(16,8,1)
        self.value = nn.Conv3d(16,16,1)
        self.geometry = nn.Sequential(nn.Linear(14,16),nn.LeakyReLU(.2),nn.Linear(16,8))
        self.offset = nn.Sequential(nn.Conv3d(48,16,3,padding=1),nn.LeakyReLU(.2),nn.Conv3d(16,21,1))
        self.gate = nn.Sequential(nn.Conv3d(57,16,1),nn.LeakyReLU(.2),nn.Conv3d(16,7,1),nn.Sigmoid())
        self.channel_weights = nn.Parameter(torch.full((7,16),1/7))
        self.output = nn.Conv3d(16,64,1,bias=False)
        nn.init.normal_(self.output.weight,std=.001)

    def forward(self, dynamic, static_prior, context):
        batch,_,phases,*_ = dynamic.shape
        static = self.encoder(static_prior)
        shape = static.shape[-3:]
        key = F.normalize(self.key(static).float(),dim=1).flatten(2)
        value = self.value(static)
        source = feature_mask(context['prior_fov'])
        target = feature_mask(context['source_fov'])
        grid = grid_zyx(shape,static.device).reshape(-1,3)
        fine = grid.new_tensor(FINE)
        angles = context['geometry'] if self.mode == 'geometry' else static.new_zeros((batch,phases,14))
        embedded = self.geometry(angles)
        results = []
        observer = context.get('prior_observer')
        for t in range(phases):
            descriptor = self.encoder(dynamic[:,:,t])
            query = F.normalize(self.query(descriptor).float(),dim=1).flatten(2)
            parts = []
            for start in range(0,len(grid),self.chunk_size):
                sl = slice(start,start+self.chunk_size)
                inputs = (query[:,:,sl],key,source,grid[sl].long())
                if self.training and static.is_cuda and torch.is_grad_enabled():
                    parts.append(checkpoint(match_candidates,*inputs,use_reentrant=False))
                else:
                    parts.append(match_candidates(*inputs))
            winner, raw, relative, present = (torch.cat(values,dim=-1) for values in zip(*parts))
            chosen = static.flatten(2).gather(2,winner[:,None].expand(-1,16,-1)).reshape_as(static)
            known = present.reshape(batch,1,*shape) & target[:,t:t+1]
            chosen = torch.where(known,chosen,0.)
            local = torch.cat((descriptor,chosen,(descriptor-chosen).abs()),dim=1)
            embedding = embedded[:,t,:,None,None,None].expand(-1,-1,*shape)
            gate_input = torch.cat((local,embedding,raw.reshape(batch,1,*shape).to(local.dtype)),dim=1)
            gates = self.gate(gate_input).reshape(batch,7,-1)
            offsets = self.offset(local).float().tanh().reshape(batch,7,3,-1).movedim(2,-1)
            centers = torch.stack((winner // (shape[1]*shape[2]),
                                   (winner // shape[2]) % shape[1],winner % shape[2]),dim=-1).float()
            outputs = []
            finite_counts, gate_sums, value_squares = [], [], []
            for start in range(0,len(grid),self.chunk_size):
                sl = slice(start,start+self.chunk_size)
                coords = centers[:,None,sl] + fine[None,:,None] + offsets[:,:,sl]
                samples, fine_valid = pull3d(value,coords,source)
                weighted = (samples * gates[:,None,:,sl] * self.channel_weights.T[None,:,:,None]).sum(dim=2)
                outputs.append(weighted*relative[:,None,sl])
                if observer is not None:
                    finite_counts.append(fine_valid.sum(dim=1))
                    gate_sums.append((gates[:,:,sl]*fine_valid).sum(dim=1))
                    value_squares.append(samples.float().square().mean(dim=1).sum(dim=1))
            combined = torch.cat(outputs,dim=-1).reshape(batch,16,*shape).to(static.dtype)
            correction = torch.where(known,self.output(combined),0.)
            if observer is not None:
                count, gate_sum, square_sum = (torch.cat(parts,dim=-1) for parts in
                                               (finite_counts,gate_sums,value_squares))
                observer(t,{'raw':raw,'relative':relative,'eligible':known.flatten(1),
                            'fine_count':count,'gate_mean':gate_sum/count.clamp_min(1),
                            'value_rms':(square_sum/count.clamp_min(1)).sqrt(),
                            'delta_rms':correction.flatten(2).float().square().mean(dim=1).sqrt()})
            results.append(correction)
        return torch.stack(results,dim=2)
