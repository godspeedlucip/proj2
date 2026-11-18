"""Borrow adjacent-state features using backward sampling in output voxels."""
import torch
from torch import nn

from .sampling_grid import FEATURE_STRIDE, feature_mask, grid_zyx, pull3d


def spatial_encoder():
    return nn.Sequential(nn.Conv3d(1,8,3,stride=2,padding=1),nn.LeakyReLU(.2),
                         nn.Conv3d(8,16,3,stride=(1,2,2),padding=1),nn.LeakyReLU(.2))


class AdjacentMotionFusion(nn.Module):
    def __init__(self, mode, max_displacement_vox):
        super().__init__()
        self.mode = mode
        self.max_displacement_vox = tuple(max_displacement_vox)
        self.compress = nn.Conv3d(64,8,1)
        self.gate = nn.Sequential(nn.Conv3d(24,8,1),nn.LeakyReLU(.2),nn.Conv3d(8,1,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,-4.)
        if mode == 'learned':
            self.encoder = spatial_encoder()
            self.flow_head = nn.Sequential(nn.Conv3d(48,24,3,padding=1),nn.LeakyReLU(.2),
                nn.Conv3d(24,16,3,padding=1),nn.LeakyReLU(.2),nn.Conv3d(16,3,1))
            nn.init.normal_(self.flow_head[-1].weight,std=.001)
            nn.init.zeros_(self.flow_head[-1].bias)

    def forward(self, dynamic, features, context, return_aux=False):
        batch,_,phases,*shape = features.shape
        grid = grid_zyx(shape,features.device)
        stride = grid.new_tensor(FEATURE_STRIDE)
        limit = grid.new_tensor(self.max_displacement_vox).view(1,3,1,1,1)
        masks = feature_mask(context['source_fov'])
        if self.mode == 'learned':
            encoded = self.encoder(dynamic[:,0].reshape(batch*phases,1,*dynamic.shape[-3:]))
            encoded = encoded.reshape(batch,phases,16,*shape)
        deltas, pairs, flows, valid_pairs = [], [], [], []
        for t in range(phases):
            target = features[:,:,t]
            target_head = self.compress(target)
            logits = [torch.zeros_like(target[:,:1])]
            residuals = []
            for s in range(max(0,t-1), min(phases,t+2)):
                if s == t:
                    continue
                if self.mode == 'learned':
                    a,b = encoded[:,t],encoded[:,s]
                    flow = limit * self.flow_head(torch.cat((a,b,(a-b).abs()),dim=1)).float().tanh()
                else:
                    flow = grid.new_zeros((batch,3,*shape))
                aligned, valid = pull3d(features[:,:,s],grid[None]+flow.movedim(1,-1)/stride,masks[:,s])
                valid = valid & masks[:,t]
                source_head = self.compress(aligned)
                logit = self.gate(torch.cat((target_head,source_head,(target_head-source_head).abs()),dim=1))
                logits.append(logit.masked_fill(~valid[:,None],-torch.inf))
                residuals.append(aligned-target)
                if return_aux:
                    pairs.append((t,s)); flows.append(flow); valid_pairs.append(valid)
            weights = torch.stack(logits,dim=0).float().softmax(dim=0).to(target.dtype)
            deltas.append(sum(weight*residual for weight,residual in zip(weights[1:],residuals)))
        aux = {}
        if return_aux:
            aux = {'pairs':torch.tensor(pairs,device=features.device),
                   'flow_vox':torch.stack(flows,dim=1),'valid_samples':torch.stack(valid_pairs,dim=1),
                   'origin':context.get('origin'),'stride':FEATURE_STRIDE}
        return torch.stack(deltas,dim=2), aux
