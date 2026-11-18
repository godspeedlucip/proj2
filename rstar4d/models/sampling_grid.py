"""FP32 ZYX coordinates and validity for differentiable trilinear sampling."""
from itertools import product

import torch
import torch.nn.functional as F


FEATURE_STRIDE = (2,4,4)
CORNERS = tuple(product((0,1), repeat=3))


def grid_zyx(shape, device):
    axes = [torch.arange(length, device=device, dtype=torch.float32) for length in shape]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)


def sampling_valid(coords, valid):
    shape = valid.shape[-3:]
    dims = coords.new_tensor(shape)
    support = ((coords >= 0) & (coords <= dims-1)).all(dim=-1)
    if valid.ndim == 5:
        valid = valid[:,0]
    low = coords.floor().to(torch.int64)
    last = dims.long() - 1
    has_upper = coords != low
    valid_flat = valid.reshape(valid.shape[0], -1)
    for corner in CORNERS:
        offset = low.new_tensor(corner)
        point = (low + offset).clamp_min(0).minimum(last)
        linear = (point[...,0]*shape[1] + point[...,1])*shape[2] + point[...,2]
        used = (has_upper | (offset == 0)).all(dim=-1)
        support &= ~used | valid_flat.gather(1, linear.reshape(len(valid),-1)).reshape_as(support)
    return support


def pull3d(source, coords_zyx, valid=None):
    coords = coords_zyx.float()
    point_shape = coords.shape[1:-1]
    sizes = coords.new_tensor(source.shape[-3:])
    grid = (2*(coords+.5)/sizes-1).flip(-1).reshape(coords.shape[0],1,1,-1,3)
    if grid.shape[-2] == 0:
        return source.new_zeros((source.shape[0],source.shape[1],*point_shape)), coords.new_zeros(coords.shape[:-1],dtype=torch.bool)
    sampled = F.grid_sample(source.float(),grid,mode='bilinear',padding_mode='border',
                            align_corners=False).reshape(source.shape[0],source.shape[1],*point_shape)
    if valid is None:
        support = torch.ones_like(coords[...,0], dtype=torch.bool)
    else:
        support = sampling_valid(coords,valid)
    return torch.where(support[:,None],sampled,0.).to(source.dtype), support


def feature_mask(valid):
    return valid[..., ::2, ::4, ::4]


def flow_to_points(flow, points_local):
    stride = points_local.new_tensor(FEATURE_STRIDE)
    flow = flow.float()
    # Center the samples so a constant displacement stays exact at FOV edges.
    anchor = flow[:,:,:1,:1,:1]
    samples = pull3d(flow-anchor,points_local / stride)[0].movedim(1,-1)
    broadcast = (*anchor.shape[:1], *([1]*(samples.ndim-2)),flow.shape[1])
    return samples + anchor[:,:,0,0,0].reshape(broadcast)
