"""PIDA-Net Figure 3, reproduced from DOI 10.1016/j.apradiso.2025.112276."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


FUSION_WEIGHT_NAMES = frozenset(f'conv5_2.conv{axis}.pida_weight' for axis in (1, 2, 3))


def is_pida_parameter(name):
    return name.startswith('pida_da.') or name in FUSION_WEIGHT_NAMES


def architecture(config):
    if not config or not config.get('enabled'):
        return {'enabled': False}
    return {
        'enabled': True,
        'version': 1,
        'source': 'paper_reproduction',
        'doi': '10.1016/j.apradiso.2025.112276',
        'scale': 'stage2_after_conv5',
        'channels': 64,
        'plane_axes': 'YX within each B,T,Z',
        'prior_input': 'static_prior_with_normalized_phase_channel_zeroed',
        'ports': {'Q': 'conv3(target)', 'K': 'conv3(prior)',
                  'V': 'conv3_3(conv3_2(K))+K', 'F': 'conv5(STAM(bottleneck))'},
        'selection': 'cosine_top1_row_major_first',
        'weight': 'global_raw_dot_product_softmax_fp32',
        'offset_scale': 10,
        'fusion': 'residual' if config.get('train_policy') == 'pida_only' else 'concat',
    }


def _unit_vectors(vectors):
    # Rescale first so that squaring unusually large finite features stays in range.
    scale = vectors.abs().amax(dim=-1, keepdim=True)
    scaled = vectors / torch.where(scale == 0, 1, scale)
    return F.normalize(scaled, dim=-1)


def match_planes(query, key, query_chunk_size=0):
    """Return row-major reference indices and weights, with all keys in each row."""
    with torch.autocast(device_type=query.device.type, enabled=False):
        q = query.flatten(2).transpose(1, 2).float()
        k = key.flatten(2).transpose(1, 2).float()
        with torch.no_grad():
            unit_q = _unit_vectors(q)
            unit_k = _unit_vectors(k).transpose(1, 2)
        indices, weights = [], []
        chunk_size = query_chunk_size or q.shape[1]
        for start in range(0, q.shape[1], chunk_size):
            stop = start + chunk_size
            index = (unit_q[:, start:stop] @ unit_k).argmax(dim=-1, keepdim=True)
            scores = q[:, start:stop] @ k.transpose(1, 2)
            weights.append(scores.softmax(dim=-1).gather(2, index).squeeze(2))
            indices.append(index.squeeze(2))
        return torch.cat(indices, dim=1), torch.cat(weights, dim=1)


def modulated_sample(value, indices, offsets, masks, kernel):
    """Apply a 3x3 kernel at each reference center. Offset pairs are (dy, dx)."""
    batch, channels, height, width = value.shape
    kernel_offsets = value.new_tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                                      dtype=torch.float32)
    with torch.autocast(device_type=value.device.type, enabled=False):
        centers = torch.stack((indices // width, indices % width), dim=-1)
        shifts = offsets.float().reshape(batch, 9, 2, height * width).permute(0, 3, 1, 2)
        # [batch, target_position, kernel_point, coordinate]
        y, x = (centers.unsqueeze(2) + kernel_offsets + shifts).unbind(-1)
        grid = torch.stack(((2 * x + 1) / width - 1, (2 * y + 1) / height - 1), dim=-1)
        samples = F.grid_sample(value.float(), grid, mode='bilinear', padding_mode='zeros',
                                align_corners=False)
        samples = samples * masks.float().flatten(2).transpose(1, 2).unsqueeze(1)
    samples = samples.transpose(2, 3).reshape(batch, channels * 9, height, width)
    return F.conv2d(samples.to(kernel.dtype), kernel.reshape(kernel.shape[0], channels * 9, 1, 1))


class PIDADeformableAttention(nn.Module):
    def __init__(self, channels, query_chunk_size=256, plane_batch_size=4, fusion='concat'):
        super().__init__()
        self.channels = channels
        self.query_chunk_size = query_chunk_size
        self.plane_batch_size = plane_batch_size
        self.fusion = fusion
        self.offset = nn.Conv2d(2 * channels, 18, 3, padding=1)
        self.mask = nn.Conv2d(2 * channels, 9, 3, padding=1)
        self.kernel = nn.Parameter(torch.empty(channels, channels, 3, 3))
        if fusion == 'residual':
            nn.init.zeros_(self.kernel)
        else:
            nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

    def forward(self, query, key, value, features):
        if (query.ndim != 4 or query.shape[1] != self.channels
                or any(tensor.shape != query.shape for tensor in (key, value, features))):
            raise ValueError('PIDA expects four matching [N,C,Y,X] tensors')
        centers, strength = match_planes(query, key, self.query_chunk_size)
        coarse = value.flatten(2).gather(2, centers.unsqueeze(1).expand(-1, self.channels, -1))
        context = torch.cat((coarse.reshape_as(value), features), dim=1)
        offsets = 10 * self.offset(context).tanh()
        masks = self.mask(context).sigmoid()
        result = modulated_sample(value, centers, offsets, masks, self.kernel)
        return (result.float() * strength.reshape(-1, 1, *result.shape[-2:])).to(result.dtype)

    def forward_volume(self, query, key, value, features):
        batch, channels, phases, depth, height, width = query.shape
        plane_count = batch * phases * depth
        inputs = [tensor.permute(0, 2, 3, 1, 4, 5).reshape(plane_count, channels, height, width)
                  for tensor in (query, key, value, features)]
        outputs = []
        # Recompute a plane batch during backprop to bound activation memory.
        use_checkpoint = self.training and torch.is_grad_enabled()
        chunk_size = self.plane_batch_size or plane_count
        for planes in zip(*(tensor.split(chunk_size) for tensor in inputs)):
            if use_checkpoint:
                outputs.append(checkpoint(self, *planes, use_reentrant=False, preserve_rng_state=False))
            else:
                outputs.append(self(*planes))
        return (torch.cat(outputs).reshape(batch, phases, depth, channels, height, width)
                .permute(0, 3, 1, 2, 4, 5).contiguous())


class PIDAInputConv(nn.Conv3d):
    """Keep the original parameter keys and a separately optimizable input slice."""

    def __init__(self, conv, channels):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                         conv.padding, conv.dilation, conv.groups, conv.bias is not None,
                         conv.padding_mode, device='meta', dtype=conv.weight.dtype)
        self.weight = conv.weight
        self.bias = conv.bias
        self.in_channels += channels
        self.pida_weight = nn.Parameter(conv.weight.new_zeros(conv.out_channels, channels, *conv.kernel_size))
        self.train(conv.training)

    def forward(self, value):
        return self._conv_forward(value, torch.cat((self.weight, self.pida_weight), dim=1), self.bias)
