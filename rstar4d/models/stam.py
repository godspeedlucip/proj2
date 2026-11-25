"""One STAM Joint Block using the SAM ViT-B global attention checkpoint.

SAM attention and relative-position interpolation are adapted from Meta's
segment_anything/modeling/image_encoder.py at 6fdee8f2727f4506cfbbe553e23b895e27956588.
Copyright (c) Meta Platforms, Inc. and affiliates. Apache-2.0, see SAM-LICENSE.txt.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..utils.io import sha256


WIDTH = 768
HEADS = 12
ADAPTER_WIDTH = 64
BLOCK_INDEX = 5
SOURCE_COMMIT = '6fdee8f2727f4506cfbbe553e23b895e27956588'


def architecture(mode):
    return {
        'version': 1, 'mode': mode, 'block_index': BLOCK_INDEX, 'embed_dim': WIDTH,
        'num_heads': HEADS, 'adapter_dim': ADAPTER_WIDTH, 'temporal_grid': 'Tx1',
        'insertion': 'stage2_before_conv5',
    }


def _relative_positions(size, table):
    distance = 2 * size - 1
    if len(table) != distance:
        table = F.interpolate(table.T.unsqueeze(0), size=distance, mode='linear')[0].T
    position = torch.arange(size, device=table.device)
    return table[position[:, None] - position[None, :] + size - 1]


class SAMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, WIDTH * 3)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.rel_pos_h = nn.Parameter(torch.zeros(127, WIDTH // HEADS))
        self.rel_pos_w = nn.Parameter(torch.zeros(127, WIDTH // HEADS))

    def forward(self, x):
        batch, height, width, channels = x.shape
        n = height * width
        q, k, v = self.qkv(x).reshape(batch, n, 3, HEADS, -1).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q * ((channels // HEADS) ** -0.5)) @ k.transpose(-2, -1)
        grid = q.reshape(batch, HEADS, height, width, -1)
        rh = torch.einsum('bnhwc,hkc->bnhwk', grid, _relative_positions(height, self.rel_pos_h))
        rw = torch.einsum('bnhwc,wkc->bnhwk', grid, _relative_positions(width, self.rel_pos_w))
        scores = scores.reshape(batch, HEADS, height, width, height, width)
        scores = scores + rh.unsqueeze(-1) + rw.unsqueeze(-2)
        values = scores.reshape(batch, HEADS, n, n).softmax(dim=-1) @ v
        return self.proj(values.transpose(1, 2).reshape(batch, height, width, channels))


class SAMMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(WIDTH, 4 * WIDTH)
        self.lin2 = nn.Linear(4 * WIDTH, WIDTH)

    def forward(self, h):
        return self.lin2(F.gelu(self.lin1(h)))


class SAMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH, eps=1e-6)
        self.attn = SAMAttention()
        self.norm2 = nn.LayerNorm(WIDTH, eps=1e-6)
        self.mlp = SAMMLP()

    def load_pretrained(self, path):
        if not path:
            raise ValueError('stam.sam_checkpoint is required for new STAM training')
        path = Path(path).resolve()
        value = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(value, dict):
            raise ValueError('stam.sam_checkpoint must contain a SAM ViT-B state dictionary')
        value = value.get('state_dict', value)
        if not isinstance(value, dict):
            raise ValueError('stam.sam_checkpoint state_dict must be a mapping')
        prefix = f'image_encoder.blocks.{BLOCK_INDEX}.'
        state = {k[len(prefix):]: v for k, v in value.items() if isinstance(k, str) and k.startswith(prefix)}
        expected = self.state_dict()
        missing = sorted(expected.keys() - state.keys())
        extra = sorted(state.keys() - expected.keys())
        invalid = {k: f'{getattr(state[k], "shape", None)}; expected {expected[k].shape} and finite floats'
                   for k in expected.keys() & state.keys()
                   if not torch.is_tensor(state[k]) or state[k].shape != expected[k].shape
                   or not state[k].is_floating_point() or not torch.isfinite(state[k]).all()}
        if missing or extra or invalid:
            raise ValueError(f'stam.sam_checkpoint block {BLOCK_INDEX}: missing={missing}, unexpected={extra}, invalid={invalid}')
        self.load_state_dict(state, strict=True)
        return {'path': str(path), 'sha256': sha256(path), 'source_commit': SOURCE_COMMIT}


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(WIDTH, ADAPTER_WIDTH)
        self.up = nn.Linear(ADAPTER_WIDTH, WIDTH)

    def forward(self, h):
        return h + self.up(F.relu(self.down(h)))


class STAMJointBlock(nn.Module):
    def __init__(self, mode, z_chunk_size=1, activation_checkpointing=True):
        super().__init__()
        if mode not in ('joint', 'temporal_only'):
            raise ValueError('Invalid STAM architecture')
        self.mode = mode
        self.z_chunk_size = z_chunk_size
        self.activation_checkpointing = activation_checkpointing
        self.sam = SAMBlock().requires_grad_(False)
        self.input_projection = nn.Linear(128, WIDTH)
        self.output_projection = nn.Linear(WIDTH, 128)
        self.temporal_adapter = Adapter()
        self.fusion_adapter = Adapter()
        # Construct shared layers in the same order in both architecture modes.
        self.spatial_adapter = Adapter() if mode == 'joint' else None
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _layer_groups(self, features):
        groups, phases, height, width, _ = features.shape
        u = self.input_projection(features)
        spatial = u
        if self.spatial_adapter is not None:
            xy = u.reshape(groups * phases, height, width, WIDTH)
            xy = self.spatial_adapter(self.sam.attn(self.sam.norm1(xy)))
            spatial = u + xy.reshape_as(u)

        temporal = u.permute(0, 2, 3, 1, 4).reshape(groups * height * width, phases, 1, WIDTH)
        temporal = temporal + self.temporal_adapter(self.sam.attn(self.sam.norm1(temporal)))
        temporal = temporal.reshape(groups, height, width, phases, WIDTH).permute(0, 3, 1, 2, 4)

        fused = spatial + temporal
        h = self.sam.norm2(fused)
        return features + self.output_projection(fused + self.sam.mlp(h) + self.fusion_adapter(h))

    def forward(self, features):
        batch, channels, phases, depth, height, width = features.shape
        layers = features.permute(0, 3, 2, 4, 5, 1).reshape(batch * depth, phases, height, width, channels)
        use_checkpoint = self.training and self.activation_checkpointing and torch.is_grad_enabled()
        outputs = []
        for chunk in layers.split(self.z_chunk_size or len(layers)):
            if use_checkpoint:
                outputs.append(checkpoint(self._layer_groups, chunk, use_reentrant=False, preserve_rng_state=False))
            else:
                outputs.append(self._layer_groups(chunk))
        return torch.cat(outputs).reshape(batch, depth, phases, height, width, channels).permute(0, 5, 2, 1, 3, 4).contiguous()
