"""TT U-Net temporal transformer adapted to the Stage II x2 skip."""

import torch
from torch import nn
from torch.nn import functional as F


TRAINABLE_PARAMETERS = 1_887_136
TOPOLOGIES = ('ordered_states', 'cyclic')


def architecture(mode):
    return {
        'version': 1, 'mode': mode, 'channels': 64, 'embed_dim': 256,
        'num_heads': 16, 'mlp_dim': 1024, 'num_blocks': 2, 'drop_path': 0.1,
        'insertion': 'stage2_x2_skip',
    }


def temporal_visibility(phases, mode, topology=None, device=None):
    if mode == 'full':
        return torch.ones(phases, phases, dtype=torch.bool, device=device)
    if mode != 'local3' or topology not in TOPOLOGIES:
        raise ValueError('ttl.mode=local3 requires ordered_states or cyclic temporal topology')
    indices = torch.arange(phases, device=device)
    distance = (indices[:, None] - indices).abs()
    if topology == 'cyclic':
        distance = torch.minimum(distance, phases - distance)
    return distance <= 1


class _TemporalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256)
        self.qkv = nn.Linear(256, 3 * 256)
        self.head_bias = nn.Parameter(torch.zeros(16))
        self.proj = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)
        self.mlp = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def attention(self, tokens, visible=None):
        # Keep the axial sample axis separate from the spatial token axis.
        slices, locations, phases, channels = tokens.shape
        q, k, v = self.qkv(tokens).reshape(slices, locations, phases, 3, 16, 16).permute(3, 0, 1, 4, 2, 5)
        logits = (q * 0.25) @ k.transpose(-1, -2)
        logits = logits + self.head_bias.to(logits.dtype).view(1, 1, 16, 1, 1)
        if visible is not None:
            logits = logits.masked_fill(~visible[:, None, None], float('-inf'))
        attended = logits.softmax(dim=-1) @ v
        return self.proj(attended.transpose(2, 3).reshape(slices, locations, phases, channels))

    def forward(self, tokens, visible, masks):
        attn = self.attention(self.norm1(tokens), visible)
        tokens = tokens + (attn if masks is None else attn * masks[0].to(attn.dtype))
        mlp = self.mlp(self.norm2(tokens))
        return tokens + (mlp if masks is None else mlp * masks[1].to(mlp.dtype))


class TemporalTransformerSkip(nn.Module):
    def __init__(self, mode, z_chunk_size=0):
        super().__init__()
        if mode not in ('full', 'local3'):
            raise ValueError('TTL requires full or local3 mode')
        if type(z_chunk_size) is not int or z_chunk_size < 0:
            raise ValueError('ttl.z_chunk_size must be a nonnegative integer')
        self.mode = mode
        self.z_chunk_size = z_chunk_size
        self.embed = nn.Conv3d(64, 256, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.blocks = nn.ModuleList(_TemporalBlock() for _ in range(2))
        self.unembed = nn.ConvTranspose3d(256, 64, (1, 3, 3), padding=(0, 1, 1))
        self.temporal = nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0))
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)

    def _slice_correction(self, slices, visible, masks):
        embedded = self.embed(slices)
        count, channels, phases, h, w = embedded.shape
        tokens = embedded.permute(0, 3, 4, 2, 1).reshape(count, h * w, phases, channels)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, visible, None if masks is None else masks[index])
        grid = tokens.reshape(count, h, w, phases, channels).permute(0, 4, 3, 1, 2)
        restored = self.unembed(F.interpolate(grid, scale_factor=(1, 2, 2), mode='nearest'))
        return F.leaky_relu(self.temporal(restored), negative_slope=0.2)

    def forward(self, x2, temporal_topology=None):
        if x2.ndim != 6 or x2.shape[1] != 64:
            raise ValueError('Expected TTL input [B,64,T,Z,H,W]')
        b, channels, t, z, h, w = x2.shape
        if t < 2 or h % 2 or w % 2:
            raise ValueError('TTL requires at least two phases and even H/W')
        visible = None
        if self.mode == 'local3':
            topologies = temporal_topology
            if isinstance(topologies, str):
                topologies = [topologies] * b
            if (topologies is None or len(topologies) != b
                    or any(label not in TOPOLOGIES for label in topologies)):
                raise ValueError('ttl.mode=local3 requires one valid temporal topology per case')
            visible = torch.stack([
                temporal_visibility(t, self.mode, label, x2.device) for label in topologies
            ]).repeat_interleave(z, dim=0)
        slices = x2.permute(0, 3, 1, 2, 4, 5).reshape(b * z, channels, t, h, w)
        masks = None
        if self.training:
            # Sample once per block, residual branch and axial layer, independently of chunk size.
            masks = (torch.rand(2, 2, b * z, 1, 1, 1, device=x2.device) >= 0.1) / 0.9
        chunk_size = self.z_chunk_size or b * z
        corrections = []
        for start in range(0, b * z, chunk_size):
            section = slice(start, start + chunk_size)
            corrections.append(self._slice_correction(
                slices[section], None if visible is None else visible[section],
                None if masks is None else masks[:, :, section],
            ))
        correction = torch.cat(corrections).reshape(b, z, channels, t, h, w)
        return x2 + correction.permute(0, 2, 3, 1, 4, 5)
