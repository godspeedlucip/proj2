"""Shared CycN-Net three-phase encoder and its shallow RSTAR readout."""
import torch
from torch import nn
from torch.nn import functional as F


class CycN3DBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_encode1 = nn.Sequential(nn.Conv3d(1, 4, (3, 5, 5)), nn.PReLU())
        self.conv_encode1_2 = nn.Sequential(nn.Conv2d(4, 4, 5), nn.PReLU())

    def forward(self, triplet):
        if triplet.ndim != 5 or triplet.shape[1:3] != (1, 3):
            raise ValueError('CycN expects [N,1,3,H,W] triplets')
        if min(triplet.shape[-2:]) < 9:
            raise ValueError('CycN needs at least a 9-by-9 XY field')
        return self.conv_encode1_2(self.conv_encode1(triplet).squeeze(2))


def phase_windows(t, topology, device=None):
    """Return the neighbors of every actual phase, in existing window order."""
    if t < 2:
        raise ValueError('Temporal windows require at least two phases')
    positions = torch.arange(t, device=device)[:, None]
    offsets = torch.tensor([-1, 0, 1], device=device)
    indices = positions + offsets
    if topology == 'cyclic':
        return indices.remainder(t)
    if topology == 'ordered_states':
        period = 2 * (t - 1)
        reflected = indices.remainder(period)
        return torch.minimum(reflected, period - reflected)
    raise ValueError(f'Unknown temporal_topology: {topology!r}')


class CycNBranch(nn.Module):
    def __init__(self, config, stem_channels):
        super().__init__()
        self.mode = config['mode']
        self.fusion = config['fusion']
        self.z_chunk = config['z_chunk']
        self.block = CycN3DBlock()
        channels = stem_channels if self.fusion == 'xy_pre_z' else 1
        self.fuse = nn.Conv2d(12, channels, 1, bias=False)
        nn.init.zeros_(self.fuse.weight)

    def _readout(self, gated, windows):
        b, z, t, h, w = gated.shape
        if self.mode == 'center_only':
            triplets = gated.unsqueeze(3).expand(-1, -1, -1, 3, -1, -1)
        else:
            triplets = torch.stack([
                sample.index_select(1, indices).unflatten(1, (t, 3))
                for sample, indices in zip(gated, windows)
            ])
        encoded = self.block(triplets.reshape(b * z * t, 1, 3, h, w))
        if self.mode == 'center_only':
            joined = torch.cat((encoded, encoded, encoded), dim=1)
        else:
            joined = torch.stack([
                sample.index_select(1, indices).unflatten(1, (t, 3)).flatten(2, 3)
                for sample, indices in zip(encoded.unflatten(0, (b, z, t)), windows)
            ]).flatten(0, 2)
        delta = self.fuse(joined).unflatten(0, (b, z, t))
        return delta.permute(0, 3, 2, 1, 4, 5)

    def _windows(self, gated, temporal_topology, case_ids):
        b, t = gated.shape[:2]
        topologies = [temporal_topology] * b if isinstance(temporal_topology, str) or temporal_topology is None else list(temporal_topology)
        cases = [case_ids] if isinstance(case_ids, str) else case_ids
        if cases is None:
            cases = [f'batch index {i}' for i in range(b)]
        if len(topologies) != b or len(cases) != b:
            raise ValueError('Provide temporal_topology and case identity per input sample')
        if self.mode == 'five_phase' and t < 5:
            raise ValueError(f'CycN five_phase needs at least five phases for {list(cases)}')
        indices = {}
        for case, topology in zip(cases, topologies):
            if topology not in ('cyclic', 'ordered_states'):
                raise ValueError(f'{case}: missing or unknown temporal_topology {topology!r}')
            if self.mode == 'five_phase' and topology not in indices:
                indices[topology] = phase_windows(t, topology, gated.device).flatten()
        return [indices[topology] for topology in topologies] if indices else None

    def correction(self, gated, temporal_topology, case_ids=None, xy_shape=None):
        """Map [B,T,Z,H,W] FDK to [B,C,T,Z,H,W] corrections."""
        windows = self._windows(gated, temporal_topology, case_ids)
        h, w = gated.shape[-2:]
        real_h, real_w = xy_shape or (h, w)
        if min(real_h, real_w) < 9:
            raise ValueError(f'CycN needs at least a 9-by-9 XY field for {case_ids or "input"}')
        fdk = gated[..., :real_h, :real_w].transpose(1, 2)
        parts = [F.pad(self._readout(chunk, windows), (4, 4 + w - real_w, 4, 4 + h - real_h))
                 for chunk in fdk.split(self.z_chunk, dim=1)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=3)

    def stem(self, stem, x, temporal_topology, case_ids=None, xy_shape=None):
        delta = self.correction(x[:, 0], temporal_topology, case_ids, xy_shape)
        if self.fusion == 'image_residual':
            corrected = x[:, :1] + delta.to(dtype=x.dtype)
            return stem(torch.cat((corrected, x[:, 1:]), dim=1))
        _, _, phases, depth, height, width = x.shape
        xy = stem.conv1(x.flatten(2, 3)).unflatten(2, (phases, depth))
        features = xy + delta.to(dtype=xy.dtype)
        if stem.stride[0] == 2:
            features = stem.conv2(features.flatten(4)).unflatten(-1, (height, width))
        return stem.lrelu(features).contiguous()
