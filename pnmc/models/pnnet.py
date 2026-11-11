"""Two-encoder PN-Net with five Add fusions and three non-local decoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .non_local import NonLocal2D


class DenseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        growth = max(1, output_channels // 4)
        self.layers = nn.ModuleList()
        for index in range(3):
            current = input_channels + index * growth
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(current), nn.ReLU(inplace=False),
                nn.Conv2d(current, growth, kernel_size=3, padding=1)))
        self.compress = nn.Conv2d(input_channels + 3 * growth, output_channels, 1)

    def forward(self, image):
        features = [image]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.compress(torch.cat(features, dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut = (nn.Identity() if input_channels == output_channels
                         else nn.Conv2d(input_channels, output_channels, 1))

    def forward(self, image):
        branch = F.relu(self.bn1(self.conv1(image)))
        branch = self.bn2(self.conv2(branch))
        return F.relu(branch + self.shortcut(image))


class PNNet2D(nn.Module):
    def __init__(self, channels=(24, 48, 96, 96, 192), query_chunk=256,
                 checkpoint_chunks=True, prior_fusion="add",
                 use_non_local=True):
        super().__init__()
        if len(channels) != 5 or any(channel <= 0 for channel in channels):
            raise ValueError("PN-Net requires five positive channel widths")
        if prior_fusion not in {"add", "concat"}:
            raise ValueError("unknown prior fusion")
        if type(use_non_local) is not bool:
            raise ValueError("use_non_local must be bool")
        self.channels = tuple(channels)
        self.prior_fusion = prior_fusion
        self.use_non_local = use_non_local
        # Empty in the default model, preserving every legacy PN-Net state key.
        self.fusion_convs = (nn.ModuleList() if prior_fusion == "add" else
                             nn.ModuleList([nn.Conv2d(2*c, c, 1) for c in channels]))
        self.prior_blocks = nn.ModuleList()
        self.sparse_blocks = nn.ModuleList()
        for level, output_channels in enumerate(channels):
            input_channels = 1 if level == 0 else channels[level - 1]
            self.prior_blocks.append(DenseBlock(input_channels, output_channels))
            self.sparse_blocks.append(DenseBlock(input_channels, output_channels))
        self.upconvs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for stage in range(4):
            source_channels = channels[4 - stage]
            target_channels = channels[3 - stage]
            self.upconvs.append(nn.ConvTranspose2d(source_channels, target_channels,
                                                   kernel_size=4, stride=2, padding=1))
            self.residuals.append(ResidualBlock(2 * target_channels, target_channels))
            if stage < 3:
                self.attentions.append(
                    NonLocal2D(target_channels, query_chunk, checkpoint_chunks)
                    if use_non_local else nn.Identity())
        self.head = nn.Conv2d(channels[0], 1, 1)

    def _fuse(self, sparse, prior, level):
        if self.prior_fusion == "add":
            return sparse + prior
        return self.fusion_convs[level](torch.cat((sparse, prior), dim=1))

    def encode(self, sparse, prior):
        if sparse.shape != prior.shape or sparse.ndim != 4 or sparse.shape[1] != 1:
            raise ValueError("sparse and prior must be matching [B,1,H,W] tensors")
        prior_feature = self.prior_blocks[0](prior)
        sparse_feature = self.sparse_blocks[0](sparse)
        fusion = self._fuse(sparse_feature, prior_feature, 0)
        features = [fusion]
        for level in range(1, 5):
            prior_feature = self.prior_blocks[level](F.max_pool2d(prior_feature, 2))
            sparse_feature = self.sparse_blocks[level](F.max_pool2d(fusion, 2))
            fusion = self._fuse(sparse_feature, prior_feature, level)
            features.append(fusion)
        return tuple(features)

    def forward(self, sparse, prior):
        if sparse.shape != prior.shape or sparse.ndim != 4:
            raise ValueError("sparse and prior shapes must match")
        height, width = sparse.shape[-2:]
        padding = (0, (-width) % 16, 0, (-height) % 16)
        if padding[1] or padding[3]:
            sparse = F.pad(sparse, padding, mode="replicate")
            prior = F.pad(prior, padding, mode="replicate")
        features = self.encode(sparse, prior)
        decoded = features[4]
        for stage, skip in enumerate((features[3], features[2], features[1], features[0])):
            decoded = self.upconvs[stage](decoded)
            decoded = self.residuals[stage](torch.cat((decoded, skip), dim=1))
            if stage < 3:
                decoded = self.attentions[stage](decoded)
        return self.head(decoded)[..., :height, :width]
