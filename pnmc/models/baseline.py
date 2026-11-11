"""Matched-intensity adapter for the untouched legacy PriorNet."""

import torch.nn as nn
import torch.nn.functional as F

from PriorNet_pytorch import Dense2D
from pnmc.data.normalization import denormalize
from pnmc.data.schema import NormalizationSpec


class PriorNetMatched(nn.Module):
    def __init__(self, normalization: NormalizationSpec):
        super().__init__()
        if normalization.mode != "affine_clipped":
            raise ValueError("PriorNetMatched requires affine_clipped network values")
        self.normalization = normalization
        self.legacy = Dense2D()

    def forward(self, sparse, prior):
        if sparse.shape != prior.shape or sparse.ndim != 4 or sparse.shape[1] != 1:
            raise ValueError("sparse and prior must be matching [B,1,H,W] tensors")
        height, width = sparse.shape[-2:]
        sparse_mu = denormalize(sparse, self.normalization)
        prior_mu = denormalize(prior, self.normalization)
        old_sparse = sparse_mu.clamp_min(0) * 50
        old_prior = prior_mu.clamp_min(0) * 50
        padding = (0, (-width) % 16, 0, (-height) % 16)
        if padding[1] or padding[3]:
            old_sparse = F.pad(old_sparse, padding, mode="replicate")
            old_prior = F.pad(old_prior, padding, mode="replicate")
        _, legacy_prediction = self.legacy(old_sparse, old_prior)
        prediction_mu = legacy_prediction[..., :height, :width] / 50
        return 2 * (prediction_mu - self.normalization.a) / (
            self.normalization.b - self.normalization.a) - 1
