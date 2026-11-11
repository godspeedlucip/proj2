"""Exact global non-local attention with query-wise activation recomputation."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class NonLocal2D(nn.Module):
    def __init__(self, channels: int, query_chunk: int = 256,
                 checkpoint_chunks: bool = True):
        super().__init__()
        if channels <= 0 or query_chunk <= 0:
            raise ValueError("channels and query_chunk must be positive")
        self.query_chunk = query_chunk
        self.checkpoint_chunks = checkpoint_chunks
        hidden = max(1, channels // 2)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, hidden, 1)
        self.output = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _attend(query_chunk, all_keys, all_values):
        scores = torch.bmm(query_chunk, all_keys)
        weights = scores.softmax(dim=-1)
        return torch.bmm(weights, all_values)

    def forward(self, image):
        if image.ndim != 4:
            raise ValueError("NonLocal2D expects [B,C,H,W]")
        batch, _, height, width = image.shape
        queries = self.query(image).flatten(2).transpose(1, 2)
        keys = self.key(image).flatten(2)
        values = self.value(image).flatten(2).transpose(1, 2)
        contexts = []
        for first in range(0, height * width, self.query_chunk):
            current = queries[:, first:first + self.query_chunk]
            if self.training and self.checkpoint_chunks and torch.is_grad_enabled():
                # Pass this chunk as an argument: a closure over `first` would
                # recompute every saved chunk using the loop's final index.
                attended = checkpoint(self._attend, current, keys, values,
                                      use_reentrant=False)
            else:
                attended = self._attend(current, keys, values)
            contexts.append(attended)
        context = torch.cat(contexts, dim=1).transpose(1, 2)
        context = context.reshape(batch, -1, height, width)
        return image + self.output(context)
