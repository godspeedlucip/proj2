"""Shared architecture for the four training recipes."""
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .backbone import backbone_forward, normalize
from .decnn import DeCNN3D


@contextmanager
def preserve_batchnorm_buffers(model):
    buffers = [value for name, value in model.named_buffers() if name.endswith(
        ("running_mean", "running_var", "num_batches_tracked")
    )]
    copies = [value.clone() for value in buffers]
    try:
        yield
    finally:
        with torch.no_grad():
            for value, saved in zip(buffers, copies):
                value.copy_(saved)


class Combined(nn.Module):
    def __init__(self, backbone, alpha, upsampling, *, decnn=True):
        super().__init__()
        self.backbone = backbone
        self.decnn = DeCNN3D(upsampling) if decnn else None
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.policy = "stage1"
        self.set_policy("stage1")

    def set_policy(self, policy):
        self.policy = policy
        self.backbone.requires_grad_(policy not in {"frozen", "preheat"})
        if policy == "stage1":
            from rstar4d.models.model import is_z_parameter
            for name, parameter in self.backbone.named_parameters():
                if is_z_parameter(name):
                    parameter.requires_grad_(False)
        if getattr(self.backbone, "stam", None) is not None:
            self.backbone.stam.sam.requires_grad_(policy == "random_joint")
        self.alpha.requires_grad_(policy in {"random_joint", "unfrozen"})
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.policy in {"frozen", "preheat"}:
            self.backbone.eval()
        if getattr(self.backbone, "stam", None) is not None and self.policy != "random_joint":
            self.backbone.stam.sam.eval()
        return self

    def normalized_backbone(self, batch, norm):
        return backbone_forward(self.backbone, batch["x"], batch["a_s"], norm)

    def phase_batches(self, batch, size):
        _, _, phases, z, h, w = batch["x"].shape
        count = batch["x"].shape[0]
        for start in range(0, phases, size):
            end = min(start + size, phases)
            x = batch["x"][:, :, start:end].transpose(1, 2).reshape(-1, 1, z, h, w)
            prior = batch["a_d"].unsqueeze(1).expand(count, end - start, 1, z, h, w)
            yield slice(start, end), x, prior.reshape_as(x)

    def normalized_decnn(self, x, prior, transform, recompute=False):
        inputs = normalize(x, transform), normalize(prior, transform)
        if recompute:
            return checkpoint(
                self.decnn, *inputs, use_reentrant=False,
                context_fn=lambda: (nullcontext(), preserve_batchnorm_buffers(self.decnn)),
            )
        return self.decnn(*inputs)
