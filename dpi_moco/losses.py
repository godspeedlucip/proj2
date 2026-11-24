from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def critic_losses(critic, fake, real, *, training):
    fake, real = fake.detach(), real.detach()
    difference = critic(fake).mean() - critic(real).mean()
    alpha = torch.rand((len(fake), 1, 1, 1), device=fake.device, dtype=fake.dtype)
    with torch.enable_grad():
        mixed = (alpha * real + (1 - alpha) * fake).requires_grad_()
        gradient, = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=training)
        penalty = (gradient.flatten(1).norm(dim=1) - 1).square().mean()
    return {"critic": difference + 10 * penalty, "critic_wasserstein": difference, "gp": penalty}


def generator_losses(critic, prediction, target, degraded, mode):
    mse = F.mse_loss(prediction, target)
    if mode == "mse":
        return {"generator": mse, "mse": mse}
    score, anchor = critic(prediction, return_features=True)
    adversarial = -score.mean()
    result = {"mse": mse, "adversarial": adversarial}
    total = mse + 0.01 * adversarial
    if mode == "full":
        with torch.no_grad():
            _, positive = critic(target, return_features=True)
            _, negative = critic(degraded, return_features=True)
        contrastive, details = contrastive_loss(anchor, positive, negative)
        result.update(contrastive=contrastive, **details)
        total = total + 10 * contrastive
    return {"generator": total, **result}


def contrastive_loss(anchor, positive, negative):
    details, weighted = {}, []
    for layer, weight, a, p, n in zip((2, 4, 6, 8), (1 / 16, 1 / 8, 1 / 4, 1),
                                      anchor, positive, negative):
        numerator = (a - p).square().flatten(1).mean(dim=1)
        denominator = (a - n).square().flatten(1).mean(dim=1)
        ratio = (numerator / (denominator + 1e-8)).mean()
        details[f"cl_layer_{layer}"] = ratio
        details[f"cl_denominator_{layer}"] = denominator.mean()
        weighted.append(weight * ratio)
    return sum(weighted), details


def warp_pull(source, field):
    if source.ndim != 5 or field.shape != (len(source), 3, *source.shape[-3:]):
        raise ValueError("Expected source [B,C,Z,Y,X] and field [B,3,Z,Y,X]")
    z, y, x = source.shape[-3:]
    coordinates = [torch.arange(n, device=source.device, dtype=source.dtype) for n in (z, y, x)]
    zz, yy, xx = torch.meshgrid(*coordinates, indexing="ij")
    sample = torch.stack((2 * (xx + field[:, 0] + 0.5) / x - 1,
                          2 * (yy + field[:, 1] + 0.5) / y - 1,
                          2 * (zz + field[:, 2] + 0.5) / z - 1), dim=-1)
    return F.grid_sample(source, sample, mode="bilinear", padding_mode="border", align_corners=False)


def local_ncc(moving, fixed, window=9):
    if window < 1 or window % 2 == 0:
        raise ValueError("NCC window must be a positive odd integer")
    depth, height, width = moving.shape[-3:]
    slab = max(1, 1_048_576 // (height * width * len(moving)))
    if depth <= slab:
        return _ncc_region(moving, fixed, window, 0, depth)
    radius = window // 2
    score = moving.new_zeros(())
    for start in range(0, depth, slab):
        end = min(start + slab, depth)
        lower, upper = max(0, start - radius), min(depth, end + radius)
        args = (moving[:, :, lower:upper], fixed[:, :, lower:upper], window, start - lower, end - lower)
        value = (checkpoint(_ncc_region, *args, use_reentrant=False)
                 if torch.is_grad_enabled() else _ncc_region(*args))
        score = score + value * ((end - start) / depth)
    return score


def _ncc_region(moving, fixed, window, start, end):
    products = torch.cat((moving, fixed, moving.square(), fixed.square(), moving * fixed), dim=1)
    radius = window // 2
    for axis in range(3):
        kernel = tuple(window if i == axis else 1 for i in range(3))
        padding = tuple(radius if i == axis else 0 for i in range(3))
        if products.shape[axis + 2] < window:
            explicit = tuple(p for value in reversed(padding) for p in (value, value))
            products = F.pad(products, explicit)
            padding = (0, 0, 0)
        products = F.avg_pool3d(products, kernel, stride=1, padding=padding,
                                count_include_pad=True, divisor_override=1)
    m, f, m2, f2, mf = products[:, :, start:end].chunk(5, dim=1)
    count = window ** 3
    covariance = mf - m * f / count
    var_m = (m2 - m.square() / count).clamp_min(0)
    var_f = (f2 - f.square() / count).clamp_min(0)
    return -(covariance.square() / (var_m * var_f + 1e-5)).mean()


def field_smoothness(field):
    differences = [field.diff(dim=axis).square().mean() for axis in (2, 3, 4)
                   if field.shape[axis] > 1]
    return sum(differences, field.new_zeros(())) / 3
