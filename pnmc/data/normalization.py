"""One train-fitted map shared by input, prior, and reference."""

from __future__ import annotations

import math

import numpy as np
import torch

from .manifest import _open_array
from .schema import NetworkManifest, NormalizationSpec


def normalize(values, spec: NormalizationSpec):
    if spec.mode == "legacy50":
        if isinstance(values, torch.Tensor):
            return values.clamp_min(0) * 50
        return np.maximum(values, 0) * 50
    if isinstance(values, torch.Tensor):
        return 2 * (values.clamp(spec.a, spec.b) - spec.a) / (spec.b - spec.a) - 1
    return 2 * (np.clip(values, spec.a, spec.b) - spec.a) / (spec.b - spec.a) - 1


def denormalize(values, spec: NormalizationSpec):
    if spec.mode == "legacy50":
        return values / 50
    return (values + 1) * (spec.b - spec.a) / 2 + spec.a


def fit_normalization(manifest: NetworkManifest) -> NormalizationSpec:
    train_subjects = set(manifest.split_subjects["train"])
    lower = math.inf
    upper = -math.inf
    count = 0
    for record in manifest.records:
        if record.subject_id not in train_subjects:
            continue
        valid = None
        if "valid_mask" in record.arrays:
            valid = np.asarray(_open_array(record.arrays["valid_mask"]), dtype=bool)
        for name in ("input", "prior", "label"):
            ref = record.arrays.get(name)
            if ref is None:
                continue
            if ref.unit != "1/mm":
                raise ValueError("cannot fit physical normalization from unit " + ref.unit)
            values = np.asarray(_open_array(ref), dtype=np.float64)
            if valid is not None:
                if values.shape != valid.shape:
                    raise ValueError("valid_mask and image shape disagree")
                values = values[valid]
            if values.size == 0:
                continue
            if not np.isfinite(values).all():
                raise ValueError("non-finite training values")
            lower = min(lower, float(values.min()))
            upper = max(upper, float(values.max()))
            count += values.size
    if count == 0 or not lower < upper:
        raise ValueError("training data lacks a nonconstant valid intensity interval")
    return NormalizationSpec("affine_clipped", lower, upper, manifest.manifest_sha256)
