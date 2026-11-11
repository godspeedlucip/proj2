"""Lazy slice loading without reference access during inference."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .manifest import _open_array
from .normalization import normalize
from .schema import ArrayRef, NetworkManifest, NormalizationSpec


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: NetworkManifest, *, split: str,
                 normalization: NormalizationSpec, require_label: bool):
        if split not in manifest.split_subjects:
            raise ValueError("unknown split: " + split)
        subjects = set(manifest.split_subjects[split])
        self.records = [record for record in manifest.records if record.subject_id in subjects]
        self.grids = manifest.grids
        self.normalization = normalization
        self.require_label = require_label
        self._verified_paths = set()
        if require_label and any("label" not in record.arrays for record in self.records):
            raise ValueError("label required for supervised dataset")

    def __len__(self):
        return len(self.records)

    def _read(self, ref: ArrayRef, shape):
        if ref.path not in self._verified_paths:
            digest = hashlib.sha256(ref.path.read_bytes()).hexdigest()
            if digest != ref.sha256:
                raise ValueError("array sha256 mismatch: " + str(ref.path))
            self._verified_paths.add(ref.path)
        values = np.asarray(_open_array(ref))
        if ref.axes != "y,x" or values.shape != shape:
            raise ValueError("slice axes or grid shape disagree")
        if not np.isfinite(values).all():
            raise ValueError("non-finite slice values")
        return values

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        grid = self.grids[record.grid_id]
        shape = grid.shape_zyx[1:]
        result = {"subject_id": record.subject_id, "case_id": record.case_id,
                  "acquisition_id": record.acquisition_id, "phase_id": record.phase_id,
                  "slice_index": record.slice_index, "grid_id": record.grid_id}
        for name in ("input", "prior"):
            ref = record.arrays[name]
            if ref.unit != "1/mm" and self.normalization.mode != "legacy50":
                raise ValueError("network intensity unit is unknown: " + ref.unit)
            values = self._read(ref, shape).astype(np.float32, copy=False)
            result[name] = torch.from_numpy(np.asarray(normalize(values, self.normalization),
                                                       dtype=np.float32).copy())[None]
        if self.require_label:
            ref = record.arrays["label"]
            values = self._read(ref, shape).astype(np.float32, copy=False)
            result["label"] = torch.from_numpy(np.asarray(normalize(values, self.normalization),
                                                           dtype=np.float32).copy())[None]
            if "valid_mask" in record.arrays:
                valid = self._read(record.arrays["valid_mask"], shape)
                if not np.isin(valid, [0, 1]).all():
                    raise ValueError("valid_mask must be binary")
                result["valid_mask"] = torch.from_numpy(valid.astype(np.float32, copy=True))[None]
            else:
                result["valid_mask"] = torch.ones((1,) + shape, dtype=torch.float32)
        return result
