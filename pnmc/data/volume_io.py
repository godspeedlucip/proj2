"""Physical volume IO with checked shape, affine, and frame metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile
import torch

from .schema import ArrayRef, GridSpec


def _affine(grid: GridSpec):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(grid.direction_xyz) @ np.diag(grid.spacing_xyz_mm)
    matrix[:3, 3] = grid.origin_xyz_mm
    return matrix


def _sidecar(path: Path):
    return Path(str(path) + ".grid.json")


def save_volume(path: Path, values: torch.Tensor, grid: GridSpec) -> ArrayRef:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    if tuple(array.shape) != grid.shape_zyx or not np.isfinite(array).all():
        raise ValueError("volume shape/grid mismatch or non-finite values")
    array = np.asarray(array, dtype=np.float32)
    name = path.name.lower()
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        nib.save(nib.Nifti1Image(array.transpose(2, 1, 0), _affine(grid)), str(path))
    elif name.endswith(".tif") or name.endswith(".tiff"):
        tifffile.imwrite(str(path), array, photometric="minisblack")
    elif name.endswith(".npy"):
        np.save(path, array, allow_pickle=False)
    else:
        raise ValueError("unsupported volume format: " + path.name)
    _sidecar(path).write_text(json.dumps({"grid": grid.to_dict(),
                                          "grid_sha256": grid.fingerprint()},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    return ArrayRef(path, None, "z,y,x", "1/mm", hashlib.sha256(path.read_bytes()).hexdigest())


def load_volume(ref: ArrayRef, grid: GridSpec) -> torch.Tensor:
    if ref.axes != "z,y,x" or ref.unit != "1/mm":
        raise ValueError("volume axes/unit must be z,y,x and 1/mm")
    path = ref.path
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref.sha256:
        raise ValueError("volume sha256 mismatch")
    sidecar = _sidecar(path)
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("grid_sha256") != grid.fingerprint():
            raise ValueError("volume grid fingerprint mismatch")
    name = path.name.lower()
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        image = nib.load(str(path))
        if not np.allclose(image.affine, _affine(grid), rtol=1e-5, atol=1e-5):
            raise ValueError("volume affine and grid disagree")
        array = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0)
    elif name.endswith(".tif") or name.endswith(".tiff"):
        if not sidecar.exists():
            raise ValueError("TIFF volume requires grid sidecar")
        array = tifffile.imread(str(path))
    elif name.endswith(".npy"):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError("unsupported volume format: " + path.name)
    if tuple(array.shape) != grid.shape_zyx or not np.isfinite(array).all():
        raise ValueError("volume shape/grid mismatch or non-finite values")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
