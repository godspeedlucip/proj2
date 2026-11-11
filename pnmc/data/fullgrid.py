"""Offline full-grid labels and train-only slice manifest preparation."""

from __future__ import annotations

import hashlib
import gc
import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from pnmc.reconstruction.projector import TigreProjector

from .grids import canonicalize_to_scanner
from .manifest import load_case_manifest, load_network_manifest
from .schema import ArrayRef, CaseManifest, GridSpec, NetworkManifest
from .uniform_import import case_to_dict
from .volume_io import load_volume, save_volume


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_generator_source(ref: ArrayRef, source_grid: GridSpec):
    """Read source native [z,y,x] array only in this offline module."""
    if ref.axes != "z,y,x" or ref.unit != "HU":
        raise ValueError("generator source must declare native z,y,x HU")
    if _sha256(ref.path) != ref.sha256:
        raise ValueError("source CT sha256 mismatch")
    if ref.path.name.lower().endswith((".nii", ".nii.gz")):
        values = np.asarray(nib.load(str(ref.path)).dataobj, dtype=np.float32)
    elif ref.path.suffix.lower() == ".npy":
        values = np.asarray(np.load(ref.path, allow_pickle=False), dtype=np.float32)
    else:
        raise ValueError("source CT must be NIfTI or NPY")
    if values.shape != source_grid.shape_zyx:
        raise ValueError("source CT shape disagrees with generator grid")
    valid = np.isfinite(values)
    return values, valid


def _calibrate_hu(hu: np.ndarray, valid: np.ndarray, calibration: dict) -> torch.Tensor:
    try:
        coefficients = np.asarray(calibration["polynomial_coefficients"], dtype=np.float64)
        threshold = float(calibration["air_threshold"])
        air = float(calibration["air_value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("explicit generator attenuation calibration required") from exc
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all() or not np.isfinite([threshold, air]).all():
        raise ValueError("invalid generator attenuation calibration")
    result = np.empty(hu.shape, np.float32)
    for start in range(0, hu.shape[0], 16):
        end = min(start + 16, hu.shape[0])
        block = np.where(valid[start:end], hu[start:end], -1000.)
        mu = np.maximum(.02 * (1. + block / 1000.), 0.).astype(np.float32)
        mapped = np.polyval(coefficients, mu.astype(np.float64))
        result[start:end] = np.where(mapped < threshold, air, mapped).astype(np.float32)
        result[start:end][~valid[start:end]] = air
    return torch.from_numpy(result)


def resample_valid_mask(valid: torch.Tensor, source_grid: GridSpec,
                        target_grid: GridSpec) -> torch.Tensor:
    if tuple(valid.shape) != source_grid.shape_zyx:
        raise ValueError("source validity shape differs from grid")
    sampled = canonicalize_to_scanner(valid.to(torch.float32), source_grid, target_grid)
    return sampled >= 1 - 1e-6


def _save_binary(path: Path, mask: torch.Tensor) -> ArrayRef:
    values = mask.detach().cpu().numpy().astype(np.uint8)
    np.save(path, values, allow_pickle=False)
    return ArrayRef(path.resolve(), None, "z,y,x", "binary", _sha256(path))


def _refs_to_dict(value):
    if isinstance(value, ArrayRef):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _refs_to_dict(item) for key, item in value.items()}
    return value


def prepare_fullgrid_reference(case: CaseManifest, *, source_phase_refs: dict,
                               source_grid: GridSpec, calibration: dict,
                               output: Path) -> CaseManifest:
    if case.projection_volume_grid.fingerprint() != source_grid.fingerprint():
        raise ValueError("source CT grid differs from projection generator grid")
    if set(source_phase_refs) != set(case.phase_ids):
        raise ValueError("one explicit source CT reference required per phase")
    if "prior" not in case.arrays or "sparse_by_phase" not in case.arrays:
        raise ValueError("fullgrid references require FDK prior and sparse inputs")
    if set(case.arrays["sparse_by_phase"]) != set(case.phase_ids):
        raise ValueError("FDK sparse phases incomplete")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fullgrid output directory already contains files")
    output.mkdir(parents=True, exist_ok=True)
    angles = np.asarray(case.angles_rad, dtype=np.float32)
    try:
        fdk_padding_vu = tuple(int(x) for x in case.provenance["fdk"]["padding_vu"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source case lacks matching FDK padding provenance") from exc
    source_projector = TigreProjector(case.geometry, source_grid, angles, device="cpu")
    fdk_projector = TigreProjector(case.geometry, case.reconstruction_grid, angles,
                                   device="cpu", fdk_padding_vu=fdk_padding_vu)
    all_views = np.arange(len(angles), dtype=np.int64)
    references, valid_refs, coverage = {}, {}, {}
    source_hashes = {}
    support = canonicalize_to_scanner(torch.ones(source_grid.shape_zyx),
                                      source_grid, case.reconstruction_grid) > 0
    support_ref = _save_binary(output / "support_q0.npy", support)
    for phase in case.phase_ids:
        ref = source_phase_refs[phase]
        hu, valid_source = _load_generator_source(ref, source_grid)
        calibrated = _calibrate_hu(hu, valid_source, calibration)
        phase_projections = source_projector.project(calibrated, all_views)
        reference = fdk_projector.fdk(phase_projections, all_views)
        reference = torch.where(support, reference, torch.zeros_like(reference))
        valid = resample_valid_mask(torch.from_numpy(valid_source.copy()),
                                    source_grid, case.reconstruction_grid) & support
        references[phase] = save_volume(output / ("reference_" + phase + ".npy"),
                                         reference, case.reconstruction_grid)
        valid_refs[phase] = _save_binary(output / ("valid_" + phase + ".npy"), valid)
        coverage[phase] = int(valid.sum())
        source_hashes[phase] = ref.sha256
        del hu, valid_source, calibrated, phase_projections, reference, valid
        gc.collect()
    derived = replace(case,
                      arrays={**case.arrays, "reference_by_phase": references,
                              "valid_by_phase": valid_refs, "support_q0": support_ref},
                      provenance={**case.provenance,
                                  "reference_kind": "single_phase_all_view_fdk",
                                  "calibration": calibration,
                                  "source_ct_sha256_by_phase": source_hashes,
                                  "valid_voxels_by_phase": coverage,
                                  "reference_grid_sha256": case.reconstruction_grid.fingerprint()})
    payload = case_to_dict(derived)
    payload["arrays"] = _refs_to_dict(derived.arrays)
    path = output / "fullgrid_case.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_case_manifest(path, require_physics=True)


def build_fullgrid_network_manifest(cases: list, split_subjects: dict,
                                    *, output: Path = None) -> NetworkManifest:
    if set(split_subjects) != {"train", "valid", "test"}:
        raise ValueError("fullgrid subject split must contain train, valid, test")
    seen = set()
    for split, subjects in split_subjects.items():
        for subject in subjects:
            if subject in seen:
                raise ValueError("subject appears in multiple fullgrid splits")
            seen.add(subject)
    if not cases:
        raise ValueError("fullgrid network manifest needs cases")
    output = Path(output or (cases[0].manifest_path.parent / "network_manifest.json")).resolve()
    if output.exists():
        raise FileExistsError("fullgrid network manifest already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    slice_dir = output.parent / (output.stem + "_slices")
    slice_dir.mkdir(exist_ok=True)
    records = []
    grids = {}
    for case in cases:
        if case.subject_id not in seen:
            raise ValueError("fullgrid case subject absent from split table")
        grid = case.reconstruction_grid
        grid_id = grid.fingerprint()
        grids[grid_id] = grid.to_dict()
        required = ("prior", "sparse_by_phase", "reference_by_phase", "valid_by_phase")
        if any(name not in case.arrays for name in required):
            raise ValueError("case lacks fullgrid FDK inputs or references")
        prior = load_volume(case.arrays["prior"], grid).numpy()
        for phase in case.phase_ids:
            sparse = load_volume(case.arrays["sparse_by_phase"][phase], grid).numpy()
            label = load_volume(case.arrays["reference_by_phase"][phase], grid).numpy()
            valid_ref = case.arrays["valid_by_phase"][phase]
            if _sha256(valid_ref.path) != valid_ref.sha256 or valid_ref.unit != "binary":
                raise ValueError("fullgrid valid mask integrity/unit failed")
            valid = np.load(valid_ref.path, allow_pickle=False)
            if valid.shape != grid.shape_zyx or not np.isin(valid, [0, 1]).all():
                raise ValueError("invalid fullgrid valid mask")
            for z in range(grid.shape_zyx[0]):
                name = hashlib.sha256((case.subject_id + "|" + case.case_id + "|" +
                                       case.acquisition_id + "|" + phase + "|" + str(z)).encode()).hexdigest()[:24]
                path = slice_dir / (name + ".npz")
                if path.exists():
                    raise FileExistsError("fullgrid slice already exists: " + str(path))
                np.savez_compressed(path, input=sparse[z], prior=prior[z],
                                    label=label[z], valid_mask=valid[z])
                digest = _sha256(path)
                arrays = {key: {"path": str(path), "key": key, "axes": "y,x",
                                "unit": "binary" if key == "valid_mask" else "1/mm",
                                "sha256": digest}
                          for key in ("input", "prior", "label", "valid_mask")}
                records.append({"subject_id": case.subject_id, "case_id": case.case_id,
                                "acquisition_id": case.acquisition_id,
                                "phase_id": phase, "slice_index": z,
                                "grid_id": grid_id, "arrays": arrays})
    payload = {"schema_version": "1", "records": records,
               "split_subjects": split_subjects, "grids": grids,
               "provenance": {"reference_kind": "single_phase_all_view_fdk",
                              "source_case_hashes": [case.manifest_sha256 for case in cases]}}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_network_manifest(output)
