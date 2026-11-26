"""Generate a calibrated all-view FDK using the upstream reconstruction."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import hashlib
import json
import os

import h5py
import nibabel as nib
import numpy as np
from rstar4d.simulation import backend
from rstar4d.simulation.phase_mapping import build_projection_phase_mapping
from rstar4d.simulation.reference import (
    SeriesSettings, TigreTool, apply_reconstruction_calibration, calibrated_volume,
)
from rstar4d.utils.io import canonical_hash, sha256

from .data import check_grid, check_prior, sidecar_metadata


def geometry_for(row):
    meta = row["metadata"]
    settings = SeriesSettings(**meta["settings"])
    geo = backend.geometry(meta["source_shape_zyx"], meta["source_spacing_zyx_mm"], settings)
    grid = backend.output_grid(geo, settings)
    shape = tuple(meta["geometry"]["shape_zyx"])
    origin = meta.get("stam_smoke_origin_zyx", meta.get("crop_origin_zyx", [0, 0, 0]))
    if any(type(i) is not int or i < 0 or i + n > m for i, n, m in zip(origin, shape, grid["shape_zyx"])):
        raise ValueError("crop lies outside reconstruction coordinates")
    crop = tuple(slice(a, a + n) for a, n in zip(origin, shape))
    cropped = dict(grid, shape_zyx=list(shape))
    affine = np.asarray(grid["voxel_to_scanner_zyx"])
    affine[:3, 3] = (affine @ np.r_[origin, 1])[:3]
    cropped["voxel_to_scanner_zyx"] = affine.tolist()
    cropped["center_zyx_mm"] = (affine @ np.r_[(np.array(shape) - 1) / 2, 1])[:3].tolist()
    check_grid(cropped, meta["geometry"])
    if not np.allclose(geo.offOrigin, meta["source_offOrigin_zyx_mm"], atol=1e-6):
        raise ValueError("recomputed source geometry differs")
    return settings, geo, grid, crop


def check_view_arrays(mixed, angles, phases, row, settings, geo):
    if mixed.shape != (settings.n_projections, *geo.nDetector) or not np.isfinite(mixed).all():
        raise ValueError("invalid mixed projection dimensions or values")
    expected = TigreTool.make_angles(settings.n_projections, 0, settings.angle_span_deg)
    if angles.shape != expected.shape or not np.allclose(angles, expected, rtol=0, atol=1e-6):
        raise ValueError("projection angles differ from the recorded scan geometry")
    labels, counts = np.unique(phases, return_counts=True)
    if phases.shape != (len(mixed),) or set(labels) != set(row["metadata"]["phase_labels"]) or min(counts) < 2:
        raise ValueError("projection phase topology differs")


def read_projections(row, grid, settings, geo):
    with h5py.File(row["projections"], "r") as f:
        source_grid = json.loads(f.attrs["geometry_json"])
        check_grid(source_grid, grid)
        for key in ("DSO", "DSD", "nDetector", "dDetector", "offDetector", "rotDetector", "COR", "filter", "parker"):
            if key not in source_grid or source_grid[key] != grid[key]:
                raise ValueError(f"projection geometry differs ({key})")
        mixed, angles, phases = [f[key][:] for key in ("mixed_line_integral", "angles_rad", "state_label")]
        if not np.array_equal(f["projection_index"][:], np.arange(len(angles))):
            raise ValueError("projection order differs")
    check_view_arrays(mixed, angles, phases, row, settings, geo)
    return mixed, angles, phases


def replay(row, raw, settings, geo, cycle_csv, seed):
    meta = row["metadata"]
    cycle_digest = sha256(cycle_csv)
    if cycle_digest != meta["cycle_sha256"] or str(raw["animal_id"]) != str(row["animal_id"]):
        raise ValueError("raw animal or cycle trace differs from source scan")
    for key in ("round", "series"):
        if row.get(key) is not None and raw.get(key) != row[key]:
            raise ValueError(f"raw scan identity differs ({key})")
    labels = meta["phase_labels"]
    if set(raw["phase_paths"]) != {str(label) for label in labels}:
        raise ValueError("raw phase labels differ")
    mapping = build_projection_phase_mapping(
        cycle_csv, len(labels), min_phase=settings.min_phase, max_phase=settings.max_phase,
        phase_labels=sorted(labels), n_projections=settings.n_projections, angle_span_deg=settings.angle_span_deg,
    )
    if [view.projection_index for view in mapping] != list(range(settings.n_projections)):
        raise ValueError("replayed projection mapping has an unexpected order")
    phases = np.array([view.phase_label for view in mapping], dtype=np.int32)
    angles = TigreTool.make_angles(settings.n_projections, 0, settings.angle_span_deg)
    mixed = np.empty((len(angles), *geo.nDetector), dtype=np.float32)
    for label in labels:
        path = raw["phase_paths"][str(label)]
        if sha256(path) != meta["source_hashes"][str(label)]:
            raise ValueError(f"raw phase {label} checksum differs")
        image = nib.load(path)
        if list(image.shape) != meta["source_shape_zyx"] or not np.allclose(image.affine, meta["source_affine"], atol=1e-5, rtol=0):
            raise ValueError(f"raw phase {label} grid differs")
        volume = calibrated_volume(image.dataobj, np.poly1d([1, 0]), 0, 0, apply_air_threshold=False)
        selected = phases == label
        mixed[selected] = backend.project(volume, geo, angles[selected])
    noise = meta["noise"]
    if noise["kind"] == "poisson":
        if seed is None:
            raise ValueError("Poisson replay requires the original simulation seed")
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(meta["cache_key"][:8], 16)]))
        photons = noise["photons"]
        counts = rng.poisson(photons * np.exp(-mixed.astype(np.float64)))
        mixed = (-np.log(np.maximum(counts, 1) / photons)).astype(np.float32)
    elif noise["kind"] != "none":
        raise ValueError("unsupported source noise")
    check_view_arrays(mixed, angles, phases, row, settings, geo)
    return mixed, angles, phases


def reconstruct_image(row, mixed, geo, angles, settings, expected_grid, crop):
    image, spacing = backend.reconstruct(mixed, geo, angles, settings)
    if list(image.shape) != expected_grid["shape_zyx"] or not np.allclose(spacing, expected_grid["voxel_spacing_zyx_mm"], atol=1e-6, rtol=0):
        raise ValueError("FDK output grid differs from recorded geometry")
    meta = row["metadata"]
    if meta["calibration_application"] == "post_fdk":
        image = apply_reconstruction_calibration(image, np.poly1d(meta["calibration"]["coefficients"]))
    elif meta["calibration_application"] != "none":
        raise ValueError("unsupported FDK calibration order")
    return image[crop]


def generate_prior(row, output, raw, cycle_csv, seed):
    settings, geo, grid, crop = geometry_for(row)
    provenance = {
        "method": "stored_mixed_line_integral" if row.get("projections") else "raw_replay",
        "backend": backend.backend_fingerprint(),
    }
    if row.get("projections"):
        mixed, angles, _ = read_projections(row, grid, settings, geo)
        provenance["mixed_projections_sha256"] = row["projections_sha256"]
    else:
        mixed, angles, phases = replay(row, raw, settings, geo, cycle_csv, seed)
        provenance.update(
            mixed_projections_sha256=hashlib.sha256(memoryview(mixed)).hexdigest(),
            cycle_sha256=sha256(cycle_csv), noise_seed=seed,
        )
        errors = []
        with h5py.File(row["path"], "r") as f:
            for index, label in enumerate(row["metadata"]["phase_labels"]):
                select = phases == label
                image = reconstruct_image(row, mixed[select], geo, angles[select], settings, grid, crop)
                existing = f["cbct"][index]
                if not np.allclose(image, existing, rtol=1e-5, atol=1e-7):
                    raise ValueError(f"replayed gated FDK differs for phase {label}")
                errors.append(float(np.max(np.abs(image - existing))))
        provenance["gated_max_abs"] = errors
    meta = sidecar_metadata(row, provenance)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{row['case_id']}-{canonical_hash(meta)[:20]}.h5"
    if path.is_file():
        if check_prior(path, row, sha256(path)) != meta:
            raise ValueError("existing prior provenance differs")
        return path
    image = reconstruct_image(row, mixed, geo, angles, settings, grid, crop)
    with NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        with h5py.File(temp, "w") as f:
            f.attrs["metadata_json"] = json.dumps(meta, allow_nan=False)
            f.create_dataset("prior", data=image.astype(np.float32), compression="gzip", compression_opts=1)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path
