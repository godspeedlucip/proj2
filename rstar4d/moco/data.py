from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
from scipy.ndimage import zoom

from ..data.contracts import metadata, read_manifest, validate_manifest, write_manifest
from ..utils.io import atomic_json, canonical_hash, read_json, read_jsonl, resolve, sha256


def case_rows(manifest):
    rows = validate_manifest(manifest, require_target=False, full=False, verify_projections=False)
    for row in rows:
        meta = row["metadata"]
        labels = meta["phase_labels"]
        expected = row.get("expected_phase_labels", meta.get("expected_phase_labels", labels))
        if (not isinstance(expected, list) or any(type(label) is not int for label in expected)
                or len(set(expected)) != len(expected)
                or labels != [label for label in expected if label in labels]):
            raise ValueError(f'{row["case_id"]}: expected_phase_labels conflicts with stored phase_labels')
        declared = row.get("phase_labels", labels)
        if declared != labels:
            raise ValueError(f'{row["case_id"]}: manifest/HDF5 phase_labels conflict')
        if meta["intensity_domain"] not in ("mu", "calibrated_mu"):
            raise ValueError(f'{row["case_id"]}: MoCo requires physical attenuation (mu/calibrated_mu)')
        if not np.allclose(meta["output_spacing_zyx_mm"], meta["geometry"]["voxel_spacing_zyx_mm"],
                           rtol=1e-5, atol=1e-6):
            # For legacy FDK zooms, the affine stores endpoint spacing. Use the affine
            # for physical distances and retain both original values in source metadata.
            row["spacing_convention"] = "legacy FDK zoom: affine endpoint spacing"
            if not meta.get("settings") or "used_spacing_zyx_mm" not in meta:
                raise ValueError(f'{row["case_id"]}: declared spacing conflicts with affine')
        row["expected_phase_labels"] = expected
        row["source_metadata_hash"] = canonical_hash(meta)
    return rows


def split_scans(manifest, split_file, out):
    rows = read_manifest(manifest)
    split = read_json(split_file)
    names = ("train", "val", "test")
    if set(split) != set(names) or any(not isinstance(split[k], list) for k in names):
        raise ValueError("Scan split must contain train/val/test lists of case IDs")
    ids = [case for key in names for case in split[key]]
    by_case = {row["case_id"]: row for row in rows}
    if len(set(ids)) != len(ids) or set(ids) != set(by_case):
        raise ValueError("Scan split must assign every source case exactly once")
    out = Path(out).resolve()
    paths = {}
    animals = {}
    for key in names:
        path = out / f"{key}_cases.jsonl"
        assigned = [{**by_case[case], "scan_split_name": key} for case in split[key]]
        write_manifest(path, assigned)
        paths[key] = {"path": str(path), "sha256": sha256(path)}
        for row in assigned:
            animals.setdefault(str(row["animal_id"]), []).append(
                {"case_id": row["case_id"], "split": key})
    result = {
        "source_manifest_sha256": sha256(manifest), "scan_split_sha256": sha256(split_file),
        "manifests": paths, "animal_scans": animals,
        "repeated_animals": sorted(k for k, scans in animals.items() if len(scans) > 1),
        "animals_in_multiple_splits": sorted(
            k for k, scans in animals.items() if len({scan["split"] for scan in scans}) > 1),
    }
    atomic_json(out / "scan_split_summary.json", result)
    return result


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".partial")
    os.close(fd)
    try:
        yield Path(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def run_lock(out):
    """Serialize writers to a run directory, with automatic release on process exit."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".moco.lock").open("a+b") as file:
        if os.name == "nt":
            import msvcrt
            file.seek(0)
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another MoCo process is writing {out}") from exc
        else:
            import fcntl
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another MoCo process is writing {out}") from exc
        yield


def artifact_rows(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= "manifest.jsonl"
    rows = {}
    for row in read_jsonl(path):
        case = row["case_id"]
        if case in rows:
            raise ValueError(f"{path}: duplicate artifact case {case}")
        rows[case] = {**row, "path": str(resolve(row["path"], path.parent))}
        if row.get("support_path"):
            rows[case]["support_path"] = str(resolve(row["support_path"], path.parent))
    if not rows:
        raise ValueError(f"{path}: empty artifact manifest")
    return rows


def validate_artifact(artifact, row, kind, mode=None):
    case = row["case_id"]
    if artifact is None:
        raise ValueError(f"{case}: missing {kind} artifact")
    if (artifact.get("source_sha256") != row["sha256"]
            or artifact.get("source_metadata_hash") != row["source_metadata_hash"]
            or str(artifact.get("animal_id")) != str(row["animal_id"])):
        raise ValueError(f"{case}: {kind} source hash, metadata or animal ID conflict")
    if artifact.get("sha256") != sha256(artifact["path"]):
        raise ValueError(f"{case}: {kind} artifact SHA-256 mismatch")
    with h5py.File(artifact["path"], "r") as file:
        meta = metadata(file)
        for key, value in row["metadata"].items():
            if meta.get(key) != value:
                raise ValueError(f"{case}: {kind} metadata conflicts at {key}")
        dataset = "prediction" if kind == "b" else "correction"
        if dataset not in file or list(file[dataset].shape) != row["shape"] or file[dataset].dtype.kind != "f":
            raise ValueError(f"{case}: invalid {kind} dataset layout")
    if kind == "b":
        provider = meta.get("inference", {})
        if (provider.get("stage") != 2 or provider.get("model_profile") != "official_v1"
                or not re.fullmatch(r"[0-9a-f]{64}", str(provider.get("checkpoint_sha256")))
                or provider.get("checkpoint_sha256") != artifact.get("checkpoint_sha256")
                or provider.get("source_sha256") != row["sha256"]
                or provider.get("mode") not in ("full", "tiled")
                or any(not isinstance(provider.get(key), list) or len(provider[key]) != 3
                       for key in ("tile", "overlap"))):
            raise ValueError(f"{case}: b requires traceable official_v1 Stage II inference, matching source and checkpoint")
        return {k: v for k, v in provider.items() if k != "source_sha256"}
    details = meta.get("pigae", {})
    if (details.get("mode") != mode or details.get("calibration_order") != "after_pigae"
            or details.get("eligible") is False
            or not details.get("provenance") or details.get("intensity_units") != "mm^-1"
            or details.get("source_sha256") != row["sha256"]
            or not details.get("input_sha256s")
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(v))
                   for v in details["input_sha256s"].values())):
        raise ValueError(f"{case}: c provenance, mode, units, source or calibration order conflict")
    return details


def baseline_artifacts(config, rows, manifest, out, baseline_manifest, checkpoint):
    if baseline_manifest and checkpoint:
        raise ValueError("Provide either --baseline or --baseline-checkpoint")
    expected_sha = config["baseline"]["checkpoint_sha256"]
    if not baseline_manifest:
        if not checkpoint:
            raise ValueError("b needs --baseline or --baseline-checkpoint")
        from ..inference.predict import predict_manifest
        from ..training.checkpoint import load_checkpoint
        ck = load_checkpoint(checkpoint, require_training=False)
        if ck.get("stage") != 2 or ck.get("model_profile") != "official_v1":
            raise ValueError("--baseline-checkpoint must be official_v1 Stage II")
        digest = sha256(checkpoint)
        if expected_sha and digest != expected_sha:
            raise ValueError("baseline.checkpoint_sha256 conflicts with Stage II checkpoint")
        inference = config["baseline"]
        # A case-level cache lets a failed case resume without regenerating other scans.
        out = Path(out)
        settings = {k: inference[k] for k in ("mode", "tile", "overlap", "amp")}
        settings["amp"] = bool(settings["amp"] and config["device"] == "cuda")
        records = {}
        for row in rows:
            key = canonical_hash({"source": row["sha256"], "case": row["case_id"],
                                  "animal": row["animal_id"], "checkpoint": digest,
                                  "settings": settings, "device": config["device"]})
            folder = out / key
            cache_manifest = folder / "prediction" / "manifest.jsonl"
            if not cache_manifest.exists():
                folder.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=folder) as staging:
                    subset = Path(staging) / "case.jsonl"
                    write_manifest(subset, [row])
                    prediction_dir = Path(staging) / "prediction"
                    predict_manifest(checkpoint, subset, prediction_dir, split_name="all",
                                     device=config["device"], input_only=True,
                                     preview_reference=False, **settings)
                    try:
                        os.replace(prediction_dir, folder / "prediction")
                    except OSError:
                        if not cache_manifest.exists():
                            raise
            records.update(artifact_rows(cache_manifest))
    else:
        records = artifact_rows(baseline_manifest)
    providers = {}
    for row in rows:
        case = row["case_id"]
        providers[case] = validate_artifact(records.get(case), row, "b")
        if expected_sha and providers[case]["checkpoint_sha256"] != expected_sha:
            raise ValueError(f"{case}: baseline.checkpoint_sha256 conflicts with cached b")
    if len({canonical_hash(provider) for provider in providers.values()}) != 1:
        raise ValueError("Selected b cases must share the same RSTAR provider and inference settings")
    return records, next(iter(providers.values()))


def phase(dataset, index):
    image = np.asarray(dataset[index], dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f"{dataset.file.filename} {dataset.name} phase index {index}: nonfinite image")
    return image


def geometry_support(file, row):
    """Return a support reader and the provenance used for inference."""
    meta = row["metadata"]
    if "geometry_support" in file:
        dataset = file["geometry_support"]
        shape = row["shape"]
        if list(dataset.shape) not in (shape, shape[1:]) or not meta.get("geometry_support_rule"):
            raise ValueError("Explicit geometry_support needs a matching shape and geometry_support_rule")

        def read(index):
            mask = dataset[index] if dataset.ndim == 4 else dataset[:]
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("geometry_support must be binary")
            return mask.astype(bool)

        return read, meta["geometry_support_rule"]

    shape = row["shape"][1:]
    if not meta.get("settings") or "used_spacing_zyx_mm" not in meta:
        mask = np.ones(shape, dtype=bool)
        return lambda _: mask, "declared grid extent"

    settings = meta["settings"]
    grid = meta["geometry"]
    fov = np.asarray(grid["nDetector"]) * grid["dDetector"] * grid["DSO"] / grid["DSD"]
    spacing = np.asarray(meta["used_spacing_zyx_mm"], dtype=float)
    native = np.minimum(np.maximum(np.floor(fov[[0, 1, 1]] / spacing).astype(int), 1),
                        meta["source_shape_zyx"])
    width = settings["fov_output_pixels"]
    if settings["fov_source_pixels"]:
        native[1:] = width
        spacing[1:] *= settings["fov_source_pixels"] / width
    coords = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(native, spacing)]
    radius = fov[1] / 2 - settings["fov_mask_shrink_pixels"] / 2 * spacing[1:]
    if (radius <= 0).any():
        raise ValueError("FDK support has nonpositive transverse radius")
    disk = (coords[1][:, None] / radius[0]) ** 2 + (coords[2][None, :] / radius[1]) ** 2 <= 1
    if min(native[1:]) < width:
        disk = zoom(disk.astype(np.float32), (width / native[1], width / native[2]),
                    order=1, mode="nearest", prefilter=False) > 0
    else:
        y, x = (native[1:] - width) // 2
        disk = disk[y:y + width, x:x + width]
    mask = (abs(coords[0])[:, None, None] <= fov[0] / 2) & disk[None]
    if list(mask.shape) != shape:
        raise ValueError("FDK geometry support/output grid mismatch")
    return lambda _: mask, "FDK unpadded detector FOV on the reconstruction grid"
