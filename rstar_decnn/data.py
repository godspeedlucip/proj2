"""Immutable paired scans and scan-specific FDK priors."""
from pathlib import Path
import json

import h5py
import numpy as np
import torch
from rstar4d.data.contracts import validate_manifest, validate_split
from rstar4d.utils.io import atomic_jsonl, canonical_hash, read_jsonl, resolve, sha256

from .backbone import normalize


MODES = {"phase_weighted_fdk", "all_views_fdk"}
DATA_FIELDS = ("geometry", "intensity_domain", "calibration_application", "calibration", "coordinate_convention")


def check_grid(actual, expected):
    for key in ("shape_zyx", "voxel_spacing_zyx_mm", "voxel_to_scanner_zyx", "center_zyx_mm"):
        if key not in actual or np.shape(actual[key]) != np.shape(expected[key]):
            raise ValueError(f"missing or incompatible grid field {key}")
        if not np.allclose(actual[key], expected[key], rtol=0, atol=1e-6):
            raise ValueError(f"physical grid mismatch ({key})")


def check_metadata(actual, expected):
    check_grid(actual["geometry"], expected["geometry"])
    for key in DATA_FIELDS[1:]:
        if actual.get(key) != expected[key]:
            raise ValueError(f"incompatible {key}")


def ordered_indices(file, key, meta):
    own = json.loads(file.attrs.get(f"{key}_metadata_json", "{}"))
    check_metadata({**meta, **own}, meta)
    labels = own.get("phase_labels", meta["phase_labels"])
    if len(labels) != len(set(labels)) or set(labels) != set(meta["phase_labels"]):
        raise ValueError(f"{key} phase labels do not match gated scan")
    return [labels.index(label) for label in meta["phase_labels"]]


def validate_transform(value):
    if set(value) != {"scale", "offset", "unit", "source", "physical_range"}:
        raise ValueError("g requires scale, offset, unit, source and physical_range")
    low, high = value["physical_range"]
    if (not np.isfinite([value["scale"], value["offset"], low, high]).all()
            or value["scale"] <= 0 or high <= low or low + value["offset"] < 0
            or not value["source"] or not value["unit"]):
        raise ValueError("g must be invertible and map the registered supervision range into ReLU support")
    return value


def units(meta):
    return {"hu": "HU", "mu": "mm^-1", "calibrated_mu": "mm^-1", "legacy": "legacy"}[meta["intensity_domain"]]


def sidecar_metadata(row, provenance):
    return {
        **{key: row["metadata"][key] for key in DATA_FIELDS},
        "case_id": row["case_id"], "animal_id": str(row["animal_id"]), "source_sha256": row["sha256"],
        "phase_labels": row["metadata"]["phase_labels"], "prior_mode": "all_views_fdk",
        "provenance": provenance,
    }


def check_prior(path, row, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError("DeCNN prior checksum differs")
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
        check_metadata(meta, row["metadata"])
        expected = sidecar_metadata(row, meta.get("provenance"))
        for key in ("case_id", "animal_id", "source_sha256", "prior_mode", "phase_labels"):
            if meta.get(key) != expected[key]:
                raise ValueError(f"DeCNN prior identity differs ({key})")
        provenance = meta.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("mixed_projections_sha256"):
            raise ValueError("DeCNN prior requires mixed projection provenance")
        if f["prior"].shape != tuple(row["shape"][1:]) or not np.isfinite(f["prior"][:]).all():
            raise ValueError("invalid DeCNN prior array")
    return meta


def _base_rows(manifest, require_target=False):
    try:
        return validate_manifest(manifest, require_target=require_target, full=False)
    except (OSError, ValueError) as exc:
        from rstar4d.data.contracts import read_manifest, validate_case
        # The upstream aggregate validator omits the case ID for some errors.
        for row in read_manifest(manifest):
            try:
                validate_case(row["path"], require_target=require_target, full=False)
                if row.get("projections") and sha256(row["projections"]) != row.get("projections_sha256"):
                    raise ValueError("mixed projection hash differs")
            except (OSError, ValueError) as detail:
                raise ValueError(f"{row['case_id']}: {detail}") from detail
        raise exc


def prepare(manifest, output, *, prior_mode, prior_index=None, shared=False,
            raw_manifest=None, cycle_csv=None, seed=None, engineering=False):
    if (prior_mode not in MODES and not (engineering and prior_mode == "engineering_fixture")):
        raise ValueError("unknown backbone prior mode")
    if (raw_manifest is None) != (cycle_csv is None):
        raise ValueError("raw replay requires both --raw-manifest and --cycle-csv")
    if shared and prior_index:
        raise ValueError("choose a shared prior or an all-views prior index")
    output = Path(output).resolve()
    if output == Path(manifest).resolve() or (prior_index and output == Path(prior_index).resolve()):
        raise ValueError("prepared manifest must have its own output path")
    base = _base_rows(manifest)
    imported = {}
    if prior_index:
        prior_index = Path(prior_index).resolve()
        for item in read_jsonl(prior_index):
            if item["case_id"] in imported:
                raise ValueError("duplicate DeCNN prior case")
            imported[item["case_id"]] = dict(item, path=resolve(item["path"], prior_index.parent))
        if set(imported) - {row["case_id"] for row in base}:
            raise ValueError("DeCNN prior index contains unknown scan IDs")
    raw = {}
    if raw_manifest:
        from rstar4d.simulation.pipeline import read_raw_manifest
        raw = {r["case_id"]: r for r in read_raw_manifest(raw_manifest)}

    finished = []
    for row in base:
        case = row["case_id"]
        meta = row["metadata"]
        try:
            if meta.get("prior_mode") != prior_mode and not (
                engineering and prior_mode == "engineering_fixture" and "prior_mode" not in meta
            ):
                raise ValueError("backbone prior mode is absent or differs from the selected checkpoint contract")
            with h5py.File(row["path"], "r") as f:
                for key in ("target", "valid_mask"):
                    if key in f:
                        ordered_indices(f, key, meta)
                check_metadata({**meta, **json.loads(f.attrs.get("prior_metadata_json", "{}"))}, meta)
            combined = {k: v for k, v in row.items() if k not in {"metadata", "shape"}}
            if shared or (prior_mode == "all_views_fdk" and case not in imported):
                if not shared and not (row.get("projections_sha256") or meta.get("cache_key") or meta.get("prior_source_sha256")):
                    raise ValueError("stored all-views prior lacks scan provenance")
                combined["decnn_prior"] = {
                    "kind": "shared" if shared else "embedded", "mode": prior_mode,
                    "source_sha256": row["sha256"],
                }
            else:
                if case in imported:
                    item = imported[case]
                    path = item["path"]
                    digest = item["sha256"]
                else:
                    from .projection import generate_prior
                    if not row.get("projections") and case not in raw:
                        raise ValueError("missing same-scan mixed projections or verifiable raw replay inputs")
                    path = generate_prior(row, output.parent / "priors", raw.get(case), cycle_csv, seed)
                    digest = sha256(path)
                prior_meta = check_prior(path, row, digest)
                combined["decnn_prior"] = {
                    "kind": "sidecar", "mode": "all_views_fdk", "path": str(path), "sha256": digest,
                    "source_sha256": row["sha256"], "provenance": prior_meta["provenance"],
                }
            combined.update(backbone_prior_mode=prior_mode, engineering=engineering)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            raise ValueError(f"{case}: {exc}") from exc
        finished.append(combined)
    if output.exists():
        if read_jsonl(output) != finished:
            raise FileExistsError("prepared manifest already exists with a different source or contract")
    else:
        atomic_jsonl(output, finished)
    return output


def load_rows(manifest, split=None, *, target=False):
    rows = sorted(_base_rows(manifest, target), key=lambda row: row["case_id"])
    for row in rows:
        try:
            prior = row["decnn_prior"]
            meta = row["metadata"]
            compatibility_fixture = row.get("engineering") and row["backbone_prior_mode"] == "engineering_fixture"
            if (prior["source_sha256"] != row["sha256"] or
                (row["backbone_prior_mode"] != meta.get("prior_mode") and not compatibility_fixture)):
                raise ValueError("prepared source identity differs")
            if prior["kind"] == "sidecar":
                prior["path"] = str(resolve(prior["path"], Path(manifest).resolve().parent))
                m = check_prior(prior["path"], row, prior["sha256"])
                if prior["mode"] != "all_views_fdk" or prior["provenance"] != m["provenance"]:
                    raise ValueError("prepared prior provenance differs")
            elif prior["kind"] not in {"embedded", "shared"} or prior["mode"] != row["backbone_prior_mode"]:
                raise ValueError("unsupported prepared prior")
            elif prior["kind"] == "embedded" and prior["mode"] != "all_views_fdk":
                raise ValueError("embedded DeCNN prior must be all-views FDK")
            with h5py.File(row["path"], "r") as file:
                for key in ("target", "valid_mask"):
                    if key in file:
                        ordered_indices(file, key, meta)
                check_metadata({**meta, **json.loads(file.attrs.get("prior_metadata_json", "{}"))}, meta)
        except (ValueError, KeyError, OSError) as exc:
            raise ValueError(f"{row['case_id']}: {exc}") from exc
    return rows, validate_split(rows, split) if split is not None else None


def snapshot(rows, split):
    cases = []
    for row in rows:
        prior = {k: v for k, v in row["decnn_prior"].items() if k != "path"}
        cases.append({
            "case_id": row["case_id"], "animal_id": str(row["animal_id"]), "sha256": row["sha256"],
            "backbone_prior_mode": row["backbone_prior_mode"], "decnn_prior": prior,
            "metadata": row["metadata"], "engineering": row["engineering"],
        })
    return {"cases": cases, "split": split}


def read_case(row, spatial=None, *, flips=(), supervision=True):
    meta = row["metadata"]
    spatial = spatial or (slice(None),) * 3

    def phases(file, key):
        indices = ordered_indices(file, key, meta)
        return np.stack([file[key][(index, *spatial)] for index in indices])

    with h5py.File(row["path"], "r") as f:
        x = np.asarray(f["cbct"][(slice(None), *spatial)], np.float32)
        a_s = np.asarray(f["prior"][spatial], np.float32)
        mask = phases(f, "valid_mask")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{row['case_id']}: valid_mask must be binary")
        y = phases(f, "target").astype(np.float32) if supervision and "target" in f else None
    spec = row["decnn_prior"]
    if spec["kind"] == "sidecar":
        with h5py.File(spec["path"], "r") as f:
            a_d = np.asarray(f["prior"][spatial], np.float32)
    else:
        a_d = a_s
    values = {"x": x, "a_s": a_s, "a_d": a_d, "y": y, "valid": mask.astype(bool)}
    for key, value in values.items():
        if value is not None:
            if not np.isfinite(value).all():
                raise ValueError(f"{row['case_id']}: nonfinite {key}")
            axes = tuple(value.ndim - 3 + axis for axis in flips)
            values[key] = np.ascontiguousarray(np.flip(value, axis=axes))
    grid = meta["geometry"]
    index_map = np.eye(4)
    shape = x.shape[-3:]
    for axis, section in enumerate(spatial):
        index_map[axis, axis] = -1 if axis in flips else 1
        index_map[axis, 3] = (section.start or 0) + (shape[axis] - 1 if axis in flips else 0)
    affine = np.asarray(grid["voxel_to_scanner_zyx"]) @ index_map
    values["geometry"] = {
        **grid, "shape_zyx": list(shape), "voxel_to_scanner_zyx": affine.tolist(),
        "center_zyx_mm": (affine @ np.r_[(np.asarray(shape) - 1) / 2, 1])[:3].tolist(),
    }
    values["phase_labels"] = meta["phase_labels"]
    return values


def sample_training(rows, split, patch, rng, device, *, flips=True):
    by_animal = {}
    for row in rows:
        if str(row["animal_id"]) in split["train"]:
            by_animal.setdefault(str(row["animal_id"]), []).append(row)
    if not by_animal:
        raise ValueError("empty training split")
    for row in (r for cases in by_animal.values() for r in cases):
        if any(p > n for p, n in zip(patch, row["shape"][1:])):
            raise ValueError(f"{row['case_id']}: requested real patch {patch} exceeds volume {row['shape'][1:]}")
    animal = rng.choice(sorted(by_animal))
    row = by_animal[animal][int(rng.integers(len(by_animal[animal])))]
    for _ in range(32):
        start = [int(rng.integers(n - p + 1)) for n, p in zip(row["shape"][1:], patch)]
        spatial = tuple(slice(a, a + n) for a, n in zip(start, patch))
        axes = tuple(i for i in (1, 2) if flips and rng.random() < 0.5)
        batch = read_case(row, spatial, flips=axes)
        if batch["valid"].any():
            return {key: torch.from_numpy(batch[key])[None, None].to(device)
                    for key in ("x", "a_s", "a_d", "y", "valid")}
    raise ValueError(f"{row['case_id']}: no valid training patch after 32 attempts")


def range_counts(array, valid, transform):
    low, high = transform["physical_range"]
    values = array[valid]
    return {
        "below_registered": int((values < low).sum()), "above_registered": int((values > high).sum()),
        "below_relu": int((normalize(values, transform) < 0).sum()), "valid_voxels": values.numel() if torch.is_tensor(values) else int(values.size),
    }
