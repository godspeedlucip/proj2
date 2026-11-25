from __future__ import annotations

from contextlib import ExitStack
import copy
from dataclasses import replace
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from .data import artifact_rows, atomic_path, case_rows, phase, validate_artifact
from .infer import array_options
from .settings import dependencies
from ..simulation.backend import backend_fingerprint, geometry, output_grid, project, reconstruct
from ..simulation.pipeline import read_raw_manifest
from ..simulation.reference import (SeriesSettings, TigreTool, apply_reconstruction_calibration,
                                    calibrated_volume)
from ..utils.io import atomic_json, atomic_jsonl, canonical_hash, read_json, sha256


def pigae(sparse, prior, reprojected, mode, coefficients=None):
    if mode == "static_consistent":
        image = prior + (sparse - reprojected)
    elif mode == "printed":
        image = sparse - (prior - reprojected)
    else:
        raise ValueError(f"Unknown PIGAE correction mode {mode}")
    return calibrate(image, coefficients)


def calibrate(image, coefficients):
    image = np.asarray(image, dtype=np.float32).copy()
    return image if coefficients is None else apply_reconstruction_calibration(image, np.poly1d(coefficients))


def read_views(row, source_geo, settings):
    path = row.get("projections") or row.get("acquisition")
    if not path:
        raise ValueError("Missing projections/acquisition HDF5 in the source manifest")
    expected = row.get("projections_sha256") if row.get("projections") else row.get("acquisition_sha256")
    if not expected or sha256(path) != expected:
        raise ValueError("Projections/acquisition SHA-256 mismatch")
    with h5py.File(path, "r") as file:
        required = {"projection_index", "angles_rad", "state_label"}
        if required - set(file):
            raise ValueError(f"View table missing {sorted(required - set(file))}; actual keys {list(file)}")
        stored_grid = json.loads(file.attrs.get("geometry_json", "{}"))
        if stored_grid != row["metadata"]["geometry"]:
            raise ValueError(f"Projection geometry_json conflicts with source grid; actual attributes {list(file.attrs)}")
        index, angles, state = (np.asarray(file[key]) for key in ("projection_index", "angles_rad", "state_label"))
        count = len(index)
        if (index.ndim != 1 or index.dtype.kind not in "iu" or np.any(index < 0)
                or np.any(index >= settings.n_projections) or len(np.unique(index)) != count):
            raise ValueError("Invalid or duplicate projection_index")
        if (angles.shape != (count,) or state.shape != (count,) or state.dtype.kind not in "iu"
                or not np.isfinite(angles).all() or set(state.tolist()) != set(row["metadata"]["phase_labels"])):
            raise ValueError("angles_rad/state_label conflict with projection_index or stored phase_labels")
        positions = np.argsort(index)
        index, angles, state = index[positions], angles[positions].astype(np.float32), state[positions]
        expected_angles = TigreTool.make_angles(settings.n_projections, 0, settings.angle_span_deg)[index]
        if not np.allclose(angles, expected_angles, rtol=0, atol=1e-6) and row["metadata"].get("cycle_sha256"):
            raise ValueError("angles_rad disagree with the recorded simulation protocol")
        mixed = np.asarray(file["mixed_line_integral"], dtype=np.float32)[positions] if "mixed_line_integral" in file else None
        if mixed is not None and (mixed.shape != (count, *source_geo.nDetector) or not np.isfinite(mixed).all()):
            raise ValueError("Invalid mixed_line_integral view or detector layout")
        # Optional geometry dataset rows are indexed by the explicit acquisition
        # index, independent of the storage order of mixed projections.
        per_view = {}
        if "view_geometry" in file:
            geometry_group = file["view_geometry"]
            keys = set(geometry_group) - {"projection_index"}
            allowed = {"offDetector": (2,), "rotDetector": (3,), "COR": (), "fdk_weight": ()}
            if keys - allowed.keys() or "projection_index" not in geometry_group:
                raise ValueError("Unsupported view_geometry or missing geometry projection_index")
            geo_indices = np.asarray(geometry_group["projection_index"])
            if (geo_indices.ndim != 1 or geo_indices.dtype.kind not in "iu"
                    or len(np.unique(geo_indices)) != len(geo_indices)):
                raise ValueError("Invalid view_geometry projection_index")
            lookup = {i: n for n, i in enumerate(geo_indices.tolist())}
            if any(i not in lookup for i in index):
                raise ValueError("Missing geometry for acquired view index")
            view_order = [lookup[i] for i in index]
            for key in sorted(keys):
                values = np.asarray(geometry_group[key], dtype=np.float32)
                if values.shape != (len(lookup), *allowed[key]) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid per-view geometry or weights: {key}")
                per_view[key] = values[view_order]
        if "fdk_weight" in per_view and np.any(per_view["fdk_weight"] <= 0):
            raise ValueError("FDK weights must be strictly positive for acquired views")
        keys = list(file)
        attributes = list(file.attrs)
    return index, angles, state, mixed, per_view, expected, keys, attributes


def subgeometry(geo, view_geometry, selection):
    output = copy.deepcopy(geo)
    for name, value in view_geometry.items():
        if name != "fdk_weight":
            if name == "COR" and np.all(value[selection] == 0):
                continue
            setattr(output, name, value[selection])
    return output


def fdk(projections, geo, angles, settings, view_geometry, selection):
    weights = view_geometry.get("fdk_weight")
    if weights is not None:
        projections = projections * weights[selection, None, None]
    result, _ = reconstruct(projections, subgeometry(geo, view_geometry, selection), angles, settings,
                            use_view_geometry=bool(set(view_geometry) - {"fdk_weight"}))
    return result


def replay(row, raw, geo, index, state, angles, view_geometry, settings, batch):
    meta = row["metadata"]
    if not raw or str(raw["animal_id"]) != str(row["animal_id"]):
        raise ValueError("Replay requires a matching --raw-manifest case and animal_id")
    paths = raw["phase_paths"]
    labels = meta["phase_labels"]
    if not set(map(str, labels)).issubset(paths):
        raise ValueError("Missing raw CT source phase path for replay")
    if not meta.get("cycle_sha256") or not meta.get("noise"):
        raise ValueError("Missing original simulation mapping/noise provenance for replay")
    expected_hashes = meta["source_hashes"]
    mixed = np.empty((len(index), *geo.nDetector), dtype=np.float32)
    for label in labels:
        selected = state == label
        path = paths[str(label)]
        if sha256(path) != expected_hashes.get(str(label)):
            raise ValueError(f"Replay CT hash mismatch at phase {label}")
        volume = nib.load(path)
        if tuple(volume.shape) != tuple(meta["source_shape_zyx"]) or not np.allclose(
            volume.affine, meta["source_affine"], rtol=0, atol=1e-5
        ):
            raise ValueError(f"Replay CT grid mismatch at phase {label}")
        mu = calibrated_volume(volume.dataobj, np.poly1d([1., 0.]), 0, 0, apply_air_threshold=False)
        mixed[selected] = project(mu, subgeometry(geo, view_geometry, selected), angles[selected], batch)
    noise = meta["noise"]
    if noise["kind"] == "poisson":
        if not np.array_equal(index, np.arange(settings.n_projections)):
            raise ValueError("Poisson replay requires the full saved view table for the original RNG stream")
        if "simulation_seed" in meta:
            seed = meta["simulation_seed"]
        elif raw.get("simulation_seed") is not None:
            seed = raw["simulation_seed"]
        else:
            raise ValueError("Replay requires original simulation_seed for Poisson noise")
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(meta["cache_key"][:8], 16)]))
        photons = noise["photons"]
        counts = rng.poisson(photons * np.exp(-mixed.astype(np.float64)))
        mixed = (-np.log(np.maximum(counts, 1) / photons)).astype(np.float32)
    elif noise["kind"] != "none":
        raise ValueError("Unsupported recorded simulation noise protocol")
    return mixed


def reconstruction_difference(actual, expected, atol, rtol):
    error = actual - expected
    return error, {
        "bias": float(error.mean(dtype=np.float64)), "max_abs": float(np.abs(error).max()),
        "rmse": float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
        "passed": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
    }


def build_case(config, row, raw, folder, backend, evidence):
    meta = row["metadata"]
    if meta["calibration_application"] not in ("none", "post_fdk"):
        raise ValueError("PIGAE requires raw mu with none/post_fdk calibration")
    required = {"settings", "source_shape_zyx", "source_spacing_zyx_mm",
                "used_spacing_zyx_mm", "source_offOrigin_zyx_mm", "prior_mode"}
    if required - meta.keys():
        raise ValueError(f"Missing original FDK settings/geometry {sorted(required - meta.keys())}; import an external correction")
    settings = SeriesSettings(**meta["settings"])
    original = geometry(meta["source_shape_zyx"], meta["source_spacing_zyx_mm"], settings)
    grid = output_grid(original, settings)
    if (grid != meta["geometry"] or not np.allclose(original.dVoxel, meta["used_spacing_zyx_mm"])
            or not np.allclose(original.offOrigin, meta["source_offOrigin_zyx_mm"])):
        raise ValueError("Original projection geometry/settings conflict with paired physical grid")
    index, angles, state, mixed, view_geometry, digest, keys, attributes = read_views(row, original, settings)
    mode = config["mode"]
    labels = meta["phase_labels"]
    if any((state == label).sum() < 2 for label in labels):
        raise ValueError("FDK requires at least two available views per phase")
    origin = "mixed_line_integral" if mixed is not None else "assigned_view_simulation_replay"
    if mixed is None:
        mixed = replay(row, raw, original, index, state, angles, view_geometry, settings, config["view_batch"])
    all_views = np.ones(len(index), dtype=bool)
    # PI_D on the target grid is kept distinct from an expanded prior used for Ax.
    prior = fdk(mixed, original, angles, settings, view_geometry, all_views)
    prior_for_ax, prior_grid = prior, grid
    if meta.get("full_fov_verified_from_source_support") is False:
        fov = original.sDetector * original.DSO / original.DSD
        native = np.minimum(np.maximum(np.floor(fov[1] / original.dVoxel[1:]).astype(int), 1), original.nVoxel[1:])
        wide_settings = replace(settings, fov_source_pixels=0, fov_output_pixels=int(min(native)))
        wide_grid = output_grid(original, wide_settings)
        extent = np.asarray(wide_grid["shape_zyx"]) * wide_grid["voxel_spacing_zyx_mm"]
        target_extent = np.asarray(grid["shape_zyx"]) * grid["voxel_spacing_zyx_mm"]
        if np.all(extent >= target_extent - 1e-5) and np.any(extent > target_extent + 1e-5):
            prior_for_ax = fdk(mixed, original, angles, wide_settings, view_geometry, all_views)
            prior_grid = wide_grid
    prior_geo = TigreTool.build_tigre_geometry(
        prior_for_ax.shape, prior_grid["voxel_spacing_zyx_mm"], n_detector=original.nDetector,
        d_detector=original.dDetector, dso=original.DSO, dsd=original.DSD)
    prior_geo.offOrigin = np.asarray(prior_grid["center_zyx_mm"], dtype=np.float32)
    coefficients = meta["calibration"]["coefficients"]
    details = {
        "mode": mode, "provenance": origin, "calibration_order": "after_pigae", "intensity_units": "mm^-1",
        "source_sha256": row["sha256"], "input_sha256s": {"view_file": digest},
        "backend": backend, "projection_keys": keys, "projection_attributes": attributes,
        "phase_views": {str(label): index[state == label].tolist() for label in labels},
        "view_geometry_fields": sorted(view_geometry), "prior_R_mode": meta["prior_mode"],
        "prior_D_mode": "all_views_fdk", "prior_D_projection_grid": prior_grid,
        "target_fov_verified_from_source_support": meta.get("full_fov_verified_from_source_support"),
        "tolerance": {"atol": config["atol"], "rtol": config["rtol"], **evidence},
    }
    if origin == "assigned_view_simulation_replay":
        details["input_sha256s"].update({f"ct_phase_{label}": meta["source_hashes"][str(label)] for label in labels})
    folder.mkdir(parents=True, exist_ok=True)
    output, error_path = folder / "correction.h5", folder / "reconstruction_errors.h5"
    entries = []
    shape = tuple(row["shape"])
    options = array_options(shape)
    with ExitStack() as stack:
        temporary = stack.enter_context(atomic_path(output))
        errors = stack.enter_context(atomic_path(error_path))
        dest = stack.enter_context(h5py.File(temporary, "w"))
        error_file = stack.enter_context(h5py.File(errors, "w"))
        paired = stack.enter_context(h5py.File(row["path"], "r"))
        correction = dest.create_dataset("correction", shape=shape, dtype="f4", **options)
        error = error_file.create_dataset("cbct_difference", shape=shape, dtype="f4", **options)
        weighted_prior = np.zeros_like(prior)
        output_range = [float("inf"), float("-inf")]
        for i, label in enumerate(labels):
            selected = state == label
            xs = fdk(mixed[selected], original, angles[selected], settings, view_geometry, selected)
            if meta["prior_mode"] == "phase_weighted_fdk":
                weighted_prior += xs * (int(selected.sum()) / len(state))
            delta, metrics = reconstruction_difference(calibrate(xs, coefficients), phase(paired["cbct"], i),
                                                        config["atol"], config["rtol"])
            entries.append({"phase_label": label, "views": int(selected.sum()), **metrics})
            error[i] = delta
            projected = project(prior_for_ax, subgeometry(prior_geo, view_geometry, selected),
                                angles[selected], config["view_batch"])
            qs = fdk(projected, original, angles[selected], settings, view_geometry, selected)
            cs = pigae(xs, prior, qs, mode, coefficients)
            if not np.isfinite(cs).all():
                raise ValueError(f"Nonfinite correction at phase {label}")
            correction[i] = cs
            output_range[0] = min(output_range[0], float(cs.min()))
            output_range[1] = max(output_range[1], float(cs.max()))
        if meta["prior_mode"] not in ("phase_weighted_fdk", "all_views_fdk"):
            raise ValueError("Unknown frozen prior_R mode")
        prior_r = weighted_prior if meta["prior_mode"] == "phase_weighted_fdk" else prior
        reference = np.asarray(paired["prior"], dtype=np.float32)
        delta, metrics = reconstruction_difference(calibrate(prior_r, coefficients), reference,
                                                    config["atol"], config["rtol"])
        error_file.create_dataset("prior_difference", data=delta, compression="gzip", fletcher32=True)
        details.update(phase_reconstruction=entries, prior_reconstruction=metrics, correction_range=output_range,
                       eligible=all(p["passed"] for p in entries) and metrics["passed"])
        error_file.attrs["metadata_json"] = json.dumps({**meta, "pigae": details}, ensure_ascii=False)
        dest.attrs["metadata_json"] = json.dumps({**meta, "pigae": details}, ensure_ascii=False)
    return output, error_path, details


def prepare_moco(config, manifest, out, *, raw_manifest=None, external=None):
    sources, _, _ = dependencies(config)
    out = Path(out).resolve()
    if "c" not in sources:
        result = {"needs_correction": False, "cases": []}
        atomic_json(out / "summary.json", result)
        return result
    if not manifest:
        raise ValueError("prepare-moco needs --manifest when outputs depend on c")
    if external and raw_manifest:
        raise ValueError("--external-c and --raw-manifest are alternative correction paths")
    rows = case_rows(manifest)
    mode = config["pigae"]["mode"]
    records, cases = [], []
    external_rows = artifact_rows(external) if external else {}
    raw_by_case, backend, evidence = {}, None, None
    if not external:
        settings = config["pigae"]
        if settings["atol"] is None or settings["rtol"] is None or not settings["tolerance_evidence"]:
            raise ValueError("Register pigae.atol, pigae.rtol and pigae.tolerance_evidence before reconstructing")
        evidence_file = Path(settings["tolerance_evidence"])
        evidence = {"evidence_file": str(evidence_file), "evidence_sha256": sha256(evidence_file)}
        if raw_manifest:
            raw_by_case = {r["case_id"]: r for r in read_raw_manifest(raw_manifest)}
        backend = backend_fingerprint()
    context = out / "preparation_context.json"
    inputs = {"source_manifest_sha256": sha256(manifest), "mode": mode,
              "config": config["pigae"], "tolerance_evidence": evidence,
              "raw_manifest_sha256": sha256(raw_manifest) if raw_manifest else None,
              "external": external_rows or None, "backend": backend,
              "prepare_code_sha256": sha256(__file__)}
    signature = canonical_hash(inputs)
    if context.exists() and read_json(context)["signature"] != signature:
        raise ValueError("Output directory belongs to a different prepare-moco request")
    atomic_json(context, {**inputs, "signature": signature})
    for row in rows:
        case = row["case_id"]
        try:
            if external:
                item = external_rows.get(case)
                details = validate_artifact(item, row, "c", mode)
                records.append(item)
                cases.append({"case_id": case, "eligible": True, "pigae": details})
                continue
            folder = out / case
            output, errors, details = build_case(config["pigae"], row, raw_by_case.get(case), folder,
                                                 backend, evidence)
            cases.append({"case_id": case, "eligible": details["eligible"], "pigae": details,
                          "errors": str(errors), "errors_sha256": sha256(errors)})
            if details["eligible"]:
                records.append({"case_id": case, "animal_id": row["animal_id"], "path": str(output),
                                "sha256": sha256(output), "source_sha256": row["sha256"],
                                "source_metadata_hash": row["source_metadata_hash"]})
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            cases.append({"case_id": case, "eligible": False, "error": str(exc)})
    artifact = out / "manifest.jsonl"
    atomic_jsonl(artifact, records)
    result = {"needs_correction": True, "eligible_cases": len(records), "cases": cases,
              "manifest": str(artifact) if records else None}
    atomic_json(out / "summary.json", result)
    if not records:
        reasons = [item.get("error", "FDK tolerance failed") for item in cases]
        raise ValueError(f"No correction cases passed: {reasons}. See {out / 'summary.json'}")
    return result
