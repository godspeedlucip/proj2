from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .config import atomic_path, file_digest, fingerprint, read_json, software, write_json
from .data import load_array, load_manifest, view_assignments
from .physics import TigreProjector, pigae


def save_array(path, value):
    with atomic_path(path) as temporary, temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)


def central_roi(volume):
    return volume[tuple(slice(n // 3, max(n // 3 + 1, (2 * n) // 3)) for n in volume.shape)]


def roi_mean(volume):
    return float(central_roi(volume).mean(dtype=np.float64))


def _stats(volume):
    roi = central_roi(volume)
    return {"roi_mean": roi_mean(volume), "roi_std": float(roi.std(dtype=np.float64)),
            "min": float(volume.min()), "max": float(volume.max())}


def prepare_scan(scan, config, output, *, training):
    projections = load_array(scan["projections"])
    if projections.ndim != 3 or projections.dtype != np.float32 or len(projections) == 0:
        raise ValueError(f"Expected float32 [view,v,u] projections: {scan['scan_id']}")
    angles, all_views, bins, rows = view_assignments(scan, len(projections), training=training)
    fdk = {**config["fdk"], **scan.get("fdk", {})}
    if fdk["parker"] or fdk["angle_scale"] != "2*pi/n_views" or fdk["filter"] != "ram_lak":
        raise ValueError("Unsupported per-scan FDK settings")
    versions = software()
    settings = {"schema": 1, "projection_sha256": file_digest(scan["projections"]),
                "views_sha256": file_digest(scan["views"]), "geometry": scan["geometry"],
                "grid": scan["grid"], "phase_base": scan.get("phase_base", 0),
                "n_phases": scan["n_phases"], "pigae_mode": config["pigae_mode"],
                "training": training, "fdk": fdk, "tigre": versions["tigre"]}
    cache_id = fingerprint(settings)
    case_base = Path(output) / scan["scan_id"]
    legacy = case_base / cache_id
    root = legacy if (legacy / "prepared.json").is_file() else case_base / cache_id[:24]
    metadata_path = root / "prepared.json"
    if metadata_path.is_file():
        record = read_json(metadata_path)
        if record["cache_id"] == cache_id and all((root / location).is_file() for location in record["files"]):
            return metadata_path
    projector = TigreProjector(scan["grid"], scan["geometry"], angles, fdk)
    root.mkdir(parents=True, exist_ok=True)
    scan_id = scan["scan_id"]
    print(f"{scan_id}: " + ", ".join(f"{name}={[len(x) for x in groups]}" for name, groups in bins.items()), flush=True)
    table = root / "views.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with atomic_path(table) as temporary, temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selected = projections if len(all_views) == len(projections) else projections[all_views]
    prior = projector.fdk(selected, all_views)
    del selected
    save_array(root / "pi.npy", prior)
    files = ["pi.npy", "views.csv"]
    prior_slice = prior[len(prior) // 2]
    low, high = (float(v) for v in np.percentile(prior_slice, [1, 99]))
    high = max(high, low + 1e-6)
    branches = {}
    for name, groups in bins.items():
        phases = []
        for phase, views in enumerate(groups):
            folder = Path(name) / f"phase_{phase:02d}"
            x = projector.fdk(projections[views], views)
            q = projector.fdk(projector.project(prior, views), views)
            s = prior - q
            c = pigae(x, prior, q, config["pigae_mode"])
            paths = {"x": (folder / "x.npy").as_posix(), "c": (folder / "c.npy").as_posix(),
                     "slices": (folder / "pi_x_q_s_c.png").as_posix()}
            save_array(root / paths["x"], x)
            save_array(root / paths["c"], c)
            images = np.concatenate([v[len(v) // 2] for v in (prior, x, q, s, c)], axis=1)
            pixels = (np.clip((images - low) / (high - low), 0, 1) * 255).astype(np.uint8)
            with atomic_path(root / paths["slices"]) as temporary:
                Image.fromarray(pixels).save(temporary, format="PNG")
            phases.append({"phase_id": phase, "views": views.tolist(), **paths,
                           "stats": {key: _stats(value) for key, value in zip("xqsc", (x, q, s, c))}})
            files.extend(paths.values())
            print(f"{scan_id}: {name} phase {phase:02d} prepared", flush=True)
            del x, q, s, c
        branches[name] = phases
    metadata = {**{key: scan[key] for key in ("patient_id", "scan_id", "domain", "split")},
                "cache_id": cache_id, "grid": scan["grid"], "units": "mm^-1", "pi": "pi.npy",
                "pi_stats": _stats(prior), "branches": branches, "files": files,
                "views": "views.csv", "valid_views": all_views.tolist(), "settings": settings,
                "source_projections": str(scan["projections"]), "software": versions,
                "slice_display": {"axis": "z", "index": len(prior) // 2, "window": [low, high],
                                  "order": ["pi", "x", "q", "s", "c"]}}
    write_json(metadata_path, metadata)
    return metadata_path


def prepare(manifest, config, output):
    output = Path(output).resolve()
    cases = [prepare_scan(scan, config, output, training=True)
             for scan in list(load_manifest(manifest))]
    write_json(output / "prepared.json", {"config": config,
               "cases": [case.relative_to(output).as_posix() for case in cases]})


def load_prepared(output):
    root = Path(output).resolve()
    index = read_json(root / "prepared.json")
    cases = []
    for name in index["cases"]:
        path = root / name
        record = read_json(path)
        if not all((path.parent / file).is_file() for file in record["files"]):
            raise ValueError(f"Prepared case is incomplete: {path}")
        cases.append((path.parent, record))
    return index["config"], cases
