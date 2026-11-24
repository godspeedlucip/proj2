from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import read_json


def load_manifest(path):
    path = Path(path).resolve()
    scans = read_json(path)["scans"]
    if not scans:
        raise ValueError("Empty scan manifest")
    seen, splits = set(), {}
    for original in scans:
        scan = dict(original)
        subject, name, domain, split = (scan[key] for key in ("patient_id", "scan_id", "domain", "split"))
        if not all(isinstance(x, str) and x.strip() for x in (subject, name, domain, split)):
            raise ValueError("Patient, scan, domain and split must be nonempty strings")
        if name in {".", ".."} or any(c in name for c in '<>:"/\\|?*') or name in seen:
            raise ValueError(f"Scan IDs must be unique safe path components: {name}")
        seen.add(name)
        if splits.setdefault(subject, split) != split:
            raise ValueError(f"Patient crosses data splits: {subject}")
        for key in ("projections", "views"):
            scan[key] = (path.parent / scan[key]).resolve()
        validate_grid(scan["grid"])
        if int(scan["n_phases"]) != scan["n_phases"] or scan["n_phases"] < 1:
            raise ValueError("n_phases must be a positive integer")
        yield scan


def validate_grid(grid):
    shape = np.asarray(grid["shape_zyx"])
    spacing = np.asarray(grid["spacing_xyz_mm"])
    origin = np.asarray(grid["origin_xyz_mm"])
    direction = np.asarray(grid["direction_xyz"])
    if (any(x.shape != (3,) or not np.isfinite(x).all() for x in (shape, spacing, origin))
            or np.any(shape < 1) or np.any(shape != np.floor(shape)) or np.any(spacing <= 0)
            or direction.shape != (3, 3) or not np.allclose(direction.T @ direction, np.eye(3))):
        raise ValueError("Invalid volume grid")


def view_assignments(scan, count, *, training):
    with scan["views"].open(newline="", encoding="utf-8-sig") as stream:
        label_rows = list(csv.DictReader(stream))
    rows_by_id = {int(row.get("view_id", row.get("projection_index"))): row for row in label_rows}
    if (len(label_rows) != count or set(rows_by_id) != set(range(count))):
        raise ValueError("View CSV must label each original projection index exactly once")
    rows = [dict(rows_by_id[index]) for index in range(count)]
    use_radians = "angle_rad" in rows[0]
    angles = np.asarray([float(row["angle_rad" if use_radians else "angle_deg"]) for row in rows])
    if not use_radians:
        angles = np.deg2rad(angles)
    if not np.isfinite(angles).all():
        raise ValueError("Non-finite angle")
    phases, cycles, ids = [], [], []
    for view, (row, angle) in enumerate(zip(rows, angles)):
        flag = row.get("valid", "true").strip().lower()
        if flag not in {"0", "false", "1", "true"}:
            raise ValueError(f"Invalid valid flag at view {view}: {flag}")
        row.update(view_id=view, angle_rad=float(angle), valid=flag in {"true", "1"},
                   source_phase=row.get("phase_id", row.get("phase_label")),
                   source_cycle=row.get("cycle_id", row.get("cycle")),
                   phase_id=None, cycle_id=None, pseudo_bin=None)
        if not row["valid"]:
            continue
        ids.append(view)
        phases.append(int(rows_by_id[view].get("phase_id", rows_by_id[view].get("phase_label")))
                      - scan.get("phase_base", 0))
        if training:
            cycles.append(int(rows_by_id[view].get("cycle_id", rows_by_id[view].get("cycle"))))
    if not ids:
        raise ValueError("No valid projection views")
    if training and any(a > b for a, b in zip(cycles, cycles[1:])):
        raise ValueError("Cycles must follow acquisition order")
    cycles = np.asarray(cycles, dtype=np.int64) - cycles[0] if training else None
    phases = np.asarray(phases, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    bins = phase_bins(ids, phases, cycles, scan["n_phases"])
    for index, (view, phase) in enumerate(zip(ids, phases)):
        rows[view]["phase_id"] = int(phase)
        if training:
            rows[view]["cycle_id"] = int(cycles[index])
            rows[view]["pseudo_bin"] = int((phase - cycles[index]) % scan["n_phases"])
    return angles.astype(np.float32), ids, bins, rows


def phase_bins(ids, phases, cycles, count):
    phases = np.asarray(phases)
    if np.any(phases < 0) or np.any(phases >= count):
        raise ValueError("Phase ID outside configured range")
    labels = {"real": phases}
    if cycles is not None:
        labels["pseudo"] = (phases - cycles) % count
    result = {name: [ids[labels_by_view == phase] for phase in range(count)]
              for name, labels_by_view in labels.items()}
    for name, groups in result.items():
        empty = [index for index, views in enumerate(groups) if not len(views)]
        if empty:
            raise ValueError(f"Empty {name} phase bins: {empty}")
    return result


def load_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)
