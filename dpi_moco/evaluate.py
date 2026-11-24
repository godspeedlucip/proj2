from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from .config import read_json, write_json
from .data import load_array


def volume_metrics(prediction, reference, protocol):
    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError("Evaluation requires matching [Z,Y,X] arrays")
    size, sigma, scale = (protocol[key] for key in ("ssim_window", "ssim_sigma", "data_range"))
    if size < 1 or size % 2 == 0 or min(prediction.shape[1:]) < size or min(sigma, scale) <= 0:
        raise ValueError("Invalid SSIM window, sigma or dynamic range")
    radius = size // 2
    valid = slice(radius, -radius) if radius else slice(None)
    c1, c2 = (0.01 * scale) ** 2, (0.03 * scale) ** 2
    squared_error, ssim = 0.0, 0.0

    def blur(image):
        return gaussian_filter(image, sigma, truncate=radius / sigma)[valid, valid]

    for a, b in zip(prediction, reference):
        a, b = a.astype(np.float64), b.astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Non-finite evaluation volume")
        squared_error += np.square(a - b).sum()
        ma, mb = blur(a), blur(b)
        va, vb = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2
        cov = blur(a * b) - ma * mb
        ssim += float(((2 * ma * mb + c1) * (2 * cov + c2)
                       / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())
    rmse = math.sqrt(squared_error / prediction.size)
    return {"rmse": rmse, "psnr": 20 * math.log10(scale / rmse) if rmse else "inf",
            "ssim": ssim / len(prediction)}


def _summarize(records):
    result = {}
    for name in ("rmse", "psnr", "ssim"):
        values = [float(record[name]) for record in records]
        result[name] = {"mean": float(np.mean(values)) if all(map(math.isfinite, values)) else "inf",
                        "std": float(np.std(values)) if all(map(math.isfinite, values)) else None}
    return result


def evaluate(args):
    root = Path(args.pred).resolve()
    reference_path = Path(args.reference).resolve()
    protocol = {"data_range": 0.02, "ssim_window": 11, "ssim_sigma": 1.5}
    if args.protocol:
        override = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8-sig"))
        if override.keys() - protocol.keys():
            raise ValueError(f"Unknown metrics options: {override.keys() - protocol.keys()}")
        protocol.update(override)
    prediction = read_json(root / "predictions.json")
    refs = {(case["scan_id"], phase["phase_id"]): phase
            for case in read_json(reference_path)["scans"] for phase in case["phases"]}
    patients, phases = {}, []
    for scan in prediction["scans"]:
        patient = scan["patient_id"]
        for p in scan["phases"]:
            key = (scan["scan_id"], p["phase_id"])
            ref = refs.get(key)
            if ref is None or ref["grid"] != p["grid"] or ref["units"] != p["units"]:
                raise ValueError(f"Reference must match the prediction grid and units: {key}")
            truth = load_array(reference_path.parent / ref["path"])
            for stage in args.stages:
                path = (root / p["path"]).with_name(f"{stage}.npy")
                metrics = volume_metrics(load_array(path), truth, protocol)
                phases.append({"patient_id": patient, "scan_id": key[0], "phase_id": key[1],
                               "stage": stage, **metrics})
                patients.setdefault(patient, {}).setdefault(stage, []).append(metrics)
    patient_means = {patient: {stage: _summarize(rows) for stage, rows in stages.items()}
                     for patient, stages in patients.items()}
    overall = {stage: _summarize([{metric: entry[stage][metric]["mean"] for metric in ("rmse", "psnr", "ssim")}
                                  for entry in patient_means.values()]) for stage in args.stages}
    path = Path(args.out) if args.out else root / "metrics.json"
    write_json(path, {"config": prediction["config"], "protocol": {**protocol, "ssim_dimensions": "axial_2d",
                "ssim_window_boundary": "valid", "roi": "entire_grid", "covariance": "population"},
                "reference_manifest": str(reference_path), "phases": phases, "patients": patient_means,
                "overall": overall})
    print(f"Wrote {len(phases)} phase evaluations: {path}")
