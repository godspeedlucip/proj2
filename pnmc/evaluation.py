"""Physical image metrics and subject-balanced aggregation."""

from __future__ import annotations

import math
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data.manifest import _open_array, load_network_manifest, load_case_manifest
from .data.schema import ArrayRef
from .data.volume_io import load_volume
from .data.schema import NormalizationSpec
from .losses.ssim import ssim_loss


def _images(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim == 2:
        return tensor[None, None]
    if tensor.ndim == 3:
        return tensor[:, None]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor
    raise ValueError("image must be [H,W], [Z,H,W] or [B,1,H,W]")


def image_metrics(pred_mu, reference_mu, *, normalization: NormalizationSpec,
                  valid_mask=None) -> dict:
    if normalization.mode != "affine_clipped":
        raise ValueError("image metrics require a fixed affine physical interval")
    pred = _images(pred_mu)
    reference = _images(reference_mu)
    if pred.shape != reference.shape or not torch.isfinite(pred).all() or not torch.isfinite(reference).all():
        raise ValueError("prediction/reference shape or finite values disagree")
    mask = torch.ones_like(pred, dtype=torch.bool) if valid_mask is None else _images(valid_mask).bool()
    if mask.shape != pred.shape or not mask.any():
        raise ValueError("valid_mask shape mismatch or no valid pixels")
    difference = (pred - reference)[mask]
    mae = float(difference.abs().mean())
    mse = float(difference.square().mean())
    span = normalization.b - normalization.a
    # Keep out-of-range predictions visible in evaluation rather than clipping them.
    pred_network = 2 * (pred - normalization.a) / span - 1
    reference_network = 2 * (reference - normalization.a) / span - 1
    similarity = 1 - float(ssim_loss(pred_network, reference_network, mask))
    return {"mae": mae, "rmse": math.sqrt(mse), "ssim": similarity,
            "psnr": math.inf if mse == 0 else 20 * math.log10(span) - 10 * math.log10(mse),
            "unit": "1/mm", "valid_pixels": int(mask.sum())}


_METRICS = ("mae", "rmse", "ssim", "psnr")


def _average(rows):
    return {name: sum(float(row[name]) for row in rows) / len(rows)
            for name in _METRICS if all(name in row for row in rows)}


def aggregate_subject_metrics(records: list) -> dict:
    if not records:
        raise ValueError("cannot aggregate an empty evaluation")
    phase_slices = defaultdict(list)
    identities = set()
    for record in records:
        key = (record["subject_id"], record["case_id"],
               record["acquisition_id"], record["phase_id"])
        identity = key + (record["slice_index"],)
        if identity in identities:
            raise ValueError("duplicate evaluation slice: {}".format(identity))
        identities.add(identity)
        phase_slices[key].append(record)
    acquisition_phases = defaultdict(list)
    for key, rows in phase_slices.items():
        acquisition_phases[key[:3]].append(_average(rows))
    subject_acquisitions = defaultdict(list)
    acquisitions = []
    for key, rows in acquisition_phases.items():
        metrics = _average(rows)
        subject_acquisitions[key[0]].append(metrics)
        acquisitions.append({"subject_id": key[0], "case_id": key[1],
                             "acquisition_id": key[2], **metrics})
    subjects = {subject: _average(rows) for subject, rows in subject_acquisitions.items()}
    return {"overall": _average(list(subjects.values())), "subjects": subjects,
            "acquisitions": acquisitions, "n_subjects": len(subjects)}


def _json_metric_value(value):
    if isinstance(value, dict):
        return {key: _json_metric_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_metric_value(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value


def evaluate_run(run: Path, references: Path, output: Path, *,
                 projector=None, compute_measurements: bool = False) -> dict:
    run = Path(run).resolve()
    output = Path(output).resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("inference run did not pass")
    if "targets" in report:
        return _evaluate_physical_run(report, references, output,
                                      projector=projector,
                                      compute_measurements=compute_measurements)
    manifest = load_network_manifest(references)
    lookup = {(record.subject_id, record.case_id, record.acquisition_id,
               record.phase_id, record.slice_index): record for record in manifest.records}
    rows = []
    missing = []
    for volume in report["volumes"]:
        grid = manifest.grids[volume["grid_id"]]
        if grid.fingerprint() != volume["grid_sha256"]:
            raise ValueError("reference grid differs from prediction grid")
        prediction_ref = ArrayRef.from_dict(volume["prediction_mu"], run)
        prediction = load_volume(prediction_ref, grid)
        for index in range(grid.shape_zyx[0]):
            identity = (volume["subject_id"], volume["case_id"],
                        volume["acquisition_id"], volume["phase_id"], index)
            record = lookup.get(identity)
            if record is None or "label" not in record.arrays:
                missing.append(identity)
                continue
            label_ref = record.arrays["label"]
            if label_ref.unit != "1/mm" or label_ref.axes != "y,x":
                raise ValueError("reference label unit/axes are not physical y,x")
            if hashlib.sha256(label_ref.path.read_bytes()).hexdigest() != label_ref.sha256:
                raise ValueError("reference label sha256 mismatch")
            label = np.asarray(_open_array(label_ref), dtype=np.float32)
            if label.shape != grid.shape_zyx[1:]:
                raise ValueError("reference slice shape differs from grid")
            valid = None
            if "valid_mask" in record.arrays:
                valid_ref = record.arrays["valid_mask"]
                if hashlib.sha256(valid_ref.path.read_bytes()).hexdigest() != valid_ref.sha256:
                    raise ValueError("valid_mask sha256 mismatch")
                valid = np.asarray(_open_array(valid_ref))
            metrics = image_metrics(prediction[index], label,
                                    normalization=NormalizationSpec.from_dict(
                                        report["normalization"]), valid_mask=valid)
            rows.append({"subject_id": identity[0], "case_id": identity[1],
                         "acquisition_id": identity[2], "phase_id": identity[3],
                         "slice_index": index, **metrics})
    if missing:
        result = {"status": "not_run", "reason": "missing independent reference slices",
                  "missing": [list(key) for key in missing], "records": []}
    else:
        result = {"status": "passed", "records": rows,
                  "aggregate": aggregate_subject_metrics(rows),
                  "reference_kind": manifest.provenance.get("reference_kind", "unspecified")}
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(_json_metric_value(result),
                                                    ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    return result


def _binary_volume(ref: ArrayRef, grid) -> torch.Tensor:
    if ref.axes != "z,y,x" or ref.unit != "binary":
        raise ValueError("physical validity/ROI mask must be binary z,y,x")
    if hashlib.sha256(ref.path.read_bytes()).hexdigest() != ref.sha256:
        raise ValueError("physical mask SHA mismatch")
    values = np.asarray(_open_array(ref))
    if values.shape != grid.shape_zyx or not np.isfinite(values).all():
        raise ValueError("physical mask/grid mismatch")
    return torch.from_numpy(values.astype(bool, copy=True))


def _evaluate_physical_run(report: dict, references: Path, output: Path, *,
                           projector=None, compute_measurements=False) -> dict:
    case = load_case_manifest(references, require_physics=True)
    grid = case.reconstruction_grid
    if report.get("case_id") != case.case_id or set(report["targets"]) != set(case.phase_ids):
        raise ValueError("physical evaluation case or phase identities disagree")
    if report.get("normalization") is None:
        raise ValueError("physical evaluation needs fixed training normalization")
    normalization = NormalizationSpec.from_dict(report["normalization"])
    if "reference_by_phase" not in case.arrays:
        result = {"status": "not_run", "reason": "independent physical references missing",
                  "case_id": case.case_id}
    else:
        by_phase = {}
        roi_by_phase = {}
        rows = []
        predictions = {}
        skipped_empty_slices = []
        for phase in case.phase_ids:
            final = report["targets"][phase]["final"]
            prediction_ref = ArrayRef(Path(final["path"]), None, "z,y,x", "1/mm",
                                      final["sha256"])
            prediction = load_volume(prediction_ref, grid)
            predictions[phase] = prediction
            reference = load_volume(case.arrays["reference_by_phase"][phase], grid)
            valid = (_binary_volume(case.arrays["valid_by_phase"][phase], grid)
                     if "valid_by_phase" in case.arrays else None)
            valid_z = (valid.reshape(grid.shape_zyx[0], -1).any(dim=1)
                       if valid is not None else torch.ones(grid.shape_zyx[0], dtype=torch.bool))
            if not valid_z.any():
                raise ValueError("physical reference has no valid slices for phase " + phase)
            by_phase[phase] = image_metrics(prediction[valid_z], reference[valid_z],
                                            normalization=normalization,
                                            valid_mask=valid[valid_z] if valid is not None else None)
            roi_ref = case.arrays.get("roi_by_phase", {}).get(phase)
            if roi_ref is not None:
                roi = _binary_volume(roi_ref, grid)
                roi_valid_z = (roi & valid if valid is not None else roi).reshape(
                    grid.shape_zyx[0], -1).any(dim=1)
                if not roi_valid_z.any():
                    raise ValueError("physical ROI has no valid slices for phase " + phase)
                roi_by_phase[phase] = image_metrics(
                    prediction[roi_valid_z], reference[roi_valid_z],
                    normalization=normalization,
                    valid_mask=(roi if valid is None else roi & valid)[roi_valid_z])
            for index in range(grid.shape_zyx[0]):
                slice_valid = valid[index] if valid is not None else None
                if slice_valid is not None and not slice_valid.any():
                    skipped_empty_slices.append({"phase_id": phase, "slice_index": index})
                    continue
                metrics = image_metrics(prediction[index], reference[index],
                                        normalization=normalization,
                                        valid_mask=slice_valid)
                rows.append({"subject_id": case.subject_id,
                             "case_id": case.case_id,
                             "acquisition_id": case.acquisition_id,
                             "phase_id": phase, "slice_index": index, **metrics})
        result = {"status": "passed", "case_id": case.case_id,
                  "by_phase": by_phase, "records": rows,
                  "aggregate": aggregate_subject_metrics(rows),
                  "skipped_empty_reference_slices": skipped_empty_slices,
                  "roi_status": "passed" if roi_by_phase else "not_run",
                  "roi_by_phase": roi_by_phase,
                  "reference_kind": case.provenance.get("reference_kind", "unspecified"),
                  "measurement_metrics_status": "not_run"}
        if compute_measurements:
            from pnmc.data.manifest import vector_values
            from pnmc.experiments import measurement_metrics
            from pnmc.reconstruction.projector import TigreProjector
            from register_pnmc import _geometric_support

            if projector is None:
                projector = TigreProjector(case.geometry, grid,
                                           np.asarray(vector_values(case.angles_rad)),
                                           device="cuda:0")
            result["measurement_metrics"] = measurement_metrics(
                predictions, case, projector=projector,
                support_mask=_geometric_support(case))
            result["measurement_metrics_status"] = "passed"
            ray_count = result["measurement_metrics"]["ray_count"]
            mc_by_target = {}
            for phase in case.phase_ids:
                artifact = report["targets"][phase].get("report")
                if artifact is None:
                    continue
                if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
                    raise ValueError("MC history hash invalid")
                history = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
                if history.get("status") != "passed" or not history.get("history"):
                    raise ValueError("MC history incomplete")
                final_l2 = history["history"][-1]["measurement_residual_l2"]
                mc_by_target[phase] = {"all_phase_mse_reusing_rays":
                                       final_l2*final_l2/ray_count,
                                       "ray_count_reused": ray_count,
                                       "projector_calls": history.get("projector_calls")}
            if mc_by_target:
                result["mc_by_target"] = mc_by_target
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(_json_metric_value(result),
                                                    ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    return result
