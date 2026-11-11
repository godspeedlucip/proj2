"""Fixed PNMC comparison protocol and actual-ray measurement metrics."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import replace

from pnmc.config import NetworkConfig

from pnmc.data.manifest import _open_array, vector_values
from pnmc.motion.warp import warp_pull


def resolve_ablation_variants(spec: dict, base: NetworkConfig) -> list:
    """Lock shared data, epochs and seed; change at most one named factor."""
    allowed = {"prior_fusion", "use_non_local", "loss_name"}
    rows = []
    seen = set()
    for item in spec["variants"]:
        name = item["id"]
        overrides = item["overrides"]
        if name in seen or len(overrides) > 1 or set(overrides).difference(allowed):
            raise ValueError("ablation must change one factor at a time with unique IDs")
        seen.add(name)
        rows.append({"id": name, "changed_factor": next(iter(overrides), None),
                     "config": replace(base, **overrides)})
    if not rows or rows[0]["changed_factor"] is not None:
        raise ValueError("ablation needs a default baseline first")
    return rows


def validate_field_comparison(b4_registration: dict,
                              b5_registration: dict) -> None:
    """B4 and B5 may differ in image-derived fields, not registration rules."""
    if b4_registration.get("configuration") != b5_registration.get("configuration"):
        raise ValueError("B4/B5 registration configuration differs")
    for name in ("m0", "core", "taper"):
        if (b4_registration.get("support", {}).get(name, {}).get("sha256") !=
                b5_registration.get("support", {}).get(name, {}).get("sha256")):
            raise ValueError("B4/B5 support or taper differs")


def ablation_cost(channels, image_shape) -> dict:
    """Additional concat 1x1 fusion cost relative to the Add path."""
    if len(channels) != 5 or len(image_shape) != 2 or any(x <= 0 for x in image_shape):
        raise ValueError("ablation cost needs five widths and positive H,W")
    height, width = image_shape
    padded_height = ((height + 15)//16)*16
    padded_width = ((width + 15)//16)*16
    parameters = sum(2*c*c+c for c in channels)
    macs = sum((padded_height//(2**level)) *
               (padded_width//(2**level)) * 2*c*c
               for level, c in enumerate(channels))
    return {"concat_extra_parameters": parameters,
            "concat_extra_fusion_macs": macs,
            "image_shape_yx": list(image_shape),
            "note": "concat adds 1x1 fusion layers; capacity differs"}


def build_experiment_matrix(case, checkpoints: dict, config: dict) -> list:
    """Configure baselines without treating absent checkpoints as results."""
    if not case.phase_ids or case.projections is None:
        raise ValueError("experiment matrix needs a physical case")
    required = {"initial_sha256", "registration_sha256", "taper_sha256", "sweeps"}
    if required.difference(config) or int(config["sweeps"]) <= 0:
        raise ValueError("comparison configuration lacks common MC budget or hashes")
    views = list(range(len(vector_values(case.angles_rad))))
    common = {"case_id": case.case_id, "views": views,
              "projection_sha256": case.projections.sha256}
    rows = [
        {**common, "id": "B0", "method": "sparse_fdk", "status": "configured"},
        {**common, "id": "B1", "method": "legacy_priornet",
         "status": "configured" if checkpoints.get("legacy") else "not_run",
         "checkpoint": checkpoints.get("legacy")},
        {**common, "id": "B1m", "method": "priornet_matched",
         "status": "configured" if checkpoints.get("priornet_matched") else "not_run",
         "checkpoint": checkpoints.get("priornet_matched")},
        {**common, "id": "B2", "method": "pnnet",
         "status": "configured" if checkpoints.get("pnnet") else "not_run",
         "checkpoint": checkpoints.get("pnnet")},
    ]
    mc = {**common, "initial_sha256": config["initial_sha256"],
          "sweeps": int(config["sweeps"]),
          "registration_sha256": config["registration_sha256"],
          "taper_sha256": config["taper_sha256"],
          "lambda_initial": config.get("lambda_initial", 1.),
          "lambda_decrement": config.get("lambda_decrement", .01),
          "lambda_minimum": config.get("lambda_minimum", .1)}
    rows.extend([
        {**mc, "id": "B3", "method": "identity_mc", "field_source": "identity",
         "status": "configured" if checkpoints.get("pnnet") else "not_run"},
        {**mc, "id": "B4", "method": "matched_priornet_field_mc",
         "field_source": "priornet_matched", "status": "configured" if
         checkpoints.get("pnnet") and checkpoints.get("priornet_matched") else "not_run"},
        {**mc, "id": "B5", "method": "pnnet_field_mc",
         "field_source": "pnnet", "status": "configured" if
         checkpoints.get("pnnet") else "not_run"},
    ])
    return rows


def measurement_metrics(predictions: dict, case, *, projector,
                        support_mask: torch.Tensor, fields=None) -> dict:
    """Count each real measured ray once in the common image comparison."""
    if set(predictions) != set(case.phase_ids) or case.projections is None:
        raise ValueError("prediction phase set or measurements invalid")
    grid = case.reconstruction_grid
    if tuple(support_mask.shape) != grid.shape_zyx or support_mask.dtype != torch.bool:
        raise ValueError("measurement support and grid disagree")
    measured = np.asarray(_open_array(case.projections), dtype=np.float32)
    if measured.ndim != 3 or not np.isfinite(measured).all():
        raise ValueError("non-finite or invalid physical measurements")
    device = next(iter(predictions.values())).device
    mask = support_mask.to(device=device)

    def compare(volume, phase, field=None):
        if tuple(volume.shape) != grid.shape_zyx or not torch.isfinite(volume).all():
            raise ValueError("prediction shape or finite values invalid")
        indices = np.asarray(case.view_indices_by_phase[phase], dtype=np.int64)
        moving = (volume * mask).to(device)
        if field is not None:
            if tuple(field.shape) != (3, *grid.shape_zyx):
                raise ValueError("measurement field shape invalid")
            moving = warp_pull(moving, field.to(device), grid, grid)
        computed = projector.project(moving, indices)
        observed = torch.from_numpy(measured[indices].copy()).to(
            device=computed.device, dtype=computed.dtype)
        if computed.shape != observed.shape or not torch.isfinite(computed).all():
            raise ValueError("projected measurement shape or finite values invalid")
        squared = float(torch.sum((computed.double()-observed.double())**2))
        return {"squared_error": squared, "ray_count": int(observed.numel()),
                "mse": squared / observed.numel(), "view_count": len(indices)}

    by_phase = {phase: compare(predictions[phase], phase) for phase in case.phase_ids}
    total_squared = sum(item["squared_error"] for item in by_phase.values())
    rays = sum(item["ray_count"] for item in by_phase.values())
    result = {"all_measurements_mse": total_squared / rays,
              "squared_error": total_squared, "ray_count": rays,
              "by_phase": by_phase}
    if fields is not None:
        if set(fields) != set(case.phase_ids):
            raise ValueError("MC target fields incomplete")
        per_target = {}
        for target in case.phase_ids:
            if set(fields[target]) != set(case.phase_ids):
                raise ValueError("MC source fields incomplete")
            rows = {phase: compare(predictions[target], phase, fields[target][phase])
                    for phase in case.phase_ids}
            per_target[target] = {"mse": sum(row["squared_error"] for row in rows.values()) /
                                  rays, "ray_count_reused": rays, "by_phase": rows}
        result["mc_by_target"] = per_target
    return result
