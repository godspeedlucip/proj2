"""Support-constrained motion-compensated ordered-subset SART.

The TIGRE `matched` backprojector is an approximate pair for Siddon Ax on
this installation; the interpolation transpose itself uses exact weights.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pnmc.config import ReconstructionConfig
from pnmc.data.manifest import vector_values
from pnmc.data.schema import CaseManifest
from pnmc.data.volume_io import save_volume
from pnmc.motion.warp import warp_pull, warp_transpose


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _inverse_positive(values: torch.Tensor, relative_epsilon: float, name: str):
    if not torch.isfinite(values).all():
        raise ValueError(name + " normalization contains non-finite values")
    maximum = float(values.max())
    if maximum <= 0:
        raise ValueError(name + " normalization has no positive support")
    if float(values.min()) < -relative_epsilon * maximum:
        raise ValueError(name + " normalization unexpectedly negative")
    valid = values > relative_epsilon * maximum
    inverse = torch.zeros_like(values)
    inverse[valid] = 1 / values[valid]
    return inverse, valid, maximum


def reconstruct_target(case: CaseManifest, target_phase: str, fields: dict,
                       config: ReconstructionConfig, *, projector,
                       initial_mu: torch.Tensor, support_mask: torch.Tensor,
                       output: Path) -> tuple:
    """Reconstruct one target with all actual phase subsets in manifest order."""
    grid = case.reconstruction_grid
    if grid is None or case.projections is None or target_phase not in case.phase_ids:
        raise ValueError("physical case, measurements and target phase required")
    if (set(fields) != set(case.phase_ids) or
            tuple(initial_mu.shape) != grid.shape_zyx or
            tuple(support_mask.shape) != grid.shape_zyx or
            support_mask.dtype != torch.bool or not bool(support_mask.any())):
        raise ValueError("initial image, fields or nonempty Q0 support disagree")
    if not torch.isfinite(initial_mu).all():
        raise ValueError("initial volume must be finite")
    if not (0 < config.row_relative_epsilon < 1 and
            0 < config.column_relative_epsilon < 1 and
            config.zero_row_measurement_tolerance >= 0):
        raise ValueError("invalid MC normalization tolerances")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(getattr(projector, "device", initial_mu.device))
    dtype = initial_mu.dtype
    m = support_mask.to(device=device)
    m_float = m.to(dtype)
    x = initial_mu.to(device=device).clamp_min(0) * m_float
    field_by_phase = {}
    measurements = np.load(case.projections.path, mmap_mode="r", allow_pickle=False)
    if (measurements.ndim != 3 or measurements.shape[0] != len(vector_values(case.phase_id_per_view)) or
            not np.isfinite(measurements).all()):
        raise ValueError("invalid or non-finite physical measurements")
    subsets = []
    for phase in case.phase_ids:
        field = fields[phase]
        if (tuple(field.shape) != (3, *grid.shape_zyx) or
                not torch.isfinite(field).all()):
            raise ValueError("motion field shape or finite values invalid")
        field_by_phase[phase] = field.to(device=device, dtype=dtype)
        indices = np.asarray(case.view_indices_by_phase[phase], dtype=np.int64)
        if not len(indices):
            raise ValueError("empty phase subset: " + phase)
        y = torch.from_numpy(np.asarray(measurements[indices]).copy()).to(
            device=device, dtype=dtype)
        subsets.append((phase, indices, y))
    calls = {"forward": 0, "backward": 0}

    def forward(volume, field, indices):
        moved = warp_pull(volume, field, grid, grid)
        calls["forward"] += 1
        return projector.project(moved, indices).to(device=device, dtype=dtype)

    def backward(values, field, indices):
        calls["backward"] += 1
        back = projector.backproject(values, indices).to(device=device, dtype=dtype)
        return warp_transpose(back, field, grid, grid)

    def residual_norm(volume):
        squared = 0.
        for phase, indices, y in subsets:
            difference = y - forward(volume, field_by_phase[phase], indices)
            squared += float(torch.sum(difference.double() ** 2))
        return math.sqrt(squared)

    report = {"status": "running", "case_id": case.case_id,
              "target_phase": target_phase, "grid_sha256": grid.fingerprint(),
              "approximate_backprojection": True, "history": [],
              "projector_calls": calls}
    _write_json(output / "mc_report.json", report)
    try:
        normals = []
        for phase, indices, y in subsets:
            field = field_by_phase[phase]
            row = forward(m_float, field, indices)
            row_inverse, row_valid, row_max = _inverse_positive(
                row, config.row_relative_epsilon, "row")
            measurement_tolerance = (config.zero_row_measurement_tolerance *
                                     max(float(y.abs().max()), 1.))
            if bool((~row_valid & (y.abs() > measurement_tolerance)).any()):
                raise ValueError("nonzero measurement on zero row: support/geometry conflict")
            column = m_float * backward(torch.ones_like(y), field, indices)
            column_inverse, column_valid, _ = _inverse_positive(
                column[m], config.column_relative_epsilon, "column")
            full_column_inverse = torch.zeros_like(x)
            full_column_inverse[m] = column_inverse
            normals.append((row_inverse, full_column_inverse, int(row_valid.sum()),
                            int(column_valid.sum())))
        initial_residual = residual_norm(x)
        report["initial_residual_l2"] = initial_residual
        divergence_streak = 0
        for sweep in range(config.sweeps):
            started = time.perf_counter()
            previous = x.clone()
            phase_history = []
            clipped = 0
            for (phase, indices, y), (row_inverse, column_inverse,
                                      valid_rows, valid_columns) in zip(subsets, normals):
                field = field_by_phase[phase]
                residual = y - forward(x, field, indices)
                correction = backward(residual * row_inverse, field, indices)
                candidate = x + max(config.lambda_minimum,
                                    config.lambda_initial - config.lambda_decrement * sweep
                                    ) * correction * column_inverse
                clipped += int((candidate[m] < 0).sum())
                x = candidate.clamp_min(0) * m_float
                if not torch.isfinite(x).all():
                    raise ValueError("MC update diverged to non-finite values")
                phase_history.append({"phase": phase, "view_count": len(indices),
                                      "pre_update_residual_l2": float(torch.linalg.vector_norm(residual)),
                                      "valid_rows": valid_rows,
                                      "valid_columns": valid_columns})
            after = residual_norm(x)
            ratio = after / max(initial_residual, 1e-12)
            divergence_streak = divergence_streak + 1 if ratio > 100 else 0
            record = {"sweep": sweep,
                      "lambda": max(config.lambda_minimum,
                                    config.lambda_initial-config.lambda_decrement*sweep),
                      "phase_subsets": phase_history,
                      "measurement_residual_l2": after,
                      "relative_initial_residual": ratio,
                      "update_norm_l2": float(torch.linalg.vector_norm(x-previous)),
                      "nonnegative_truncation_fraction": clipped /
                      max(int(m.sum()) * len(subsets), 1),
                      "elapsed_seconds": time.perf_counter()-started,
                      "projector_calls": dict(calls)}
            report["history"].append(record)
            _write_json(output / "mc_report.json", report)
            if divergence_streak >= 3:
                raise ValueError("MC measurement residual diverged for three sweeps")
        final = x.detach().cpu()
        ref = save_volume(output / "final.npy", final, grid)
        report["final"] = ref.to_dict()
        report["status"] = "passed"
        report["projector_calls"] = dict(calls)
        _write_json(output / "mc_report.json", report)
        return final, report["history"]
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = "{}: {}".format(type(exc).__name__, exc)
        report["projector_calls"] = dict(calls)
        _write_json(output / "mc_report.json", report)
        raise
