"""Register every ordered phase pair on the physical reconstruction grid."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pnmc.config import RegistrationConfig
from pnmc.data.grids import canonicalize_to_scanner
from pnmc.data.manifest import vector_values
from pnmc.data.schema import ArrayRef, GridSpec
from pnmc.data.volume_io import load_volume
from pnmc.motion.qc import motion_support, quality_check
from pnmc.motion.registration import register
from pnmc.reconstruction.projector import TigreProjector


def _save_array(path: Path, values: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values.detach().cpu().numpy(), allow_pickle=False)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def register_pairs(phase_ids: list, volumes: dict, grid: GridSpec,
                   config: RegistrationConfig, m0: torch.Tensor,
                   core: torch.Tensor, taper: torch.Tensor, output: Path) -> dict:
    """Write final tapered fields; a QC failure leaves a failed report and raises."""
    output = Path(output).resolve()
    if len(phase_ids) != len(set(phase_ids)) or set(volumes) != set(phase_ids):
        raise ValueError("enhanced phase identities disagree")
    if any(tuple(volumes[phase].shape) != grid.shape_zyx for phase in phase_ids):
        raise ValueError("enhanced volume grid mismatch")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "direction": "target_fixed_to_source_moving",
              "field_unit": "mm", "components": "xyz", "grid": grid.to_dict(),
              "grid_sha256": grid.fingerprint(), "configuration": asdict(config),
              "pairs": [], "support": {}}
    for name, values in (("m0", m0), ("core", core), ("taper", taper)):
        path = output / (name + ".npy")
        report["support"][name] = {"path": str(path),
                                    "sha256": _save_array(path, values)}
    _write_json(output / "registration.json", report)
    try:
        for target_index, target in enumerate(phase_ids):
            for source_index, source in enumerate(phase_ids):
                if target == source:
                    field = torch.zeros((3, *grid.shape_zyx), dtype=torch.float32)
                    details = {"method": "identity", "levels": []}
                else:
                    raw, details = register(volumes[source], volumes[target], grid,
                                            config)
                    field = raw * taper.unsqueeze(0).to(raw)
                    field[:, ~m0] = 0
                    details = {**details, "method": "diffeomorphic_demons"}
                qc = quality_check(field, grid=grid, m0=m0, core=core)
                path = output / ("field_{:02d}_{:02d}.npy".format(
                    target_index, source_index))
                digest = _save_array(path, field)
                report["pairs"].append({"target": target, "source": source,
                                        "path": str(path), "sha256": digest,
                                        "shape": list(field.shape), "unit": "mm",
                                        "components": "xyz", "grid_sha256": grid.fingerprint(),
                                        "qc": qc, "registration": details})
                _write_json(output / "registration.json", report)
        report["status"] = "passed"
        _write_json(output / "registration.json", report)
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = "{}: {}".format(type(exc).__name__, exc)
        _write_json(output / "registration.json", report)
        raise
    return report


def load_enhanced_volumes(path: Path, case) -> tuple:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "passed" or "volumes" not in raw:
        raise ValueError("enhanced inference report must have passed status")
    by_phase = {}
    identity = (case.subject_id, case.case_id, case.acquisition_id)
    for item in raw["volumes"]:
        if (item.get("subject_id"), item.get("case_id"),
                item.get("acquisition_id")) != identity:
            continue
        phase = str(item["phase_id"])
        if phase in by_phase or phase not in case.phase_ids:
            raise ValueError("duplicate or unexpected enhanced phase")
        if item["grid_sha256"] != case.reconstruction_grid.fingerprint():
            raise ValueError("enhanced grid differs from physical case")
        ref = ArrayRef.from_dict(item["prediction_mu"], path.parent)
        by_phase[phase] = (load_volume(ref, case.reconstruction_grid), ref.sha256)
    if set(by_phase) != set(case.phase_ids):
        raise ValueError("enhanced report lacks a physical case phase")
    return ({phase: by_phase[phase][0] for phase in case.phase_ids},
            {phase: by_phase[phase][1] for phase in case.phase_ids}, raw)


def geometric_support(case) -> torch.Tensor:
    source = case.projection_volume_grid
    target = case.reconstruction_grid
    return canonicalize_to_scanner(torch.ones(source.shape_zyx), source,
                                   target) > 0


def register_case(case, enhanced_report: Path, config: RegistrationConfig,
                  output: Path, *, device="cuda:0", projector=None) -> dict:
    volumes, input_hashes, enhanced = load_enhanced_volumes(enhanced_report, case)
    if projector is None:
        projector = TigreProjector(case.geometry, case.reconstruction_grid,
                                   np.asarray(vector_values(case.angles_rad)),
                                   device=device)
    sensitivities = []
    for phase in case.phase_ids:
        indices = np.asarray(case.view_indices_by_phase[phase], dtype=np.int64)
        ones = torch.ones((len(indices), *case.geometry["nDetector_vu"]),
                          dtype=torch.float32)
        sensitivities.append(projector.backproject(ones, indices).cpu())
    m0, core, taper = motion_support(sensitivities, geometric_support(case),
                                     case.reconstruction_grid,
                                     threshold=config.support_threshold,
                                     taper_mm=config.taper_mm)
    report = register_pairs(case.phase_ids, volumes, case.reconstruction_grid,
                            config, m0, core, taper, output)
    report["case_id"] = case.case_id
    report["case_manifest_sha256"] = case.manifest_sha256
    report["projection_sha256"] = case.projections.sha256
    report["enhanced_report_sha256"] = hashlib.sha256(
        Path(enhanced_report).read_bytes()).hexdigest()
    report["enhanced_checkpoint_sha256"] = enhanced.get("checkpoint_sha256")
    report["enhanced_volume_sha256_by_phase"] = input_hashes
    _write_json(Path(output).resolve() / "registration.json", report)
    return report


