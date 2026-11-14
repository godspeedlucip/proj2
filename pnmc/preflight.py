"""Validate physical PNMC inputs before FDK or MC-OSSART starts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import pickle

import torch

from pnmc.config import NetworkConfig, ReconstructionConfig, RegistrationConfig
from pnmc.data.manifest import load_case_manifest, validate_case
from pnmc.data.schema import NormalizationSpec
from pnmc.models.pnnet import PNNet2D
from pnmc.motion.registration import _import_sitk
from pnmc.reconstruction.projector import _import_tigre


def _check_dependencies() -> dict:
    tigre = _import_tigre()
    sitk = _import_sitk()
    return {"tigre": getattr(tigre, "__version__", "available"),
            "SimpleITK": getattr(sitk, "Version_VersionString", lambda: "available")()}


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_runtime_config(reconstruction: ReconstructionConfig,
                             registration: RegistrationConfig) -> None:
    def finite(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))

    if type(reconstruction.sweeps) is not int or reconstruction.sweeps <= 0:
        raise ValueError("reconstruction sweeps must be a positive integer")
    if (any(not finite(value) for value in (
            reconstruction.lambda_initial, reconstruction.lambda_decrement,
            reconstruction.lambda_minimum)) or
            reconstruction.lambda_initial <= 0 or
            reconstruction.lambda_decrement < 0 or
            reconstruction.lambda_minimum <= 0):
        raise ValueError("invalid reconstruction lambda schedule")
    if (type(reconstruction.fdk_padding) is not int or
            reconstruction.fdk_padding < 0):
        raise ValueError("FDK detector padding must be a non-negative integer")
    if (not finite(reconstruction.row_relative_epsilon) or
            not 0 < reconstruction.row_relative_epsilon < 1 or
            not finite(reconstruction.column_relative_epsilon) or
            not 0 < reconstruction.column_relative_epsilon < 1 or
            not finite(reconstruction.zero_row_measurement_tolerance) or
            reconstruction.zero_row_measurement_tolerance < 0):
        raise ValueError("invalid MC normalization tolerances")
    if (not finite(registration.support_threshold) or
            not 0 < registration.support_threshold < 1):
        raise ValueError("invalid motion support threshold")
    if (any(type(value) is not int or value <= 0
            for value in (*registration.shrink_factors, *registration.iterations)) or
            any(not finite(value) or value < 0
                for value in registration.image_sigma_mm) or
            not finite(registration.field_sigma_mm) or
            registration.field_sigma_mm < 0 or
            not finite(registration.taper_mm) or registration.taper_mm <= 0):
        raise ValueError("invalid registration scale or sigma")


def check_reconstruction_inputs(case_path: Path, checkpoint: Path,
                                network_config_path: Path,
                                reconstruction_config_path: Path, *,
                                device: str) -> dict:
    """Check files, model provenance, grid, measurements and runtime dependencies."""
    try:
        parsed_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("invalid device: " + str(device)) from exc
    if parsed_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    if (parsed_device.type == "cuda" and parsed_device.index is not None and
            parsed_device.index >= torch.cuda.device_count()):
        raise ValueError("CUDA device index is unavailable: " + str(device))
    if parsed_device.type not in {"cpu", "cuda"}:
        raise ValueError("unsupported device: " + str(device))

    network = NetworkConfig.from_json(network_config_path)
    reconstruction = ReconstructionConfig.from_json(reconstruction_config_path)
    registration = RegistrationConfig.from_dict(reconstruction.registration or {})
    _validate_runtime_config(reconstruction, registration)
    if network.model != "pnnet":
        raise ValueError("full PNMC requires a PN-Net network configuration")

    case = load_case_manifest(case_path, require_physics=True)
    validation = validate_case(case, scan_arrays=True)
    checkpoint = Path(checkpoint).resolve()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (pickle.UnpicklingError, EOFError, RuntimeError, OSError) as exc:
        raise ValueError("cannot load checkpoint: " + str(exc)) from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported network checkpoint format")
    if "model_state_dict" not in payload:
        raise ValueError("checkpoint lacks model weights")
    provenance = payload.get("provenance", {})
    architecture = provenance.get("architecture", {})
    if architecture.get("model") != "pnnet":
        raise ValueError("checkpoint is not a PN-Net model")
    NormalizationSpec.from_dict(provenance.get("normalization", {}))
    expected = {"model": network.model, "channels": list(network.channels),
                "query_chunk": network.query_chunk,
                "checkpoint_chunks": network.checkpoint_chunks,
                "prior_fusion": network.prior_fusion,
                "use_non_local": network.use_non_local}
    observed = {"model": architecture.get("model"),
                "channels": architecture.get("channels"),
                "query_chunk": architecture.get("query_chunk"),
                "checkpoint_chunks": architecture.get("checkpoint_chunks"),
                "prior_fusion": architecture.get("prior_fusion", "add"),
                "use_non_local": architecture.get("use_non_local", True)}
    if observed != expected:
        raise ValueError("checkpoint architecture differs from network configuration")
    train_grids = provenance.get("train_grid_hashes")
    case_grid = case.reconstruction_grid.fingerprint()
    if not isinstance(train_grids, list) or case_grid not in train_grids:
        raise ValueError("checkpoint training grid differs from case grid: "
                         + str(train_grids) + " != " + case_grid)
    model = PNNet2D(tuple(network.channels), network.query_chunk,
                    network.checkpoint_chunks, network.prior_fusion,
                    network.use_non_local)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del model, payload
    dependencies = _check_dependencies()
    return {"status": "passed", "case_id": case.case_id,
            "checkpoint_sha256": _digest_file(checkpoint),
            "architecture": observed, "train_grid_sha256": case_grid,
            "case_grid_sha256": case_grid, "case_manifest_sha256": case.manifest_sha256,
            "projection_sha256": case.projections.sha256,
            "validation": validation, "dependencies": dependencies,
            "reconstruction_config": asdict(reconstruction)}
