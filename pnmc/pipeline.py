"""Signature-checked, resumable physical PNMC reconstruction stages."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class StageStore:
    """Recompute only missing or mismatched stages; completion marker is last."""

    def __init__(self, root: Path, *, resume: bool = False):
        self.root = Path(root).resolve()
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise FileExistsError("output exists; use --resume to verify stages")
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, signature, builder) -> dict:
        if not name.replace("_", "").isalnum():
            raise ValueError("stage name must be alphanumeric")
        key = digest_json(signature)
        folder = (self.root / (name + "_" + key[:16])).resolve()
        if folder.parent != self.root:
            raise ValueError("stage path escapes run directory")
        marker = folder / "complete.json"
        if marker.is_file():
            try:
                prior = json.loads(marker.read_text(encoding="utf-8"))
                if (prior.get("stage") == name and prior.get("input_sha256") == key and
                        prior.get("status") == "passed" and
                        all(Path(item["path"]).resolve().parent == folder and
                            digest_file(item["path"]) == item["sha256"]
                            for item in prior["artifacts"].values())):
                    return {**prior, "folder": str(folder), "reused": True}
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        started = time.perf_counter()
        artifacts = builder(folder)
        if not artifacts:
            raise ValueError("stage must produce a verifiable artifact")
        checked = {}
        for label, path in artifacts.items():
            path = Path(path).resolve()
            if path.parent != folder or not path.is_file():
                raise ValueError("stage artifact must be a file inside its stage folder")
            checked[label] = {"path": str(path), "sha256": digest_file(path)}
        result = {"stage": name, "status": "passed", "input_sha256": key,
                  "artifacts": checked,
                  "elapsed_seconds": time.perf_counter()-started,
                  "artifact_bytes": sum(Path(item["path"]).stat().st_size
                                        for item in checked.values())}
        _write_json(marker, result)
        return {**result, "folder": str(folder), "reused": False}


def _load_fields(path: Path, case) -> tuple:
    from pnmc.motion.qc import quality_check

    raw = json.loads(path.read_text(encoding="utf-8"))
    if (raw.get("status") != "passed" or
            raw.get("direction") != "target_fixed_to_source_moving" or
            raw.get("grid_sha256") != case.reconstruction_grid.fingerprint()):
        raise ValueError("registration status, direction or grid invalid")
    expected = {(target, source) for target in case.phase_ids for source in case.phase_ids}
    masks = {}
    for label in ("m0", "core"):
        item = raw["support"][label]
        if digest_file(item["path"]) != item["sha256"]:
            raise ValueError("registration support hash invalid")
        masks[label] = torch.from_numpy(np.load(item["path"], allow_pickle=False).copy())
        if tuple(masks[label].shape) != case.reconstruction_grid.shape_zyx:
            raise ValueError("registration support grid invalid")
        masks[label] = masks[label].bool()
    pairs = {}
    for item in raw["pairs"]:
        key = (item["target"], item["source"])
        if key not in expected or key in pairs or item["qc"]["status"] != "passed":
            raise ValueError("registration phase pair or QC invalid")
        if (item["grid_sha256"] != case.reconstruction_grid.fingerprint() or
                item["unit"] != "mm" or item["components"] != "xyz" or
                digest_file(item["path"]) != item["sha256"]):
            raise ValueError("registration field provenance invalid")
        field = torch.from_numpy(np.load(item["path"], allow_pickle=False).copy())
        if tuple(field.shape) != (3, *case.reconstruction_grid.shape_zyx):
            raise ValueError("registration field shape invalid")
        quality_check(field, grid=case.reconstruction_grid,
                      m0=masks["m0"], core=masks["core"])
        pairs[key] = field
    if set(pairs) != expected:
        raise ValueError("registration lacks an ordered phase pair")
    return pairs, raw


def fields_for_target(registered_pairs: dict, reconstruction_target: str,
                      observation_phases: list) -> dict:
    """For A[t] T(k→t), sample on t using the registered t→k pull field."""
    return {phase: registered_pairs[(phase, reconstruction_target)]
            for phase in observation_phases}


def run_case(case, *, checkpoint: Path, network_config,
             reconstruction_config, output: Path, resume: bool = False,
             projector=None, enhanced_fixture: Path = None,
             registration_fixture: Path = None) -> dict:
    """Run full stages; injection arguments are for tests, never CLI options."""
    from pnmc.data.manifest import validate_case, vector_values, load_case_manifest
    from pnmc.data.volume_io import load_volume, save_volume
    from pnmc.inference import _load_model, infer_volume
    from pnmc.reconstruction.fdk import build_fdk_inputs
    from pnmc.reconstruction.mc_ossart import reconstruct_target
    from pnmc.reconstruction.projector import TigreProjector
    from pnmc.runtime import environment_report
    from pnmc.motion.stage import (load_enhanced_volumes,
                                   geometric_support, register_case)
    from pnmc.config import RegistrationConfig

    output = Path(output).resolve()
    started = time.perf_counter()
    if network_config.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(network_config.device)
    store = StageStore(output, resume=resume)
    checkpoint = Path(checkpoint).resolve()
    checkpoint_sha = digest_file(checkpoint)
    validation = validate_case(case, scan_arrays=True)
    case_key = digest_json({"manifest": case.manifest_sha256,
                            "projection": case.projections.sha256,
                            "geometry": case.geometry,
                            "grid": case.reconstruction_grid.to_dict(),
                            "angles": vector_values(case.angles_rad),
                            "phases": vector_values(case.phase_id_per_view)})
    validated = store.run("validated", case_key, lambda folder: {
        "report": _write_stage_report(folder / "validation.json", validation)})
    if projector is None:
        projector = TigreProjector(case.geometry, case.reconstruction_grid,
                                   np.asarray(vector_values(case.angles_rad)),
                                   device=network_config.device,
                                   fdk_padding_vu=(reconstruction_config.fdk_padding,)*2)
    config_sha = digest_json(asdict(reconstruction_config))
    if enhanced_fixture is None:
        fdk = store.run("fdk", {"case": case_key, "fdk":
                         [reconstruction_config.fdk_mode,
                          reconstruction_config.fdk_filter,
                          reconstruction_config.fdk_padding]},
                        lambda folder: {"case": _fdk_stage(case, projector, folder)})
        fdk_case = load_case_manifest(fdk["artifacts"]["case"]["path"],
                                      require_physics=True)

        def enhance(folder):
            model, normalization, provenance = _load_model(checkpoint,
                                                             network_config.device)
            architecture = provenance.get("architecture", {})
            if (architecture.get("model") != network_config.model or
                    list(architecture.get("channels", [])) != list(network_config.channels) or
                    architecture.get("prior_fusion") != network_config.prior_fusion or
                    architecture.get("use_non_local") != network_config.use_non_local):
                raise ValueError("checkpoint architecture differs from network configuration")
            if case.reconstruction_grid.fingerprint() not in provenance["train_grid_hashes"]:
                raise ValueError("checkpoint training grid differs from physical case")
            prior = load_volume(fdk_case.arrays["prior"], case.reconstruction_grid)
            volumes = []
            for phase in case.phase_ids:
                sparse = load_volume(fdk_case.arrays["sparse_by_phase"][phase],
                                     case.reconstruction_grid)
                predicted = infer_volume(model, sparse, prior,
                                         grid=case.reconstruction_grid,
                                         normalization=normalization,
                                         device=network_config.device,
                                         batch_size=network_config.batch_size)
                ref = save_volume(folder / ("enhanced_" + phase + ".npy"),
                                  predicted, case.reconstruction_grid)
                volumes.append({"subject_id": case.subject_id,
                                "case_id": case.case_id,
                                "acquisition_id": case.acquisition_id,
                                "phase_id": phase,
                                "grid_sha256": case.reconstruction_grid.fingerprint(),
                                "prediction_mu": ref.to_dict()})
            return {"report": _write_stage_report(folder / "report.json", {
                "status": "passed", "checkpoint_sha256": checkpoint_sha,
                "normalization": asdict(normalization), "volumes": volumes,
                "reference_evaluation": "not_run"})}

        enhanced = store.run("enhanced", {"fdk": fdk["input_sha256"],
                          "checkpoint": checkpoint_sha,
                          "network_config": asdict(network_config)}, enhance)
    else:
        fixture_sha = digest_file(enhanced_fixture)
        fdk = store.run("fdk", {"fixture": fixture_sha, "case": case_key},
                        lambda folder: {"report": _write_stage_report(
                            folder / "fixture.json", {"status": "fixture_bypass"})})
        enhanced = store.run("enhanced", {"fixture": fixture_sha,
                                       "case": case_key},
                             lambda folder: {"report": _copy_stage_file(
                                 enhanced_fixture, folder / "report.json")})
    enhanced_report = Path(enhanced["artifacts"]["report"]["path"])
    volumes, volume_hashes, enhanced_data = load_enhanced_volumes(enhanced_report, case)
    registration_config = RegistrationConfig.from_dict(
        reconstruction_config.registration or {})
    if registration_fixture is None:
        registered = store.run("registered", {
            "enhanced": enhanced["input_sha256"], "volumes": volume_hashes,
            "case": case_key, "registration": asdict(registration_config)},
            lambda folder: {"report": _registration_stage(
                case, enhanced_report, registration_config, folder, projector)})
    else:
        registered = store.run("registered", {
            "fixture": digest_file(registration_fixture),
            "enhanced": enhanced["input_sha256"], "case": case_key},
            lambda folder: {"report": _copy_stage_file(
                registration_fixture, folder / "registration.json")})
    fields, registration = _load_fields(
        Path(registered["artifacts"]["report"]["path"]), case)
    support = geometric_support(case)
    support_sha = digest_json({"shape": list(support.shape),
                               "packed": hashlib.sha256(support.numpy().tobytes()).hexdigest()})
    targets = {}
    for target in case.phase_ids:
        target_fields = fields_for_target(fields, target, case.phase_ids)
        field_hashes = {observation: next(
            item["sha256"] for item in registration["pairs"]
            if item["target"] == observation and item["source"] == target)
            for observation in case.phase_ids}
        key = {"case": case_key, "target": target, "enhanced": volume_hashes[target],
               "fields": field_hashes, "support": support_sha, "mc": config_sha}

        def reconstruct(folder, target=target, target_fields=target_fields):
            reconstruct_target(case, target, target_fields,
                               reconstruction_config, projector=projector,
                               initial_mu=volumes[target], support_mask=support,
                               output=folder)
            return {"report": folder / "mc_report.json",
                    "final": folder / "final.npy"}

        targets[target] = store.run("mc_" + str(case.phase_ids.index(target)),
                                    key, reconstruct)
    reconstructed = store.run("reconstructed", {
        "targets": {phase: stage["input_sha256"] for phase, stage in targets.items()}},
        lambda folder: {"report": _write_stage_report(folder / "targets.json", {
            "status": "passed", "targets": {phase: stage["artifacts"]
                                             for phase, stage in targets.items()}})})
    try:
        import psutil
        memory_info = psutil.Process().memory_info()
        cpu_peak = getattr(memory_info, "peak_wset", None)
        cpu_rss = memory_info.rss
    except ImportError:
        cpu_peak, cpu_rss = None, None
    result = {"status": "passed", "case_id": case.case_id,
              "reference_evaluation": "not_run",
              "normalization": enhanced_data.get("normalization"), "stages": {
                  "validated": validated, "fdk": fdk, "enhanced": enhanced,
                  "registered": registered, "reconstructed": reconstructed,
                  "evaluated": {"status": "not_run"}},
              "targets": {phase: stage["artifacts"] for phase, stage in targets.items()},
              "cost": {"elapsed_seconds": time.perf_counter()-started,
                       "cpu_peak_bytes": cpu_peak, "cpu_rss_end_bytes": cpu_rss,
                       "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                           network_config.device) if network_config.device.startswith("cuda")
                       and torch.cuda.is_available() else None,
                       "disk_bytes": sum(path.stat().st_size for path in output.rglob("*")
                                         if path.is_file())},
              "provenance": {"case_manifest_sha256": case.manifest_sha256,
                             "projection_sha256": case.projections.sha256,
                             "checkpoint_sha256": checkpoint_sha,
                             "network_config_sha256": digest_json(asdict(network_config)),
                             "reconstruction_config_sha256": config_sha,
                             "environment": environment_report()}}
    _write_json(output / "report.json", result)
    return result


def _write_stage_report(path: Path, data: dict) -> Path:
    _write_json(path, data)
    return path


def _copy_stage_file(source: Path, target: Path) -> Path:
    shutil.copy2(source, target)
    return target


def _fdk_stage(case, projector, folder: Path) -> Path:
    from pnmc.reconstruction.fdk import build_fdk_inputs
    build_fdk_inputs(case, projector, folder)
    return folder / "case_fdk.json"


def _registration_stage(case, enhanced_report: Path, config,
                        folder: Path, projector) -> Path:
    from pnmc.motion.stage import register_case
    register_case(case, enhanced_report, config, folder, projector=projector)
    return folder / "registration.json"
