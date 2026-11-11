"""Reference-free phase-volume inference with strict grid and slice identity."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pnmc.data.manifest import _open_array
from pnmc.data.normalization import denormalize, normalize
from pnmc.data.schema import GridSpec, NetworkManifest, NormalizationSpec
from pnmc.data.volume_io import save_volume
from pnmc.models.baseline import PriorNetMatched
from pnmc.models.pnnet import PNNet2D
from pnmc.runtime import create_run


def infer_volume(model, sparse_mu: torch.Tensor, prior_mu: torch.Tensor, *,
                 grid: GridSpec, normalization: NormalizationSpec, device,
                 batch_size: int) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tuple(sparse_mu.shape) != grid.shape_zyx or sparse_mu.shape != prior_mu.shape:
        raise ValueError("input volume and grid shapes disagree")
    if not torch.isfinite(sparse_mu).all() or not torch.isfinite(prior_mu).all():
        raise ValueError("input volumes must be finite")
    model = model.to(device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for first in range(0, grid.shape_zyx[0], batch_size):
            sparse = sparse_mu[first:first + batch_size, None].to(device)
            prior = prior_mu[first:first + batch_size, None].to(device)
            output = model(normalize(sparse, normalization), normalize(prior, normalization))
            if output.shape != sparse.shape or not torch.isfinite(output).all():
                raise ValueError("network produced invalid slice shape or values")
            predictions.append(denormalize(output, normalization)[:, 0].cpu())
    return torch.cat(predictions, dim=0)


def _load_model(checkpoint: Path, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or "model_state_dict" not in payload:
        raise ValueError("unsupported or incomplete network checkpoint")
    provenance = payload.get("provenance", {})
    architecture = provenance.get("architecture", {})
    normalization_data = provenance.get("normalization")
    if not normalization_data or not provenance.get("train_grid_hashes"):
        raise ValueError("checkpoint lacks normalization or training grid identity")
    normalization = NormalizationSpec.from_dict(normalization_data)
    if architecture.get("model") == "pnnet":
        model = PNNet2D(tuple(architecture["channels"]),
                        architecture["query_chunk"], architecture["checkpoint_chunks"],
                        architecture.get("prior_fusion", "add"),
                        architecture.get("use_non_local", True))
    elif architecture.get("model") == "priornet_matched":
        if architecture.get("prior_fusion", "add") != "add" or not architecture.get("use_non_local", True):
            raise ValueError("PriorNet matched checkpoint has PN-Net variant flags")
        model = PriorNetMatched(normalization)
    else:
        raise ValueError("unknown checkpoint model architecture")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, normalization, provenance


def _physical_slice(ref, expected_shape, verified_paths):
    if ref.axes != "y,x" or ref.unit != "1/mm":
        raise ValueError("inference slice axes/unit must be y,x and 1/mm")
    if ref.path not in verified_paths:
        if hashlib.sha256(ref.path.read_bytes()).hexdigest() != ref.sha256:
            raise ValueError("inference input sha256 mismatch")
        verified_paths.add(ref.path)
    values = np.asarray(_open_array(ref), dtype=np.float32)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError("inference slice grid/shape or finite values disagree")
    return torch.from_numpy(values.copy())


def infer_case(manifest: NetworkManifest, checkpoint: Path, output: Path, *,
               split: str, device) -> dict:
    if split not in manifest.split_subjects:
        raise ValueError("unknown inference split: " + split)
    subjects = set(manifest.split_subjects[split])
    records = [record for record in manifest.records if record.subject_id in subjects]
    if not records:
        raise ValueError("inference split contains no slices")
    model, normalization, provenance = _load_model(checkpoint, device)
    groups = defaultdict(list)
    for record in records:
        groups[(record.subject_id, record.case_id, record.acquisition_id,
                record.phase_id)].append(record)
    approved_grids = set(provenance["train_grid_hashes"])
    for key, group in groups.items():
        grid_ids = {record.grid_id for record in group}
        if len(grid_ids) != 1:
            raise ValueError("mixed grid ids within one phase volume")
        grid = manifest.grids[next(iter(grid_ids))]
        if grid.fingerprint() not in approved_grids:
            raise ValueError("checkpoint training grid differs from inference grid")
        indices = [record.slice_index for record in group]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate slice index within phase volume")
        if sorted(indices) != list(range(grid.shape_zyx[0])):
            raise ValueError("missing slice index within phase volume")
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    create_run(output.parent, output.name,
               {"config_sha256": hashlib.sha256(json.dumps(provenance["architecture"],
                                                  sort_keys=True).encode()).hexdigest(),
                "input_sha256": manifest.manifest_sha256,
                "checkpoint_sha256": checkpoint_sha, "split": split})
    verified_paths = set()
    volumes = []
    for number, (identity, group) in enumerate(sorted(groups.items())):
        grid_id = group[0].grid_id
        grid = manifest.grids[grid_id]
        sparse = torch.empty(grid.shape_zyx, dtype=torch.float32)
        prior = torch.empty_like(sparse)
        for record in group:
            shape = grid.shape_zyx[1:]
            sparse[record.slice_index] = _physical_slice(record.arrays["input"], shape,
                                                          verified_paths)
            prior[record.slice_index] = _physical_slice(record.arrays["prior"], shape,
                                                         verified_paths)
        predicted = infer_volume(model, sparse, prior, grid=grid,
                                 normalization=normalization, device=device, batch_size=2)
        folder = output / "volume_{:04d}".format(number)
        folder.mkdir(exist_ok=True)
        physical_ref = save_volume(folder / "prediction_mu.nii.gz", predicted, grid)
        network_values = (2 * (predicted - normalization.a) /
                          (normalization.b - normalization.a) - 1).numpy()
        normalized_path = folder / "prediction_network.npy"
        np.save(normalized_path, network_values, allow_pickle=False)
        volumes.append({"subject_id": identity[0], "case_id": identity[1],
                        "acquisition_id": identity[2], "phase_id": identity[3],
                        "grid_id": grid_id, "grid_sha256": grid.fingerprint(),
                        "prediction_mu": physical_ref.to_dict(),
                        "prediction_network": {"path": str(normalized_path),
                                               "sha256": hashlib.sha256(
                                                   normalized_path.read_bytes()).hexdigest()}})
    report = {"status": "passed", "split": split,
              "manifest_sha256": manifest.manifest_sha256,
              "checkpoint_sha256": checkpoint_sha,
              "normalization": asdict(normalization), "volumes": volumes,
              "reference_evaluation": "not_run"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    run_report_path = output / "run.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    run_report["status"] = "passed"
    run_report["artifacts"] = {"report": str(output / "report.json")}
    run_report_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    return report
