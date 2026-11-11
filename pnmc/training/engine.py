"""Shared train/validation loop with epoch-boundary reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pnmc.config import NetworkConfig
from pnmc.data.datasets import SliceDataset
from pnmc.data.normalization import fit_normalization
from pnmc.data.schema import NetworkManifest
from pnmc.losses.ssim import ssim_loss
from pnmc.models.baseline import PriorNetMatched
from pnmc.models.pnnet import PNNet2D
from pnmc.runtime import create_run

from .checkpoints import restore_checkpoint, save_checkpoint


def _move(batch, key, device):
    return batch[key].to(device=device, non_blocking=True)


def _supervised_loss(prediction, label, mask, loss_name="ssim"):
    if loss_name == "ssim":
        return ssim_loss(prediction, label, mask)
    if loss_name == "mse":
        if prediction.shape != label.shape or mask.shape != label.shape or not mask.any():
            raise ValueError("MSE loss needs matching images and nonempty valid mask")
        return (prediction-label).square()[mask.bool()].mean()
    raise ValueError("unknown loss_name: " + str(loss_name))


def _valid_window_samples(mask):
    if mask.ndim != 4 or mask.shape[-2] < 11 or mask.shape[-1] < 11:
        return torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
    full_windows = F.avg_pool2d(mask.float(), 11, stride=1) >= 1 - 1e-6
    return full_windows.flatten(1).any(dim=1)


def train_epoch(model, loader, optimizer, *, device, loss_name="ssim") -> dict:
    model.train()
    total = 0.0
    batches = 0
    samples = 0
    skipped_samples = 0
    for batch in loader:
        sparse = _move(batch, "input", device)
        prior = _move(batch, "prior", device)
        label = _move(batch, "label", device)
        mask = _move(batch, "valid_mask", device)
        valid_samples = _valid_window_samples(mask)
        skipped_samples += int((~valid_samples).sum())
        if not bool(valid_samples.any()):
            continue
        sparse, prior, label, mask = (value[valid_samples]
                                      for value in (sparse, prior, label, mask))
        optimizer.zero_grad(set_to_none=True)
        loss = _supervised_loss(model(sparse, prior), label, mask, loss_name)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * sparse.shape[0]
        samples += sparse.shape[0]
        batches += 1
    if samples == 0:
        raise ValueError("empty training loader")
    return {"loss": total / samples, "batches": batches,
            "samples": samples, "skipped_samples": skipped_samples}


@torch.no_grad()
def validate_epoch(model, loader, *, device, normalization, loss_name="ssim") -> dict:
    model.eval()
    by_subject = defaultdict(list)
    losses = []
    skipped_samples = 0
    for batch in loader:
        sparse = _move(batch, "input", device)
        prior = _move(batch, "prior", device)
        label = _move(batch, "label", device)
        mask = _move(batch, "valid_mask", device)
        valid_samples = _valid_window_samples(mask)
        skipped_samples += int((~valid_samples).sum())
        if not bool(valid_samples.any()):
            continue
        sparse, prior, label, mask = (value[valid_samples]
                                      for value in (sparse, prior, label, mask))
        subjects = [subject for subject, keep in zip(batch["subject_id"],
                                                      valid_samples.tolist()) if keep]
        prediction = model(sparse, prior)
        for index, subject in enumerate(subjects):
            loss = _supervised_loss(prediction[index:index + 1], label[index:index + 1],
                                    mask[index:index + 1], loss_name)
            similarity = 1 - float(ssim_loss(prediction[index:index + 1],
                                             label[index:index + 1],
                                             mask[index:index + 1]))
            by_subject[subject].append(similarity)
            losses.append(float(loss))
    if not by_subject:
        raise ValueError("empty validation loader")
    subject_ssim = {subject: sum(scores) / len(scores)
                    for subject, scores in by_subject.items()}
    return {"loss": sum(losses) / len(losses), "subject_ssim": subject_ssim,
            "mean_subject_ssim": sum(subject_ssim.values()) / len(subject_ssim),
            "skipped_samples": skipped_samples}


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _write_json_atomic(path: Path, value: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp",
                                        dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_ready(value):
    """Use the same list representation before and after JSON persistence."""
    return json.loads(json.dumps(value, allow_nan=False))


def _provenance(config: NetworkConfig, manifest: NetworkManifest,
                output: Path, normalization) -> dict:
    source_hashes = sorted({ref.sha256 for record in manifest.records
                            for ref in record.arrays.values()})
    train_subjects = set(manifest.split_subjects["train"])
    train_grid_hashes = sorted({manifest.grids[record.grid_id].fingerprint()
                                for record in manifest.records
                                if record.subject_id in train_subjects})
    if len(train_grid_hashes) != 1:
        raise ValueError("training requires exactly one physical grid")
    return _json_ready({"architecture": {"model": config.model, "channels": tuple(config.channels),
                             "query_chunk": config.query_chunk,
                             "checkpoint_chunks": config.checkpoint_chunks,
                             "prior_fusion": config.prior_fusion,
                             "use_non_local": config.use_non_local},
            "normalization": asdict(normalization),
            "split": manifest.split_subjects,
            "grid": {key: value.fingerprint() for key, value in manifest.grids.items()},
            "train_grid_hashes": train_grid_hashes,
            "raw": source_hashes, "manifest_sha256": manifest.manifest_sha256,
            "config": asdict(config), "run_path": str(output)})


def run_training(config: NetworkConfig, manifest: NetworkManifest,
                 output: Path, *, resume: Path = None) -> dict:
    if config.model == "priornet_matched" and (
            config.prior_fusion != "add" or not config.use_non_local):
        raise ValueError("PN-Net architecture ablations require model=pnnet")
    output = Path(output).resolve()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training device is unavailable")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    normalization = fit_normalization(manifest)
    provenance = _provenance(config, manifest, output, normalization)
    config_data = _json_ready(asdict(config))
    config_bytes = json.dumps(config_data, sort_keys=True).encode("utf-8")
    run_provenance = {"config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                      "input_sha256": manifest.manifest_sha256,
                      "training_provenance": provenance}
    create_run(output.parent, output.name, run_provenance)
    snapshot = output / "config.json"
    if snapshot.exists():
        existing = json.loads(snapshot.read_text(encoding="utf-8"))
        if existing != config_data:
            raise ValueError("run configuration snapshot differs")
    else:
        _write_json_atomic(snapshot, config_data)
    train_data = SliceDataset(manifest, split="train", normalization=normalization,
                              require_label=True)
    valid_data = SliceDataset(manifest, split="valid", normalization=normalization,
                              require_label=True)
    if not train_data or not valid_data:
        raise ValueError("training and validation splits must both have labeled slices")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, generator=generator,
                              worker_init_fn=_seed_worker if config.num_workers else None,
                              pin_memory=config.device.startswith("cuda"))
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers,
                              worker_init_fn=_seed_worker if config.num_workers else None,
                              pin_memory=config.device.startswith("cuda"))
    model = (PNNet2D(tuple(config.channels), config.query_chunk, config.checkpoint_chunks,
                     config.prior_fusion, config.use_non_local)
             if config.model == "pnnet" else PriorNetMatched(normalization))
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=list(config.milestones), gamma=.1)
    start_epoch = 0
    history = []
    best_epoch = 0
    best_score = -float("inf")
    if resume is not None:
        payload = restore_checkpoint(resume, model=model, optimizer=optimizer,
                                     scheduler=scheduler, expected_provenance=provenance)
        start_epoch = payload["completed_epoch"]
        state = payload["training_state"]
        history = list(state["history"])
        best_epoch = state["best_epoch"]
        best_score = state["best_subject_ssim"]
        generator.set_state(state["loader_generator_state"])
    elif (output / "latest.pt").exists():
        raise FileExistsError("existing checkpoint requires --resume")
    for epoch in range(start_epoch + 1, config.epochs + 1):
        lr_used = optimizer.param_groups[0]["lr"]
        training_options = {} if config.loss_name == "ssim" else {"loss_name": config.loss_name}
        trained = train_epoch(model, train_loader, optimizer, device=config.device,
                              **training_options)
        validated = validate_epoch(model, valid_loader, device=config.device,
                                   normalization=normalization, **training_options)
        scheduler.step()
        score = validated["mean_subject_ssim"]
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
        history.append({"epoch": epoch, "train_loss": trained["loss"],
                        "valid_loss": validated["loss"], "mean_subject_ssim": score,
                        "subject_ssim": validated["subject_ssim"], "lr_used": lr_used,
                        "train_samples": trained.get("samples"),
                        "train_skipped_samples": trained.get("skipped_samples", 0),
                        "valid_skipped_samples": validated.get("skipped_samples", 0)})
        state = {"best_epoch": best_epoch, "best_subject_ssim": best_score,
                 "history": history, "loader_generator_state": generator.get_state(),
                 "worker_seed_policy": "torch_initial_seed_mod_2^32"}
        fields = dict(model=model, optimizer=optimizer, scheduler=scheduler,
                      completed_epoch=epoch, provenance=provenance, training_state=state)
        save_checkpoint(output / "epoch_{:04d}.pt".format(epoch), **fields)
        save_checkpoint(output / "latest.pt", **fields)
        if improved:
            save_checkpoint(output / "best.pt", **fields)
    report = {"status": "passed", "epochs_completed": config.epochs,
              "best_epoch": best_epoch, "best_subject_ssim": best_score,
              "history": history, "normalization": asdict(normalization),
              "output": str(output), "artifacts": {"latest": str(output / "latest.pt"),
                                               "best": str(output / "best.pt")}}
    run_report_path = output / "run.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    run_report.update({"status": "passed", "artifacts": report["artifacts"],
                       "best_epoch": best_epoch, "best_subject_ssim": best_score})
    _write_json_atomic(run_report_path, run_report)
    _write_json_atomic(output / "report.json", report)
    return report
