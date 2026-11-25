from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.nn.functional import grid_sample

from .data import baseline_artifacts, case_rows, geometry_support
from .fusion import field_quality, ordered_pairs, pull_grid
from .registration import RegisNet3D, estimate_field, field_smoothness, local_ncc
from ..training.checkpoint import atomic_save, load_checkpoint
from ..training.trainer import select_device
from ..utils.io import atomic_json, canonical_hash, sha256
from ..utils.seed import restore_rng, rng_state, seed_all


def contract(config, row, provider):
    reg = config["registration"]
    meta = row["metadata"]
    return {
        "implementation": "RegisNet3D/dpi-moco-b709c28",
        "registration_code_sha256": sha256(Path(__file__).with_name("registration.py")),
        "input": reg["input"], "train_pairs": reg["train_pairs"],
        "normalization": reg["normalization"], "max_displacement_vox": reg["max_displacement_vox"],
        "intensity_domain": meta["intensity_domain"], "source_normalization": meta["normalization"],
        "calibration_application": meta["calibration_application"], "baseline_provider": provider,
    }


def load_registration(checkpoint, config, rows, provider):
    if not checkpoint:
        raise ValueError("registration.mode=learned requires --register-checkpoint for fused outputs")
    saved = load_checkpoint(checkpoint, require_training=False)
    if saved.get("stage") != "registration" or saved.get("checkpoint_kind") not in ("registration_training", "registration_weights"):
        raise ValueError("Expected a RegisNet3D registration checkpoint from train-register")
    actual = saved.get("provider_contract", {})
    expected = contract(config, rows[0], provider)
    differing = [key for key, value in expected.items() if actual.get(key) != value]
    if differing:
        raise ValueError("RegisNet checkpoint conflict at " + ", ".join(differing))
    return saved


def normalize(tensor, settings):
    return (tensor + settings["offset"]) / settings["scale"]


def pairs(rows, mode):
    result = []
    for row in rows:
        labels = row["metadata"]["phase_labels"]
        for target, sources in ordered_pairs(labels, row["expected_phase_labels"], mode).items():
            for source in sources:
                result.append((row, labels.index(source), labels.index(target)))
    if not result:
        raise ValueError("No ordered within-scan phase pairs available")
    return result


def sample(row, baseline, source, target, reg, device, rng):
    shape = row["shape"][1:]
    patch = reg["patch"] or shape
    width = [min(n, p) for n, p in zip(shape, patch)]
    with ExitStack() as stack:
        source_file = stack.enter_context(h5py.File(row["path"], "r"))
        if baseline:
            image = stack.enter_context(h5py.File(baseline["path"], "r"))["prediction"]
        else:
            image = source_file["cbct"]
        read_support, _ = geometry_support(source_file, row)
        geometry = read_support(target)
        # Select a common crop with valid fixed centers. The first validation crop is deterministic.
        for attempt in range(33):
            if attempt == 32:
                indices = np.argwhere(geometry)
                if not len(indices):
                    raise ValueError(f'{row["case_id"]}: empty target geometry support')
                origin = [int(np.clip(c - w // 2, 0, n - w))
                          for c, w, n in zip(indices[len(indices) // 2], width, shape)]
            elif rng is None and attempt == 0:
                origin = [(n - w) // 2 for n, w in zip(shape, width)]
            else:
                generator = rng if rng is not None else np.random.default_rng(attempt)
                origin = [int(generator.integers(n - w + 1)) for n, w in zip(shape, width)]
            core = tuple(slice(start, start + w) for start, w in zip(origin, width))
            if geometry[core].any():
                break
        lower = [max(0, a - h) for a, h in zip(origin, reg["context"])]
        upper = [min(n, a + w + h) for n, a, w, h in zip(shape, origin, width, reg["context"])]
        crop = tuple(slice(a, b) for a, b in zip(lower, upper))
        inner = tuple(slice(a - low, a - low + w) for a, low, w in zip(origin, lower, width))
        mask = np.zeros(tuple(b - a for a, b in zip(lower, upper)), dtype=bool)
        mask[inner] = geometry[core]
        moving, fixed = [np.asarray(image[(i,) + crop], dtype=np.float32) for i in (source, target)]
    if not (np.isfinite(moving).all() and np.isfinite(fixed).all()):
        raise ValueError(f'{row["case_id"]}: nonfinite registration image')
    tensors = [torch.from_numpy(a).to(device)[None, None] for a in (moving, fixed, mask)]
    moving, fixed, centers = tensors
    return normalize(moving, reg["normalization"]), normalize(fixed, reg["normalization"]), centers, inner


def pair_loss(model, data, reg):
    moving, fixed, mask, inner = data
    field = estimate_field(model, moving, fixed, reg["max_displacement_vox"])
    grid, _ = pull_grid(field)
    # Preserve the reference training loss at the physical boundary. In inference
    # the geometric-support sampler rejects coordinates outside the source grid.
    moved = grid_sample(moving, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ncc = local_ncc(moved, fixed, reg["ncc_window"], mask)
    smooth = field_smoothness(field[(Ellipsis,) + inner])
    return ncc + reg["smooth_weight"] * smooth, field


def train_registration(config, manifest, val_manifest, out, *, baseline=None, val_baseline=None,
                       baseline_checkpoint=None, resume=None, stop_after_epoch=None):
    reg = config["registration"]
    out = Path(out).resolve()
    if reg["mode"] == "zero":
        result = {"mode": "zero", "trained": False, "forward_calls": 0}
        atomic_json(out / "summary.json", result)
        return result
    if not manifest or not val_manifest:
        raise ValueError("train-register requires --manifest and --val-manifest")
    train_rows, val_rows = case_rows(manifest), case_rows(val_manifest)
    if {r["case_id"] for r in train_rows} & {r["case_id"] for r in val_rows}:
        raise ValueError("Registration train/validation scan overlap")
    if {r["sha256"] for r in train_rows} & {r["sha256"] for r in val_rows}:
        raise ValueError("Registration train/validation paired-file hash overlap")
    for rows, split_name in [(train_rows, "train"), (val_rows, "val")]:
        if any(row.get("scan_split_name", split_name) != split_name for row in rows):
            raise ValueError(f"Unexpected scan_split_name in {split_name} manifest")
    b_train, b_val, provider = {}, {}, None
    if reg["input"] == "b":
        b_train, provider = baseline_artifacts(config, train_rows, manifest, out / "baseline_cache",
                                              baseline, baseline_checkpoint)
        b_val, validation_provider = baseline_artifacts(
            config, val_rows, val_manifest, out / "baseline_cache", val_baseline or baseline, baseline_checkpoint)
        if provider != validation_provider:
            raise ValueError("Training and validation have different b providers")
    inputs = contract(config, train_rows[0], provider)
    if inputs != contract(config, val_rows[0], provider):
        raise ValueError("Training and validation intensity or normalization contracts differ")
    training = pairs(train_rows, reg["train_pairs"])
    validation = pairs(val_rows, reg["train_pairs"])
    snapshot = {}
    for name, rows, artifacts, path in (("train", train_rows, b_train, manifest),
                                        ("val", val_rows, b_val, val_manifest)):
        snapshot[name] = {"manifest": str(Path(path).resolve()), "manifest_sha256": sha256(path),
                          "cases": [{k: row[k] for k in ("case_id", "animal_id", "sha256",
                                                        "source_metadata_hash", "expected_phase_labels")}
                                    | {"b_sha256": artifacts.get(row["case_id"], {}).get("sha256")}
                                    for row in rows]}
    effective_train = {k: v for k, v in reg.items() if not k.startswith("inference_")}
    signature = canonical_hash({"training": effective_train, "seed": config["seed"], "device": config["device"],
                                "inputs": inputs, "data": snapshot, "training_code": sha256(__file__)})
    device = select_device(config["device"])
    seed_all(config["seed"], threads=config["threads"])
    model = RegisNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=reg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, reg["epochs"], eta_min=reg["min_lr"])
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = checkpoint_dir / "last.pt"
    best = checkpoint_dir / "best.pt"
    start, best_loss, history = 1, float("inf"), []
    if resume:
        saved = load_registration(resume, config, train_rows, provider)
        if saved.get("signature") != signature or saved.get("checkpoint_kind") != "registration_training":
            raise ValueError("Resume checkpoint conflicts with training config, inputs or data snapshot")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        start, best_loss, history = saved["completed_epoch"] + 1, saved["best_loss"], saved["history"]
        if last.exists() and load_checkpoint(last, require_training=False)["completed_epoch"] >= start:
            raise ValueError("Resume would overwrite a newer epoch in the run directory")
        best_on_disk = load_checkpoint(best, require_training=False) if best.exists() else {}
        if best_on_disk.get("epoch") != saved["best_epoch"] or best_on_disk.get("signature") != signature:
            atomic_save(best, {"schema_version": 1, "checkpoint_kind": "registration_weights",
                              "stage": "registration", "state_dict": saved["best_state_dict"],
                              "provider_contract": inputs, "signature": signature,
                              "epoch": saved["best_epoch"], "data_snapshot": snapshot})
    elif last.exists() or best.exists():
        raise FileExistsError("Training run contains registration checkpoints; use --resume")
    end = reg["epochs"] if stop_after_epoch is None else stop_after_epoch
    if not start <= end <= reg["epochs"]:
        raise ValueError("--stop-after-epoch must be at least the next epoch and at most registration.epochs")
    best_state = None if not resume else saved.get("best_state_dict")
    best_epoch = None if not resume else saved.get("best_epoch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, end + 1):
        tick = time.perf_counter()
        generator = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch]))
        steps = reg["steps_per_epoch"] or len(training)
        indices = np.concatenate([generator.permutation(len(training))
                                  for _ in range((steps + len(training) - 1) // len(training))])[:steps]
        model.train()
        train_loss = []
        for index in indices:
            row, source, target = training[index]
            sample_data = sample(row, b_train.get(row["case_id"]), source, target, reg, device, generator)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = pair_loss(model, sample_data, reg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite registration training loss")
            loss.backward()
            optimizer.step()
            train_loss.append(float(loss.detach()))
        model.eval()
        val_qc = []
        values = []
        with torch.no_grad():
            for row, source, target in validation:
                data = sample(row, b_val.get(row["case_id"]), source, target, reg, device, None)
                loss, field = pair_loss(model, data, reg)
                values.append(float(loss))
                labels = row["metadata"]["phase_labels"]
                val_qc.append({"case_id": row["case_id"], "source": labels[source], "target": labels[target],
                               **field_quality(field[0].cpu().numpy(), data[2][0, 0].cpu().numpy(),
                                               row["metadata"]["geometry"]["voxel_spacing_zyx_mm"])})
        val_loss = float(np.mean(values))
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite registration validation loss")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_loss)), "val_loss": val_loss,
                        "steps": steps, "learning_rate": optimizer.param_groups[0]["lr"],
                        "seconds": time.perf_counter() - tick, "validation_qc": val_qc,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        scheduler.step()
        payload = {"schema_version": 1, "stage": "registration", "state_dict": model.state_dict(),
                   "provider_contract": inputs, "signature": signature, "data_snapshot": snapshot,
                   "epoch": epoch}
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        atomic_save(last, {**payload, "checkpoint_kind": "registration_training", "config": config,
                           "completed_epoch": epoch, "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "rng": rng_state(), "history": history,
                           "best_loss": best_loss, "best_epoch": best_epoch, "best_state_dict": best_state})
        if improved:
            atomic_save(best, {**payload, "checkpoint_kind": "registration_weights"})
        result = {"completed_epoch": epoch, "best_epoch": best_epoch, "best_loss": best_loss,
                  "signature": signature, "history": history, "checkpoint": str(best),
                  "checkpoint_sha256": sha256(best),
                  "repeated_animals_train_val": sorted({str(r["animal_id"]) for r in train_rows}
                                                       & {str(r["animal_id"]) for r in val_rows})}
        atomic_json(out / "summary.json", result)
    return result
