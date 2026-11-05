"""Two-stage PriorNet training with optional heart-mask finetuning."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from PriorNet_pytorch import Dense2D
from utils.heart_mask_loss import (
    _boundary_band,
    _per_sample_mean,
    _sobel_difference,
    _soft_region,
    heart_reconstruction_loss,
)
from utils.read_2d_npz_wy import npz_reader_2d_data_with_prior
from utils.sophia import SophiaG


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_indices(length: int, stage: str, *, seed: int, epoch: int) -> list[int]:
    if length < 1:
        raise ValueError("training list is empty")
    if stage == "finetune":
        return list(range(length))
    if stage != "base":
        raise ValueError(f"unknown stage: {stage}")
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator)[:max(1, length // 10)].tolist()


def _prediction(model, batch, device):
    image = batch["input"].to(device)
    prior = batch["prior"].to(device)
    label = batch["label"].to(device)
    return model(image, prior)[-1], label


def _masked_batch(batch, device):
    if "mask" not in batch or "mask_quality" not in batch:
        raise ValueError("finetune stage requires mask and mask_quality for every sample")
    return batch["mask"].to(device), batch["mask_quality"].to(device)


def train_epoch(model, loader, optimizer, *, stage, loss_config, device) -> dict[str, float]:
    if stage not in {"base", "finetune"}:
        raise ValueError(f"unknown stage: {stage}")
    model.train()
    totals = {key: 0.0 for key in (
        "loss_total", "loss_full", "loss_roi_exact", "loss_roi_rough",
        "loss_boundary", "loss_gradient", "loss_weighted_roi_exact",
        "loss_weighted_roi_rough", "loss_weighted_boundary", "loss_weighted_gradient",
    )}
    count = exact_count = rough_count = unknown_count = 0
    for batch in loader:
        pred, label = _prediction(model, batch, device)
        if stage == "base":
            full = F.mse_loss(pred, label)
            pieces = {"total": full, "full": full}
        else:
            mask, quality = _masked_batch(batch, device)
            pieces = heart_reconstruction_loss(pred, label, mask, quality, **loss_config)
            exact_count += int((quality == 1).sum().item())
            rough_count += int((quality == 0).sum().item())
            unknown_count += int((quality == -1).sum().item())
        optimizer.zero_grad()
        pieces["total"].backward()
        n = pred.shape[0]
        if isinstance(optimizer, SophiaG):
            optimizer.update_hessian()
            optimizer.step(bs=n)
        else:
            optimizer.step()
        count += n
        for key in totals:
            part = key.removeprefix("loss_")
            if part in pieces:
                totals[key] += pieces[part].detach().item() * n
        if stage == "finetune":
            for part, coefficient in (
                ("roi_exact", "lambda_exact"),
                ("roi_rough", "lambda_rough"),
                ("boundary", "lambda_boundary"),
                ("gradient", "lambda_gradient"),
            ):
                totals[f"loss_weighted_{part}"] += (
                    pieces[part].detach().item() * loss_config[coefficient] * n
                )
    if count == 0:
        raise ValueError("training loader is empty")
    return {key: value / count for key, value in totals.items()} | {
        "num_samples": count,
        "exact_count": exact_count,
        "rough_count": rough_count,
        "unknown_count": unknown_count,
    }


def validate_epoch(model, loader, *, stage, loss_config, device) -> dict[str, float]:
    if stage not in {"base", "finetune"}:
        raise ValueError(f"unknown stage: {stage}")
    model.eval()
    by_case = {}
    count = exact_count = rough_count = unknown_count = 0

    def record(case_id, metric, value):
        by_case.setdefault(case_id, {}).setdefault(metric, []).append(float(value))

    with torch.no_grad():
        for batch in loader:
            pred, label = _prediction(model, batch, device)
            squared = (pred - label).square()
            absolute = (pred - label).abs()
            n = pred.shape[0]
            case_ids = batch.get("case_id")
            if case_ids is None:
                # Direct callers without IDs retain slice-level behavior. The CLI requires IDs.
                case_ids = [f"__slice_{count + i}" for i in range(n)]
            if len(case_ids) != n:
                raise ValueError("case_id count does not match batch size")
            for i, case_id in enumerate(case_ids):
                record(case_id, "full_mse", squared[i].mean().item())
                record(case_id, "full_mae", absolute[i].mean().item())
            count += n
            if stage == "finetune":
                mask, quality = _masked_batch(batch, device)
                exact = quality == 1
                rough = ~exact
                exact_count += int(exact.sum().item())
                rough_count += int((quality == 0).sum().item())
                unknown_count += int((quality == -1).sum().item())
                if exact.any():
                    indices = exact.nonzero(as_tuple=True)[0].tolist()
                    exact_mask = mask[exact]
                    band = _boundary_band(exact_mask, loss_config["radius"])
                    values = {
                        "roi_exact_mse": _per_sample_mean(squared[exact], exact_mask),
                        "roi_exact_mae": _per_sample_mean(absolute[exact], exact_mask),
                        "boundary_mse": _per_sample_mean(squared[exact], band),
                        "boundary_mae": _per_sample_mean(absolute[exact], band),
                        "boundary_gradient_mae": _per_sample_mean(
                            _sobel_difference(pred[exact], label[exact]), band
                        ),
                    }
                    for j, i in enumerate(indices):
                        for key, value in values.items():
                            record(case_ids[i], key, value[j].item())
                if rough.any():
                    indices = rough.nonzero(as_tuple=True)[0].tolist()
                    soft = _soft_region(mask[rough], loss_config["rough_sigma"])
                    values = _per_sample_mean(squared[rough], soft)
                    for j, i in enumerate(indices):
                        record(case_ids[i], "roi_rough_mse", values[j].item())
    if count == 0:
        raise ValueError("validation loader is empty")
    metrics = {"num_samples": count, "num_cases": len(by_case)}
    keys = {key for values in by_case.values() for key in values}
    for key in keys:
        case_means = [sum(values[key]) / len(values[key])
                      for values in by_case.values() if key in values]
        metrics[key] = sum(case_means) / len(case_means)
    if stage == "finetune":
        metrics.update({"exact_count": exact_count, "rough_count": rough_count,
                        "unknown_count": unknown_count})
    return metrics


def choose_selection_metric(masked_metrics: dict[str, float]) -> tuple[float, str]:
    if masked_metrics.get("exact_count", 0):
        return masked_metrics["boundary_mse"], "boundary_mse"
    if masked_metrics.get("rough_count", 0) or masked_metrics.get("unknown_count", 0):
        return masked_metrics["roi_rough_mse"], "roi_rough_mse"
    raise ValueError("masked validation data contains no usable masks")


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("stage") not in (None, "base"):
        raise ValueError(f"base checkpoint required, got stage={checkpoint['stage']}")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def save_checkpoint(path, model, optimizer, scheduler, *, epoch, stage, loss_mode,
                    config, validation_metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "loss_mode": loss_mode,
        "config": config,
        "validation_metrics": validation_metrics,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _loader(dataset, *, batch_size, num_workers, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


def _log_epoch(directory, record):
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(line + "\n")
    print(line, flush=True)


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("base", "finetune", "both"), default="both")
    parser.add_argument("--base-train-list")
    parser.add_argument("--base-valid-list", required=True)
    parser.add_argument("--masked-train-list")
    parser.add_argument("--masked-valid-list")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--loss-mode", choices=("full", "roi", "boundary", "gradient"), default="boundary")
    parser.add_argument("--lambda-exact", type=float, default=1.0)
    parser.add_argument("--lambda-rough", type=float, default=0.5)
    parser.add_argument("--lambda-boundary", type=float, default=1.0)
    parser.add_argument("--lambda-gradient", type=float, default=0.0)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--rough-sigma", type=float, default=3.0)
    return parser


def main(argv=None):
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.stage in {"base", "both"} and not args.base_train_list:
        parser.error("--base-train-list is required for base and both")
    if args.stage in {"finetune", "both"} and not (args.masked_train_list and args.masked_valid_list):
        parser.error("--masked-train-list and --masked-valid-list are required for finetune and both")
    if args.stage == "finetune" and not args.base_checkpoint:
        parser.error("--base-checkpoint is required for finetune")
    if args.loss_mode == "gradient" and args.lambda_gradient <= 0:
        parser.error("--lambda-gradient must be positive for gradient mode")
    if min(args.batch_size, args.base_epochs, args.finetune_epochs, args.patience) < 1 or args.num_workers < 0:
        parser.error("batch size, epoch counts, and patience must be positive; workers must be nonnegative")

    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    output = Path(args.output_dir)
    model = Dense2D().to(device)
    config = vars(args).copy()
    loss_config = {
        "mode": args.loss_mode,
        "radius": args.boundary_radius,
        "lambda_exact": args.lambda_exact,
        "lambda_rough": args.lambda_rough,
        "lambda_boundary": args.lambda_boundary,
        "lambda_gradient": args.lambda_gradient,
        "rough_sigma": args.rough_sigma,
    }
    full_validation = npz_reader_2d_data_with_prior(
        args.base_valid_list, training=False, return_case_id=True
    )
    full_validation_loader = _loader(full_validation, batch_size=args.batch_size,
                                     num_workers=args.num_workers)

    if args.stage in {"base", "both"}:
        base_output = output / "base"
        if (base_output / "latest.pt").exists():
            raise FileExistsError(f"base checkpoint exists: {base_output / 'latest.pt'}")
        base_dataset = npz_reader_2d_data_with_prior(args.base_train_list, training=True)
        optimizer = SophiaG(model.parameters(), lr=args.base_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
        cumulative_samples_seen = 0
        for epoch in range(1, args.base_epochs + 1):
            subset = Subset(base_dataset, stage_indices(len(base_dataset), "base", seed=args.seed, epoch=epoch))
            train_loader = _loader(subset, batch_size=args.batch_size, num_workers=args.num_workers,
                                   shuffle=True)
            train_metrics = train_epoch(model, train_loader, optimizer, stage="base",
                                        loss_config=loss_config, device=device)
            valid_metrics = validate_epoch(model, full_validation_loader, stage="base",
                                           loss_config=loss_config, device=device)
            scheduler.step()
            cumulative_samples_seen += train_metrics["num_samples"]
            record = {"stage": "base", "epoch": epoch, "train": train_metrics,
                      "valid_full": valid_metrics,
                      "cumulative_samples_seen": cumulative_samples_seen}
            _log_epoch(base_output, record)
            save_checkpoint(base_output / "latest.pt", model, optimizer, scheduler,
                            epoch=epoch, stage="base", loss_mode="full", config=config,
                            validation_metrics=valid_metrics)
        base_checkpoint = base_output / "latest.pt"
    else:
        base_checkpoint = Path(args.base_checkpoint)

    if args.stage in {"finetune", "both"}:
        finetune_output = output / "finetune" / args.loss_mode
        if (finetune_output / "best.pt").exists() or (finetune_output / "latest.pt").exists():
            raise FileExistsError(f"finetune checkpoint exists: {finetune_output}")
        load_model_weights(model, base_checkpoint, device)
        config["base_checkpoint_path"] = str(base_checkpoint.resolve())
        config["base_checkpoint_sha256"] = checkpoint_sha256(base_checkpoint)
        seed_everything(args.seed + 1000003)
        masked_train = npz_reader_2d_data_with_prior(args.masked_train_list, training=True, require_mask=True)
        masked_valid = npz_reader_2d_data_with_prior(
            args.masked_valid_list, training=False, require_mask=True, return_case_id=True
        )
        if not len(masked_train) or not len(masked_valid):
            raise ValueError("masked train and validation lists must be nonempty")
        if args.loss_mode in {"boundary", "gradient"}:
            audit_train = npz_reader_2d_data_with_prior(
                args.masked_train_list, training=False, require_mask=True
            )
            exact_count = sum(audit_train[index]["mask_quality"].item() == 1
                              for index in range(len(audit_train)))
            if exact_count == 0:
                raise ValueError("boundary and gradient modes require at least one exact training mask")
        masked_valid_loader = _loader(masked_valid, batch_size=args.batch_size,
                                      num_workers=args.num_workers)
        optimizer = SophiaG(model.parameters(), lr=args.finetune_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
        best = float("inf")
        stale = 0
        for epoch in range(1, args.finetune_epochs + 1):
            train_loader = _loader(masked_train, batch_size=args.batch_size,
                                   num_workers=args.num_workers, shuffle=True)
            train_metrics = train_epoch(model, train_loader, optimizer, stage="finetune",
                                        loss_config=loss_config, device=device)
            full_metrics = validate_epoch(model, full_validation_loader, stage="base",
                                          loss_config=loss_config, device=device)
            masked_metrics = validate_epoch(model, masked_valid_loader, stage="finetune",
                                            loss_config=loss_config, device=device)
            score, score_name = choose_selection_metric(masked_metrics)
            scheduler.step()
            record = {"stage": "finetune", "epoch": epoch, "loss_mode": args.loss_mode,
                      "train": train_metrics, "valid_full": full_metrics,
                      "valid_masked": masked_metrics, "selection_metric": score_name,
                      "boundary_assessed": bool(masked_metrics["exact_count"])}
            _log_epoch(finetune_output, record)
            save_checkpoint(finetune_output / "latest.pt", model, optimizer, scheduler,
                            epoch=epoch, stage="finetune", loss_mode=args.loss_mode,
                            config=config, validation_metrics=record)
            if score < best:
                best = score
                stale = 0
                save_checkpoint(finetune_output / "best.pt", model, optimizer, scheduler,
                                epoch=epoch, stage="finetune", loss_mode=args.loss_mode,
                                config=config, validation_metrics=record)
            else:
                stale += 1
                if stale >= args.patience:
                    break


if __name__ == "__main__":
    main()
