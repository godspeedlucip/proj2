from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import (latest_checkpoint, load_checkpoint, random_state, restore_random_state,
                         save_torch, training_lock)
from .config import file_digest, network_units, software, write_json
from .data import load_array
from .inference import enhanced_paths, load_weights, select_device, weights_contract
from .losses import critic_losses, field_smoothness, generator_losses, local_ncc, warp_pull
from .models import Critic2D, RegisNet3D, RestoreNet2D
from .prepare import load_prepared


_WEIGHTS = {"restore": "G.pt", "register": "R.pt", "denoise": "O.pt"}


class Samples:
    def __init__(self, stage, cases, config, device, patch_size, args, restore_id, domain):
        self.stage, self.device, self.config, self.patch_size = stage, device, config, patch_size
        self.arrays = {}
        self.indices = []
        restore = (load_weights(args.restore, "restore", config, device, domain=domain)
                   if stage == "register" and cases else None)
        for root, case in cases:
            if stage == "register":
                paths = enhanced_paths(root, case, restore, restore_id, config, device, args.slice_batch)
                self.indices.extend((moving, fixed) for moving in paths for fixed in paths if moving != fixed)
                continue
            prior = root / case["pi"]
            shape = self._array(prior).shape
            if min(shape[1:]) < patch_size:
                raise ValueError(f"Patch size {patch_size} exceeds image shape {shape}: {prior}")
            if stage == "restore":
                inputs = [root / phase["c"] for phase in case["branches"]["pseudo"]]
            else:
                inputs = [prior] * len(case["branches"]["real"])
            self.indices.extend((source, prior, z) for source in inputs for z in range(shape[0]))

    def _array(self, path):
        if path not in self.arrays:
            self.arrays[path] = load_array(path)
        return self.arrays[path]

    def __len__(self):
        return len(self.indices)

    def batches(self, sampler, batch_size, limit):
        indices = torch.randperm(len(self), generator=sampler).tolist()
        for start in range(0, len(indices), batch_size):
            if limit is not None and start // batch_size >= limit:
                return
            pairs = [self._sample(self.indices[i], sampler) for i in indices[start:start + batch_size]]
            source, target = (torch.from_numpy(np.stack(array)).to(self.device)[:, None]
                              for array in zip(*pairs))
            if self.stage == "denoise":
                source = source + self.config["gamma"] * torch.randn_like(source)
            yield network_units(source, self.config), network_units(target, self.config)

    def _sample(self, item, sampler):
        source, target = (self._array(path) for path in item[:2])
        if self.stage == "register":
            return source, target
        z = item[2]
        size = self.patch_size
        _, h, w = target.shape
        y = torch.randint(h - size + 1, (), generator=sampler).item()
        x = torch.randint(w - size + 1, (), generator=sampler).item()
        return source[z, y:y + size, x:x + size], target[z, y:y + size, x:x + size]


def step_2d(models, optimizers, source, target, loss_mode, *, train):
    generator = models["model"]
    critic = models.get("critic")
    prediction = generator(source)
    metrics = {}
    if critic is None:
        metrics = generator_losses(None, prediction, target, source, "mse")
    else:
        fake, real, degraded = (critic.adapt(value) for value in (prediction, target, source))
        metrics = critic_losses(critic, fake, real, training=train)
        if train:
            optimizers["critic"].zero_grad(set_to_none=True)
            metrics["critic"].backward()
            optimizers["critic"].step()
            optimizers["critic"].zero_grad(set_to_none=True)
        critic.requires_grad_(False)
        try:
            metrics.update(generator_losses(critic, fake, real, degraded, loss_mode))
        finally:
            critic.requires_grad_(True)
    if train:
        optimizers["model"].zero_grad(set_to_none=True)
        metrics["generator"].backward()
        optimizers["model"].step()
    return metrics, prediction


def step_register(models, optimizers, source, target, *, train):
    field = models["model"](source, target)
    prediction = warp_pull(source, field)
    ncc = local_ncc(prediction, target)
    smoothness = field_smoothness(field)
    total = ncc + 4 * smoothness
    if train:
        optimizers["model"].zero_grad(set_to_none=True)
        total.backward()
        optimizers["model"].step()
    return {"register": total, "ncc": ncc, "smoothness": smoothness}, prediction


def _preview(source, prediction, target):
    images = [value[0, 0].detach() for value in (source, prediction, target)]
    if images[0].ndim == 3:
        images = [image[len(image) // 2] for image in images]
    pixels = torch.stack(images).cpu()
    if max(pixels.shape[-2:]) > 256:
        pixels = torch.nn.functional.interpolate(pixels[:, None], size=(256, 256),
                                                 mode="bilinear", align_corners=False)[:, 0]
    lower, upper = torch.quantile(pixels[-1].flatten(), torch.tensor([0.01, 0.99]))
    scaled = ((pixels - lower) / (upper - lower).clamp_min(1e-6)).clamp(0, 1)
    return torch.cat(list(scaled), dim=1).unsqueeze(0)


def _write_scalars(writer, prefix, values, step):
    for name, value in values.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def _run_epoch(samples, models, optimizers, sampler, settings, *, train, writer=None, step=0):
    for model in models.values():
        model.train(train)
    totals, count, updates, last, preview = {}, 0, 0, {}, None
    expected = (len(samples) + settings["batch_size"] - 1) // settings["batch_size"]
    if settings["max_batches"] is not None:
        expected = min(expected, settings["max_batches"])
    for source, target in samples.batches(sampler, settings["batch_size"], settings["max_batches"]):
        with torch.set_grad_enabled(train):
            if samples.stage == "register":
                metrics, prediction = step_register(models, optimizers, source, target, train=train)
            else:
                metrics, prediction = step_2d(models, optimizers, source, target, settings["loss"], train=train)
        values = torch.stack([value.detach() for value in metrics.values()]).cpu()
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"Non-finite {samples.stage} losses at step {step + updates + 1}")
        last = dict(zip(metrics, values.tolist()))
        for name, value in last.items():
            totals[name] = totals.get(name, 0.0) + value * len(source)
        if preview is None:
            preview = _preview(source, prediction, target)
        count += len(source)
        updates += 1
        if train and updates < expected and (step + updates) % settings["log_every"] == 0:
            _write_scalars(writer, "train/step", last, step + updates)
    return {"losses": {name: value / count for name, value in totals.items()}, "samples": count,
            "batches": updates, "last_batch": last, "preview": preview}


def _write_epoch(writer, state):
    step = state["global_step"]
    summary = state["summary"]
    _write_scalars(writer, "train/step", summary["train"]["last_batch"], step)
    _write_scalars(writer, "learning_rate", summary["learning_rate"], step)
    writer.add_scalar("progress/epoch", state["epoch"], step)
    writer.add_scalar("progress/updates", step, step)
    for split in ("train", "validation"):
        report = summary.get(split)
        if report:
            _write_scalars(writer, f"{split}/epoch", {**report["losses"], "samples": report["samples"],
                                                       "batches": report["batches"]}, step)
            writer.add_image(f"{split}/input_prediction_target", report["preview"], step)
    writer.flush()


def _export(output, state):
    stage = state["stage"]
    save_torch(output / _WEIGHTS[stage], {"stage": stage, "domain": state["settings"]["domain"],
               "epoch": state["epoch"], "model": state["models"]["model"],
               "contract": weights_contract(state["settings"]["config"], stage),
               "restore_sha256": state["settings"]["restore_sha256"]})


def train(args):
    config, cases = load_prepared(args.prepared)
    stage = args.stage
    output = Path(args.out).resolve()
    with training_lock(output):
        _train_locked(args, config, cases, stage, output)


def _train_locked(args, config, cases, stage, output):
    latest = latest_checkpoint(output)
    resume_path = (latest if args.resume == "latest" else Path(args.resume).resolve()) if args.resume else None
    if args.resume and resume_path is None:
        raise FileNotFoundError(f"No epoch checkpoint found in {output / 'checkpoints'}")
    if latest and resume_path != latest:
        raise ValueError(f"Use --resume for the latest epoch in {output}, or a new --out to branch")
    saved = load_checkpoint(resume_path) if resume_path else None
    if saved and (saved["version"] != 1 or saved["stage"] != stage):
        raise ValueError("Checkpoint format or training stage mismatch")
    previous = dict(saved["settings"]) if saved else {}
    if saved:
        previous.setdefault("init_sha256", None)
    if saved and args.init_weights:
        raise ValueError("Use --init-weights for a new run and --resume for an existing run")
    device = select_device(args.device or previous.get("device", "auto"))
    train_cases = [(path, row) for path, row in cases if row["split"] == "train"]
    domains = {row["domain"] for _, row in train_cases}
    domain = args.domain or previous.get("domain")
    if domain is None and len(domains) == 1:
        domain = next(iter(domains))
    if domain not in domains:
        raise ValueError(f"Select a training domain with --domain from {sorted(domains)}")
    train_cases = [(path, row) for path, row in train_cases if row["domain"] == domain]
    validation = [(path, row) for path, row in cases if row["split"] in {"val", "validation"} and row["domain"] == domain]
    if stage == "register" and not args.restore and saved:
        args.restore = saved.get("restore_path")
    if stage == "register" and not args.restore:
        raise ValueError("train register requires --restore G.pt")
    restore_id = file_digest(args.restore) if stage == "register" else None
    defaults = {"epochs": 50, "batch_size": 1 if stage == "register" else 32,
                "patch_size": 128, "lr_start": 1e-4 if stage == "register" else 1e-3,
                "lr_end": 1e-5, "max_batches": None, "loss": "full", "log_every": 10,
                "deterministic": False, "slice_batch": 4}
    settings = {name: (getattr(args, name) if getattr(args, name) is not None else previous.get(name, default))
                for name, default in defaults.items()}
    settings.update(config=config, domain=domain, restore_sha256=restore_id,
                    data=[(row["scan_id"], row["split"], row["cache_id"]) for _, row in train_cases + validation],
                    device=str(device), init_sha256=(file_digest(args.init_weights) if args.init_weights
                                                       else previous.get("init_sha256")))
    args.slice_batch = settings["slice_batch"]
    if saved and previous != settings:
        changed = sorted(key for key in settings if settings[key] != previous.get(key))
        raise ValueError(f"Resume requires the saved training settings. Changed: {', '.join(changed)}")
    epochs, first, last = (settings[key] for key in ("epochs", "lr_start", "lr_end"))
    if min(epochs, settings["batch_size"], settings["log_every"]) < 1 or not np.isfinite([first, last]).all() or min(first, last) <= 0:
        raise ValueError("Epochs, batch size, logging interval and learning rates must be positive")
    if settings["max_batches"] is not None and settings["max_batches"] < 1:
        raise ValueError("--max-batches must be positive")
    if stage != "register" and settings["patch_size"] < 32:
        raise ValueError("2D training requires patch size at least 32 for batch normalization")
    stop = args.stop_after_epoch if args.stop_after_epoch is not None else epochs
    completed = saved["epoch"] if saved else 0
    if not max(1, completed) <= stop <= epochs:
        raise ValueError("--stop-after-epoch must be within the remaining training schedule")
    if settings["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(settings["deterministic"])
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    sampler = torch.Generator().manual_seed(seed)
    train_samples = Samples(stage, train_cases, config, device, settings["patch_size"], args, restore_id, domain)
    if not len(train_samples):
        raise ValueError("No training samples for selected stage")
    val_samples = Samples(stage, validation, config, device, settings["patch_size"], args, restore_id, domain)
    models = {"model": (RegisNet3D() if stage == "register" else RestoreNet2D()).to(device)}
    if args.init_weights:
        initial = load_weights(args.init_weights, stage, config, device, domain=domain, restore_id=restore_id)
        models["model"].load_state_dict(initial.state_dict())
        del initial
    if stage != "register" and settings["loss"] != "mse":
        models["critic"] = Critic2D().to(device)
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=first) for name, model in models.items()}

    def lr_factor(index):
        progress = min(index, epochs - 1) / max(1, epochs - 1)
        return (first * (1 - progress) + last * progress) / first

    schedulers = {name: torch.optim.lr_scheduler.LambdaLR(opt, lr_factor) for name, opt in optimizers.items()}
    global_step = 0
    if saved:
        for name, model in models.items():
            model.load_state_dict(saved["models"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
            schedulers[name].load_state_dict(saved["schedulers"][name])
        global_step = saved["global_step"]
        print(f"{stage}: resumed after epoch {completed}, step {global_step}", flush=True)
    else:
        write_json(output / "training.json", {"stage": stage, **settings, "software": software(device)})
    if saved and not (output / "training.json").is_file():
        write_json(output / "training.json", {"stage": stage, **settings, "software": software(device)})
    log_dir = output / "tensorboard"
    # A failed epoch can have uncommitted events. Restore the final committed
    # step from its checkpoint and hide any events after it in the same run.
    purge = max(1, global_step) if list(log_dir.glob("events.out.tfevents.*")) else None
    with SummaryWriter(str(log_dir), purge_step=purge) as writer:
        if saved:
            if latest is None:
                save_torch(output / "checkpoints" / f"epoch_{completed:04d}.pt", saved)
            weight_path = output / _WEIGHTS[stage]
            if not weight_path.is_file() or load_checkpoint(weight_path)["epoch"] != completed:
                _export(output, saved)
            _write_epoch(writer, saved)
            restore_random_state(saved["random"], sampler, device)
        for epoch in range(completed + 1, stop + 1):
            lr = {name: opt.param_groups[0]["lr"] for name, opt in optimizers.items()}
            report = _run_epoch(train_samples, models, optimizers, sampler, settings,
                                train=True, writer=writer, step=global_step)
            global_step += report["batches"]
            summary = {"train": report, "learning_rate": lr}
            if len(val_samples):
                devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(seed)
                    summary["validation"] = _run_epoch(val_samples, models, optimizers,
                                                       torch.Generator().manual_seed(seed), settings, train=False)
            for scheduler in schedulers.values():
                scheduler.step()
            state = {"version": 1, "stage": stage, "epoch": epoch, "global_step": global_step,
                     "restore_path": str(Path(args.restore).resolve()) if stage == "register" else None,
                     "settings": settings, "models": {key: model.state_dict() for key, model in models.items()},
                     "optimizers": {key: opt.state_dict() for key, opt in optimizers.items()},
                     "schedulers": {key: sch.state_dict() for key, sch in schedulers.items()},
                     "summary": summary, "random": random_state(sampler, device)}
            path = output / "checkpoints" / f"epoch_{epoch:04d}.pt"
            save_torch(path, state)
            _export(output, state)
            _write_epoch(writer, state)
            print(f"{stage}: epoch {epoch}/{epochs}, batches {report['batches']}, step {global_step}, "
                  + ", ".join(f"{k}={v:.6g}" for k, v in report["losses"].items()), flush=True)
