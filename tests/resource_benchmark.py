"""Measure the planned training blocks and the 10/11-phase inference paths."""
from pathlib import Path
import argparse
import gc
import json
import time

import numpy as np
import torch
import yaml
from rstar4d.models.model import create_model
from rstar4d.utils.io import atomic_json, environment, sha256

from rstar_decnn import checkpoint
from rstar_decnn.backbone import restore_backbone
from rstar_decnn.model import Combined
from rstar_decnn.predict import fuse, predict_case
from rstar_decnn.train import _with_stam, optimizer_for, update

from .engineering_smoke import FIXTURES, ROOT


BLOCKS = ((16, 128, 128), (32, 96, 96), (64, 64, 64))


def timing(device, operation):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize(device)
    return value, {
        "seconds": time.perf_counter() - started, "initial_allocated_bytes": initial,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def train_patch(shape, policy, config, device, source_path, steps):
    backbone, spec, _ = restore_backbone(source_path, joint=True, prior_mode="all_views_fdk", engineering=True)
    if policy == "random_joint":
        backbone = create_model(2, spec["model_profile"], config["stam"])
        spec = _with_stam(spec, config)
    spec = _with_stam(spec, config)
    model = Combined(backbone, 0.0, config["upsampling"]).to(device)
    model.set_policy(policy)
    optimizer = optimizer_for(model, config)
    scaler = torch.cuda.amp.GradScaler(init_scale=512)
    x = torch.rand(1, 1, 11, *shape, device=device) * 0.03
    batch = {"x": x, "a_s": x.mean(dim=2), "a_d": 0.95 * x.mean(dim=2),
             "y": x * 0.7 + 0.009, "valid": torch.ones_like(x, dtype=torch.bool)}
    updates = []
    for _ in range(steps):
        result, measured = timing(device, lambda: update(model, batch, spec, optimizer, scaler, config))
        updates.append({**measured, "loss": result[0], "updated": result[2], "grad_scale": scaler.get_scale()})
    return {"shape_zyx": list(shape), "phase_count": 11, "policy": policy, "precision": "amp-fp16",
            "phase_batch": config["phase_batch"], "steps": updates}


def inference(path, device):
    saved = checkpoint.load(path)
    model = checkpoint.model_from(saved, device).eval()
    transform = saved["config"]["transform"]
    rng = np.random.default_rng(2026)
    shape = (17, 33, 35)
    tile = (16, 32, 32)
    overlap = (0, 16, 16)
    zz, yy, xx = np.indices(shape)
    core = (abs(zz - 8) <= 3) & (abs(yy - 16) <= 5) & (abs(xx - 17) <= 5)
    seam = ((abs(zz - 16) <= 1) | (abs(yy - 16) <= 1) | (abs(xx - 16) <= 1))
    result = []
    for phases in (10, 11):
        x = rng.uniform(0.0, 0.035, size=(phases, *shape)).astype(np.float32)
        sample = {"x": x, "a_s": x.mean(0), "a_d": 0.95 * x.mean(0)}
        outputs, measurements = {}, {}
        for mode in ("full", "tiled"):
            options = {"mode": mode, "tile": tile, "overlap": overlap, "device": device, "amp": False}
            outputs[mode], measurements[mode] = timing(device, lambda: predict_case(
                model, saved["backbone"], sample, transform, alpha=0.35,
                backbone_options=options, decnn_options=options, branches=("S", "D"),
            ))
            outputs[mode]["fused_035"] = fuse(outputs[mode]["S"], outputs[mode]["D"], 0.35, inference=True)
        errors = {}
        for name in outputs["full"]:
            full, tiled = (outputs[mode][name] for mode in ("full", "tiled"))
            if full.shape != x.shape or tiled.shape != x.shape:
                raise ValueError("padded inference did not restore the original grid")
            diff = np.abs(full - tiled)
            errors[name] = {"max_abs": float(diff.max()), "mae": float(diff.mean()),
                            "center_mae": float(diff[:, core].mean()), "seam_mae": float(diff[:, seam].mean())}
        result.append({"phases": phases, "shape_zyx": list(shape), "tile_zyx": list(tile),
                       "overlap_zyx": list(overlap), "precision": "fp32", "unit": transform["unit"],
                       "measurements": measurements, "full_tiled_difference": errors})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026)
    device = torch.device("cuda:0")
    cfg = yaml.safe_load((ROOT / "configs" / "scratch_stage2.yaml").read_text(encoding="utf-8"))
    source_path = FIXTURES / "joint" / "checkpoints" / "best.pt"
    records = {
        "environment": environment(), "engineering": True, "seed": 2026, "training": [],
        "training_options": {k: cfg[k] for k in ("loss", "stam", "decnn_recompute", "phase_batch")},
        "fp16_initial_grad_scale": 512, "inference_checkpoint_sha256": sha256(a.checkpoint),
        "frozen_source_sha256": sha256(source_path), "inference": inference(a.checkpoint, device),
    }
    atomic_json(a.out, records)
    for policy in ("frozen", "random_joint"):
        for shape in BLOCKS:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                item = train_patch(shape, policy, cfg, device, source_path, a.steps)
            except torch.cuda.OutOfMemoryError as exc:
                item = {"shape_zyx": list(shape), "policy": policy, "error": str(exc)}
            records["training"].append(item)
            atomic_json(a.out, records)
            print(json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
