"""End-to-end CUDA engineering acceptance using a local upstream fixture."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import torch
import yaml
from rstar4d.data.contracts import basic_metadata, write_case, write_manifest
from rstar4d.training.checkpoint import load_checkpoint as upstream_load
from rstar4d.utils.io import atomic_json, atomic_jsonl, canonical_hash, read_json, sha256

from rstar_decnn.train import _fresh, read_config
from rstar_decnn.data import load_rows, read_case, sidecar_metadata


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT.parent / "RSTAR-STAM" / "runs" / "stam-acceptance" / "phantom"


def grid(shape):
    affine = np.eye(4)
    affine[:3, 3] = -(np.asarray(shape) - 1) / 2
    return {"shape_zyx": list(shape), "voxel_spacing_zyx_mm": [1, 1, 1], "center_zyx_mm": [0, 0, 0],
            "voxel_to_scanner_zyx": affine.tolist()}


def make_inputs(out):
    out.mkdir(parents=True, exist_ok=True)
    shape = (17, 32, 17)
    coordinates = np.indices(shape, dtype=np.float32)
    radius_sq = sum((g - (n - 1) / 2) ** 2 for g, n in zip(coordinates, shape))
    rows, annotations, prior_index = [], [], []
    rng = np.random.default_rng(2026)
    phases = [9, 1, 7, 3, 4, 2, 5, 8, 6, 11, 10]
    for case, count in (("train", 10), ("val", 10), ("test", 11)):
        path = out / f"{case}.h5"
        masks = np.stack([radius_sq <= r ** 2 for r in np.linspace(5, 3, count)])
        target = np.asarray(0.018 + 0.013 * masks + 0.002 * np.cos(coordinates[2] / 4), np.float32)
        target[:, 0, 0, 0] = -0.005
        x = target + rng.normal(0, 0.0015, size=target.shape).astype(np.float32)
        valid = np.ones_like(target, dtype=np.uint8)
        valid[:, 0, 0, -1] = 0
        meta = basic_metadata(
            count, [1, 1, 1], phase_labels=phases[:count], geometry=grid(shape),
            prior_mode="phase_weighted_fdk", generation_profile="engineering_fixture",
        )
        write_case(path, x, x.mean(0), target[::-1], valid, meta)
        with h5py.File(path, "r+") as f:
            f.attrs["target_metadata_json"] = json.dumps({"phase_labels": phases[:count][::-1]})
        row = {"case_id": case, "animal_id": case, "sha256": sha256(path), "path": str(path)}
        rows.append(row)
        sidecar = out / f"{case}-decnn-fdk-fixture.h5"
        with h5py.File(sidecar, "w") as f:
            f.attrs["metadata_json"] = json.dumps(sidecar_metadata(
                {**row, "metadata": meta}, {"method": "synthetic-prior-fixture",
                                           "mixed_projections_sha256": canonical_hash(row)}
            ))
            f["prior"] = 0.52 * x[0] + 0.48 * x[-1]
        prior_index.append({"case_id": case, "path": sidecar.name, "sha256": sha256(sidecar)})
        annotation = out / f"{case}-lv.h5"
        with h5py.File(annotation, "w") as f:
            f["ED"], f["ES"] = masks[0], masks[-1]
        for role, i in (("ED", 0), ("ES", -1)):
            annotations.append({
                "case_id": case, "source_sha256": row["sha256"], "phase_label": phases[:count][i],
                "role": role, "path": annotation.name, "sha256": sha256(annotation), "dataset": role,
            })
    write_manifest(out / "source.jsonl", rows)
    atomic_jsonl(out / "annotations.jsonl", annotations)
    atomic_jsonl(out / "prior-index.jsonl", prior_index)
    atomic_json(out / "split.json", {key: [key] for key in ("train", "val", "test")})

    class FixedContour(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", torch.from_numpy(radius_sq <= 16)[None, None])

        def forward(self, images):
            return self.mask.to(images.dtype) + images * 0

    with (out / "extractor.pt").open("wb") as file:
        torch.jit.save(torch.jit.trace(FixedContour(), torch.zeros(1, 1, *shape)), file)


def run_cli(out, module, *args):
    argv = [sys.executable, "-m", module, *map(str, args)]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    with (out / "commands.log").open("a", encoding="utf-8") as file:
        print(argv, result.stdout, result.stderr, sep="\n", file=file)
    if result.returncode:
        raise RuntimeError(f"{argv}\n{result.stdout}\n{result.stderr}")


def _equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(l, r) for l, r in zip(left, right))
    return left == right


def run_smoke(out, device, routes, random_sam_device):
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = FIXTURES / "joint" / "checkpoints" / "best.pt"
    ordinary = FIXTURES / "public_stage2_fixture.pt"
    data = out / "data"
    make_inputs(data)
    parameters = ["--manifest", data / "prepared.jsonl", "--split", data / "split.json"]
    cli = lambda *args: run_cli(out, "rstar_decnn", *args)
    cli("prepare", "--manifest", data / "source.jsonl", "--out", parameters[1],
        "--prior-mode", "phase_weighted_fdk", "--prior-index", data / "prior-index.jsonl", "--engineering")
    result = {}
    cfg = ROOT / "configs" / "frozen.yaml"
    config = yaml.safe_load(cfg.read_text(encoding="utf-8"))

    for recipe in routes:
        print(f"Running {recipe}", flush=True)
        route_device = random_sam_device if recipe in {"scratch_stage2", "tetris"} else device
        current = deepcopy(config)
        current.update(recipe=recipe, backbone_checkpoint=str(source.resolve()) if recipe in {"frozen", "warmup_unfreeze"} else None,
                       device=route_device, engineering=True, prior_mode="phase_weighted_fdk", amp=False, deterministic=True,
                       steps=2 if recipe == "frozen" else 3, validation_every=3, checkpoint_every=3,
                       stage1_steps=int(recipe == "tetris"), warmup_steps=int(recipe == "warmup_unfreeze"),
                       stage2_patches=[[16, 32, 16]], stage1_patch=[1, 16, 16], phase_batch=5,
                       milestones=[1, 2], threads=2, loss={"kind": "l1"},
                       validation_inference={"mode": "full", "backbone_tile": [16, 16, 16],
                                             "backbone_overlap": [0, 0, 0], "decnn_tile": [16, 16, 16],
                                             "decnn_overlap": [0, 0, 0]})
        cfg = out / f"{recipe}.yaml"
        cfg.write_text(yaml.safe_dump(current), encoding="utf-8")
        train_path = out / recipe
        cli("train", "--config", cfg, *parameters, "--out", train_path)
        trained = read_json(train_path / "summary.json")
        saved = torch.load(train_path / "last.pt", weights_only=True, map_location="cpu")
        result[recipe] = {"steps": trained["step"], "policy": trained["policy"], "device": route_device, "alpha": saved["alpha"],
                          "groups": saved["training"]["groups"]}
        if recipe in {"frozen", "warmup_unfreeze"}:
            upstream = upstream_load(source, require_training=False)
            prefix = "backbone." if recipe == "frozen" else "backbone.stam.sam."
            keys = [key for key in saved["weights"] if key.startswith(prefix)]
            assert keys and all(torch.equal(saved["weights"][key], upstream["state_dict"][key[9:]]) for key in keys)
            result[recipe]["frozen_tensors_exact"] = len(keys)
            del upstream
        if recipe == "frozen":
            rows, split = load_rows(parameters[1], parameters[3])
            torch.manual_seed(current["seed"])
            initial, _, _ = _fresh(read_config(cfg), rows, split)
            changed = [key for key, param in initial.named_parameters() if key.startswith("decnn.")
                       and not torch.equal(param, saved["weights"][key])]
            assert changed
            result[recipe]["decnn_parameters_changed"] = len(changed)
            result[recipe]["checkpoint"] = str(trained["best"])
            del initial
        if recipe in {"tetris", "warmup_unfreeze", "scratch_stage2", "frozen"}:
            resumed = out / f"{recipe}-interrupted"
            base = ("train", "--config", cfg, *parameters, "--out", resumed)
            cli(*base, "--stop-after", 1)
            before = read_json(resumed / "summary.json")
            if recipe != "frozen":
                cli(*base, "--resume", resumed / "last.pt", "--stop-after", 2)
            after = read_json(resumed / "summary.json")
            cli(*base, "--resume", resumed / "last.pt")
            restored = torch.load(resumed / "last.pt", weights_only=True, map_location="cpu")
            for field in ("weights", "alpha"):
                assert _equal(saved[field], restored[field]), (recipe, field)
            for field in ("optimizer", "scheduler", "scaler", "rng", "groups", "transitions"):
                assert _equal(saved["training"][field], restored["training"][field]), (recipe, field)
            result[recipe]["resume_exact"] = True
            result[recipe]["boundary_policies"] = [before["policy"], after["policy"]]
            del restored
            prediction = out / f"{recipe}-reload"
            cli("predict", "--checkpoint", trained["best"], *parameters, "--out", prediction,
                "--mode", "full", "--device", route_device, "--alpha", "0.35")
            from rstar_decnn import checkpoint as checkpoint_tools
            from rstar_decnn.predict import predict_case
            loaded_model = checkpoint_tools.model_from(saved, route_device).eval()
            rows, _ = load_rows(parameters[1], parameters[3])
            heldout_row = next(row for row in rows if row["case_id"] == "test")
            opts = {"mode": "full", "device": route_device, "tile": (16, 16, 16),
                    "overlap": (0, 0, 0), "amp": False}
            in_process = predict_case(loaded_model, saved["backbone"], read_case(heldout_row),
                                      current["transform"], alpha=0.35,
                                      backbone_options=opts, decnn_options=opts)
            with h5py.File(prediction / "test" / "prediction.h5", "r") as file:
                actual = file["prediction"][:]
            np.testing.assert_allclose(in_process, actual, atol=1e-7, rtol=1e-5)
            result[recipe]["new_process_max_abs"] = float(np.max(np.abs(in_process - actual)))
            del loaded_model
        del saved
        atomic_json(out / "summary.json", result)

    if "frozen" not in routes:
        return
    checkpoint = result["frozen"]["checkpoint"]
    options = ["--checkpoint", checkpoint, *parameters, "--device", device, "--mode", "full"]
    for split in ("val", "test"):
        cli("predict", *options, "--cache", "--rstar", ordinary, "--split-name", split,
            "--out", out / f"cache-{split}")
    run_cli(out, "rstar4d", "infer", "--checkpoint", source, "--manifest", data / "source.jsonl",
            "--split", data / "split.json", "--split-name", "test", "--mode", "full", "--device", device,
            "--fp32", "--out", out / "original")
    from rstar_decnn.predict import fuse

    parity = {}
    with h5py.File(out / "cache-test" / "test" / "prediction.h5", "r") as f:
        cached = {key: f[key][:] for key in ("R", "S", "D")}
        meta = json.loads(f.attrs["metadata_json"])
    with h5py.File(out / "original" / "test" / "prediction.h5", "r") as f:
        expected = f["prediction"][:]
    np.testing.assert_allclose(expected, cached["S"], atol=1e-7, rtol=1e-5)
    parity["upstream_checkpoint_sha256"] = sha256(source)
    parity["phase_labels"] = meta["phase_labels"]
    parity["unit"] = meta["unit"]
    parity["upstream_phase_max_abs"] = np.max(np.abs(expected - cached["S"]), axis=(1, 2, 3)).tolist()
    for alpha in (0, 0.35, 1):
        dirname = out / f"alpha-{alpha}"
        cli("predict", *options, "--alpha", alpha, "--out", dirname)
        with h5py.File(dirname / "test" / "prediction.h5", "r") as f:
            predicted = f["prediction"][:]
        p = fuse(cached["S"], cached["D"], alpha, inference=True)
        np.testing.assert_allclose(predicted, p, atol=1e-7, rtol=1e-5)
        parity[f"alpha_{alpha}_max_abs"] = float(np.max(np.abs(predicted - p)))
    result["parity"] = parity
    metric = ["--annotations", data / "annotations.jsonl", "--extractor", data / "extractor.pt"]
    lock_file = out / "validation-lock.json"
    cli("calibrate", *parameters, *metric, "--cache", out / "cache-val" / "manifest.jsonl", "--out", lock_file)
    cli("evaluate", *parameters, *metric, "--cache", out / "cache-test" / "manifest.jsonl",
        "--lock", lock_file, "--out", out / "heldout")
    heldout = read_json(out / "heldout" / "summary.json")
    assert len(heldout["groups"]) == 5
    result["calibration"] = heldout["alpha"]
    result["engineering_decision"] = heldout["decision"]
    atomic_json(out / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-sam-device", default="cpu", help="device for exact resume of trainable SAM")
    parser.add_argument("--routes", nargs="+", default=["frozen", "scratch_stage2", "tetris", "warmup_unfreeze"])
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    run_smoke(args.out, args.device, args.routes, args.random_sam_device)
