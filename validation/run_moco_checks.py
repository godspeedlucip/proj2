"""Exercise the MoCo CLI on synthetic scans from the local RSTAR4D acceptance run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch
import yaml

from rstar4d.data.contracts import basic_metadata, read_manifest, write_case, write_manifest
from rstar4d.moco.data import case_rows
from rstar4d.moco.prepare import calibrate, pigae, read_views
from rstar4d.moco.registration import RegisNet3D, estimate_field
from rstar4d.simulation.backend import backend_fingerprint, geometry, project, reconstruct
from rstar4d.simulation.reference import SeriesSettings, TigreTool
from rstar4d.utils.io import atomic_json, read_json, read_jsonl, sha256


def prepare_tolerance(manifest, evidence, folder):
    row, = case_rows(manifest)
    meta = row["metadata"]
    settings = SeriesSettings(**meta["settings"])
    geo = geometry(meta["source_shape_zyx"], meta["source_spacing_zyx_mm"], settings)
    index, angles, state, mixed, _, digest, _, _ = read_views(row, geo, settings)
    repeat = []
    for label in meta["phase_labels"]:
        selected = state == label
        a, _ = reconstruct(mixed[selected], geo, angles[selected], settings)
        b, _ = reconstruct(mixed[selected], geo, angles[selected], settings)
        repeat.append({"label": label, "max_abs": float(np.abs(a - b).max())})
    atol = max(1e-7, 2 * max(item["max_abs"] for item in repeat))
    atomic_json(evidence, {"scope": "synthetic engineering check", "unit": "mm^-1", "expected_atol": atol,
                           "rtol": 0., "paired_sha256": row["sha256"], "view_sha256": digest,
                           "backend": backend_fingerprint(), "repeat_fdk": repeat,
                           "floor": "1e-7 mm^-1 numerical floor for the engineering fixture"})
    # Replay branch receives only a view table, in a different storage order.
    acquisition = folder / "acquisition.h5"
    with h5py.File(acquisition, "w") as file:
        file["projection_index"] = index[::-1]
        file["angles_rad"] = angles[::-1]
        file["state_label"] = state[::-1]
        file.attrs["geometry_json"] = json.dumps(meta["geometry"])
    replay = folder / "val_replay.jsonl"
    row.pop("projections")
    row.pop("projections_sha256")
    row.update(acquisition=str(acquisition), acquisition_sha256=sha256(acquisition))
    write_manifest(replay, [row])
    return atol, replay


def tiled_field_check(device):
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    moving, fixed = torch.rand((2, 1, 1, 145, 17, 19), device=device)
    model = RegisNet3D().to(device).eval()
    result, fields = {}, []
    with torch.inference_mode():
        for name, tile in (("full", None), ("tiled", (61, 17, 19))):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            fields.append(estimate_field(model, moving, fixed, 4., tile, 64).cpu())
            result[name] = {"tile": tile, "peak_allocated_bytes":
                            torch.cuda.max_memory_allocated() if device == "cuda" else None}
    difference = (fields[0] - fields[1]).abs()
    result["max_abs_vox"] = float(difference.max())
    result["seam_max_abs_vox"] = float(difference[:, :, [60, 61, 121, 122]].max())
    torch.testing.assert_close(*fields, atol=1e-7, rtol=1e-5)
    return result


def static_physics_check():
    shape = (16, 16, 16)
    z, y, x = np.indices(shape) - 7.5
    obj = (((z + 1) / 4) ** 2 + ((y - 1) / 5) ** 2 + (x / 3) ** 2 < 1).astype(np.float32) * .02
    settings = SeriesSettings(
        voxel_spacing_multiplier=1, n_detector_rows=32, n_detector_cols=32,
        d_detector_row_mm=1, d_detector_col_mm=1, fov_source_pixels=16, fov_output_pixels=16,
        padding_row_pixels=0, padding_col_pixels=0, fov_mask_shrink_pixels=0)
    geo = geometry(shape, (1, 1, 1), settings)
    angles = TigreTool.make_angles(48)
    prior, _ = reconstruct(project(obj, geo, angles), geo, angles, settings)
    views = angles[1::3]
    sparse, reprojection = [reconstruct(project(prior, geo, views), geo, views, settings)[0] for _ in range(2)]
    coefficients = [2., .98, .001]
    correction = pigae(sparse, prior, reprojection, "static_consistent", coefficients)
    expected = calibrate(prior, coefficients)
    np.testing.assert_allclose(correction, expected, rtol=0, atol=1e-7)
    return {"views": len(views), "max_abs_raw_sparse_minus_q": float(np.abs(sparse - reprojection).max()),
            "max_abs_correction_minus_calibrated_prior": float(np.abs(correction - expected).max())}


def check_contrast_only(root, run):
    folder = root / "contrast_only"
    folder.mkdir()
    shape = (9, 25, 25)
    z, y, x = np.indices(shape)
    contrast = np.clip((6.5 - np.abs(x - 12)) / 2., 0, 1).astype(np.float32)
    contrast *= (abs(z - 4) <= 3) & (abs(y - 12) <= 7)
    stacks = np.stack([.012 + strength * contrast for strength in (.011, .020, .014)]).astype(np.float32)
    id_map = np.broadcast_to(((abs(x - 12) <= 2) & (abs(y - 12) <= 3)
                              & (abs(z - 4) <= 1)).astype(np.int32), stacks.shape)
    affine = np.eye(4)
    meta = basic_metadata(3, [1., 1., 1.], target_kind="analytic_contrast_only",
                          geometry={"shape_zyx": list(shape), "voxel_to_scanner_zyx": affine.tolist(),
                                    "center_zyx_mm": [4., 12., 12.], "voxel_spacing_zyx_mm": [1., 1., 1.]})
    paired = folder / "paired.h5"
    write_case(paired, stacks, stacks.mean(0), stacks, np.ones_like(stacks, bool), meta, id_map)
    manifest = folder / "manifest.jsonl"
    digest = sha256(paired)
    write_manifest(manifest, [{"case_id": "contrast_only", "animal_id": "analytic", "path": str(paired), "sha256": digest}])
    annotations = folder / "annotations.json"
    atomic_json(annotations, {"cases": {"contrast_only": {"source_sha256": digest, "wall_points": {
        str(label): [{"id": "x-positive", "center_scanner_zyx_mm": [4., 12., 17.5],
                      "normal_scanner_zyx": [0., 0., 1.]}] for label in (1, 2, 3)}}}})
    metrics = folder / "metrics.yaml"
    metrics.write_text(yaml.safe_dump({"annotations": str(annotations), "blood_pool_label": 1,
                                      "reference_group": "x"}), encoding="utf-8")
    cfgs = root / "configs"
    run("infer-moco", "--config", cfgs / "a_zero.yaml", "--manifest", manifest, "--out", folder / "zero")
    run("infer-moco", "--config", cfgs / "a2_input_x.yaml", "--manifest", manifest, "--out", folder / "learned",
        "--register-checkpoint", root / "reg_x/checkpoints/best.pt")
    run("evaluate-moco", "--manifest", manifest, "--metric-config", metrics, "--out", folder / "scores",
        "--predictions", f"x={folder / 'zero/x/manifest.jsonl'}",
        "--predictions", f"zero={folder / 'zero/d/manifest.jsonl'}",
        "--predictions", f"learned={folder / 'learned/d/manifest.jsonl'}")
    case, = read_jsonl(folder / "scores/cases.jsonl")
    assert case["case_equal_phase"]["x"]["full"]["blood_pool_curve_mae"] == 0.
    assert case["case_equal_phase"]["x"]["full"]["edge_location_error_mm"] == 0.
    assert case["case_equal_phase"]["zero"]["full"]["blood_pool_curve_mae"] > 0.
    assert len(case["registration_qa"]["learned"]["pairs"]) == 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="demo_final directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    source = args.source.resolve()
    root = args.out.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("--out must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]

    def run(*options):
        command = [sys.executable, "-m", "rstar4d", *map(str, options)]
        print(">", subprocess.list2cmdline(command), flush=True)
        return subprocess.run(command, cwd=repository, check=True, capture_output=False)

    cases = read_manifest(source / "prepared" / "manifest.jsonl")
    index = {row["animal_id"]: row["case_id"] for row in cases}
    scan_split = {split: [index[animal] for animal in animals]
                  for split, animals in json.loads((source / "raw" / "split.json").read_text()).items()}
    atomic_json(root / "scan_split.json", scan_split)
    split_dir = root / "splits"
    run("split-moco", "--manifest", source / "prepared" / "manifest.jsonl",
        "--scan-split", root / "scan_split.json", "--out", split_dir)
    train = split_dir / "train_cases.jsonl"
    val = split_dir / "val_cases.jsonl"
    checkpoint = root / "public_stage2.pt"
    if not checkpoint.exists():
        run("convert-weights", "--input", repository / "RSTAR/githubcode/checkpoints/checkpoints_Tetris2.pth",
            "--out", checkpoint, "--stage", 2, "--intensity-domain", "mu", "--normalization-scale", .02)

    cfgs = root / "configs"
    cfgs.mkdir(exist_ok=True)
    pigae = {}
    if args.device == "cuda":
        evidence = root / "fdk_tolerance_evidence.json"
        atol, replay = prepare_tolerance(val, evidence, root)
        pigae = {"atol": atol, "rtol": 0., "tolerance_evidence": str(evidence)}
    for path in sorted((repository / "configs/moco").glob("*.yaml")):
        run("infer-moco", "--dry-run", "--config", path, "--out", root / "expanded" / path.stem)
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        config.update(device=args.device, baseline={"tile": [16, 32, 32], "overlap": [8, 16, 16], "amp": False})
        config.setdefault("registration", {}).update(epochs=2, steps_per_epoch=3)
        if config.get("pigae"):
            config["pigae"].update(pigae)
        (cfgs / path.name).write_text(yaml.safe_dump(config), encoding="utf-8")

    for split, manifest in (("train", train), ("val", val)):
        run("infer-moco", "--config", cfgs / "baseline.yaml", "--manifest", manifest,
            "--baseline-checkpoint", checkpoint, "--out", root / ("baseline_" + split))
        original = root / ("original_" + split)
        run("infer", "--checkpoint", checkpoint, "--manifest", manifest, "--split-name", "all",
            "--out", original, "--mode", "tiled", "--tile", 16, 32, 32, "--overlap", 8, 16, 16,
            "--device", args.device, "--fp32")
        item = read_json(root / f"baseline_{split}/summary.json")["cases"][0]["b"]
        with h5py.File(original / item["case_id"] / "prediction.h5", "r") as expected, h5py.File(
            item["path"], "r"
        ) as actual:
            np.testing.assert_allclose(actual["prediction"][:], expected["prediction"][:], atol=1e-6, rtol=1e-5)
    atomic_json(root / "tiled_field_parity.json", tiled_field_check(args.device))

    b_run = root / "baseline_val/b/manifest.jsonl"
    model_dir = root / "reg_b"
    x_model_dir = root / "reg_x"
    b_train_options = ("--config", cfgs / "a_learned.yaml", "--manifest", train,
                       "--val-manifest", val, "--baseline", root / "baseline_train/b/manifest.jsonl",
                       "--val-baseline", b_run, "--out", model_dir)
    if not (model_dir / "checkpoints/last.pt").exists():
        run("train-register", *b_train_options, "--stop-after-epoch", 1)
    run("train-register", *b_train_options, "--resume", model_dir / "checkpoints/last.pt")
    run("train-register", "--config", cfgs / "a2_input_x.yaml", "--manifest", train,
        "--val-manifest", val, "--out", x_model_dir, "--stop-after-epoch", 1)
    check_contrast_only(root, run)
    for name, reg, extras in (
        ("a_zero", None, []),
        ("a_learned", model_dir / "checkpoints/best.pt", ["--baseline", b_run]),
        ("a2_input_x", x_model_dir / "checkpoints/best.pt", []),
    ):
        run("infer-moco", "--config", cfgs / (name + ".yaml"), "--manifest", val, "--out", root / name,
            *(["--register-checkpoint", reg] if reg else []), *extras)
    ten_base = root / "baseline_train/b/manifest.jsonl"
    ten_config = yaml.safe_load((cfgs / "a_learned.yaml").read_text(encoding="utf-8"))
    ten_config.update(outputs=["d"], fusion={"pairs": "all"})
    ten_path = cfgs / "engineering_10_all_x.yaml"
    ten_path.write_text(yaml.safe_dump(ten_config), encoding="utf-8")
    for name, config in (("ten_adjacent", cfgs / "a_learned.yaml"), ("ten_all", ten_path)):
        run("infer-moco", "--config", config, "--manifest", train, "--out", root / name,
            "--register-checkpoint", model_dir / "checkpoints/best.pt", "--baseline", ten_base)
    ten_adj = read_json(root / "ten_adjacent/summary.json")
    ten_all = read_json(root / "ten_all/summary.json")
    assert (ten_adj["fields_total"], ten_adj["forward_calls"]) == (18, 18)
    assert (ten_all["fields_total"], ten_all["cache_hits"], ten_all["forward_calls"]) == (90, 18, 72)
    run("evaluate-moco", "--manifest", val, "--split-name", "val", "--out", root / "scores_a",
        "--predictions", f"b={b_run}", "--predictions", f"x={root / 'a_zero/x/manifest.jsonl'}",
        "--predictions", f"zero={root / 'a_zero/d/manifest.jsonl'}",
        "--predictions", f"learned={root / 'a_learned/d/manifest.jsonl'}",
        "--predictions", f"input_x={root / 'a2_input_x/d/manifest.jsonl'}")
    if args.device == "cpu":
        return
    atomic_json(root / "static_physics.json", static_physics_check())
    run("prepare-moco", "--config", cfgs / "b_source_c.yaml", "--manifest", val, "--out", root / "correction")
    run("prepare-moco", "--config", cfgs / "b_source_c.yaml", "--manifest", replay,
        "--raw-manifest", source / "raw/raw_manifest.jsonl", "--out", root / "correction_replay")
    run("prepare-moco", "--config", cfgs / "printed.yaml", "--manifest", val, "--out", root / "correction_printed")
    reg_all = root / "reg_all"
    run("train-register", "--config", cfgs / "d_train_all.yaml", "--manifest", train, "--val-manifest", val,
        "--baseline", root / "baseline_train/b/manifest.jsonl", "--val-baseline", b_run,
        "--out", reg_all, "--stop-after-epoch", 1)
    suffixes = ("b_source_x", "b_source_b", "b_source_c", "c_alpha_0", "c_alpha_025", "c_alpha_05", "c_alpha_1",
                "d_infer_all", "d_train_all", "printed", "raw_sum")
    for name in suffixes:
        correction = "correction_printed" if name == "printed" else "correction"
        reg = reg_all if name == "d_train_all" else model_dir
        run("infer-moco", "--config", cfgs / (name + ".yaml"), "--manifest", val, "--out", root / name,
            "--register-checkpoint", reg / "checkpoints/best.pt", "--baseline", b_run,
            "--correction", root / correction / "manifest.jsonl")
    controls = (("b", b_run), ("source_x", root / "b_source_x/d/manifest.jsonl"),
                ("source_b", root / "b_source_b/d/manifest.jsonl"), ("source_c", root / "b_source_c/d/manifest.jsonl"),
                ("printed", root / "printed/d/manifest.jsonl"), ("raw_sum_control", root / "raw_sum/d/manifest.jsonl"))
    run("evaluate-moco", "--manifest", val, "--split-name", "val", "--out", root / "scores_b",
        *(argument for name, path in controls for argument in ("--predictions", f"{name}={path}")))
    controls = (("b", b_run), ("adjacent", root / "b_source_c/d/manifest.jsonl"),
                ("all", root / "d_infer_all/d/manifest.jsonl"), ("all_trained", root / "d_train_all/d/manifest.jsonl"))
    run("evaluate-moco", "--manifest", val, "--split-name", "val", "--out", root / "scores_d",
        *(argument for name, path in controls for argument in ("--predictions", f"{name}={path}")))
    assert read_json(root / "a_zero/summary.json")["forward_calls"] == 0
    assert read_json(root / "b_source_c/summary.json")["cache_hits"] == 20
    all_pairs = read_json(root / "d_infer_all/summary.json")
    assert (all_pairs["fields_total"], all_pairs["forward_calls"], all_pairs["cache_hits"]) == (110, 90, 20)
    assert read_json(root / "d_train_all/summary.json")["forward_calls"] == 110
    correction_files = ["correction/phantom_1/correction.h5", "correction_replay/phantom_1/correction.h5"]
    with h5py.File(root / correction_files[0], "r") as saved, h5py.File(root / correction_files[1], "r") as replayed:
        np.testing.assert_allclose(saved["correction"][:], replayed["correction"][:], rtol=0., atol=atol)
    for name in ("y_single", "y_multi"):
        with h5py.File(root / f"c_alpha_0/{name}/phantom_1/prediction.h5", "r") as output:
            item = read_json(root / "baseline_val/summary.json")["cases"][0]["b"]
            with h5py.File(item["path"], "r") as baseline:
                np.testing.assert_array_equal(output["prediction"][:], baseline["prediction"][:])
    for alpha in ("025", "05", "1"):
        with h5py.File(root / f"c_alpha_{alpha}/y_single/phantom_1/prediction.h5", "r") as single, h5py.File(
            root / f"c_alpha_{alpha}/y_multi/phantom_1/prediction.h5", "r"
        ) as multi:
            assert not np.array_equal(single["prediction"][:], multi["prediction"][:])
    candidates = []
    for alpha in ("0", "025", "05", "1"):
        for kind in ("y_single", "y_multi"):
            candidates += ["--predictions", f"{alpha}_{kind}={root / f'c_alpha_{alpha}/{kind}/manifest.jsonl'}"]
    run("evaluate-moco", "--manifest", val, "--split-name", "val", "--out", root / "scores_c",
        "--predictions", f"b={b_run}", *candidates)
    lock = root / "test_lock.json"
    run("lock-moco", "--validation", root / "scores_c/summary.json", "--out", lock)
    test_manifest = split_dir / "test_cases.jsonl"
    run("infer-moco", "--config", cfgs / "baseline.yaml", "--manifest", test_manifest,
        "--split-name", "test", "--test-lock", lock, "--baseline-checkpoint", checkpoint, "--out", root / "test_b")
    run("evaluate-moco", "--manifest", test_manifest, "--split-name", "test", "--test-lock", lock,
        "--out", root / "scores_test", "--predictions", f"b={root / 'test_b/b/manifest.jsonl'}")


if __name__ == "__main__":
    main()
