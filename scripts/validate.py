"""Run TIGRE phantom, CLI integration, and interrupted-training checks."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpi_moco.checkpoint import load_checkpoint
from dpi_moco.config import read_json, software, write_json
from dpi_moco.physics import TigreProjector
from dpi_moco.prepare import save_array


def phantom(output):
    shape = (16, 32, 32)
    grid = {"shape_zyx": list(shape), "spacing_xyz_mm": [1., 1., 1.],
            "origin_xyz_mm": [-15.5, -15.5, -7.5], "direction_xyz": np.eye(3).tolist()}
    geo = {"DSO_mm": 400., "DSD_mm": 600., "nDetector_vu": [48, 64], "dDetector_vu_mm": [1., 1.]}
    fdk = {"filter": "ram_lak", "padding_vu": [0, 0], "dowang": False}
    z, y, x = np.indices(shape)
    base = np.where((x - 15.5) ** 2 + (y - 15.5) ** 2 <= 10**2, 0.012, 0).astype(np.float32)
    n = 120
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
    ids = np.arange(n, dtype=np.int64)
    projector = TigreProjector(grid, geo, angles, fdk)
    variable_geo = {**geo, "offDetector_vu_mm": np.column_stack((ids * 0.1, ids * -0.2)).tolist()}
    variable_projector = TigreProjector(grid, variable_geo, angles, fdk)
    selected = ids[[1, 4, 11, 27]]
    subset, subset_angles = variable_projector._geometry_for_views(selected, (0, 0))
    np.testing.assert_array_equal(subset_angles, angles[selected])
    np.testing.assert_allclose(subset.offDetector, np.asarray(variable_geo["offDetector_vu_mm"])[selected])
    folder = output / "phantom"
    folder.mkdir(parents=True, exist_ok=True)
    columns = ("view_id", "phase_id", "cycle_id", "angle_rad")
    with (folder / "views.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows((i, i % 2, i // 2, float(angle)) for i, angle in enumerate(angles))
    with (folder / "test_views.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("view_id", "phase_id", "angle_rad"))
        writer.writerows((i, i % 2, float(angle)) for i, angle in enumerate(angles))
    cylinder_recon = projector.fdk(projector.project(base, ids), ids)
    center = (slice(4, 12), slice(12, 20), slice(12, 20))
    cylinder_mean = float(cylinder_recon[center].mean())
    assert abs(cylinder_mean / 0.012 - 1) <= 0.05, cylinder_mean
    measured, reference = {}, []
    for scan_id, split, offset in (("training", "train", 0), ("validation", "val", -1), ("testing", "test", 1)):
        projections = np.empty((n, *geo["nDetector_vu"]), dtype=np.float32)
        for phase in range(2):
            center_zyx = (8, 14, 19 + offset + phase * 4)
            cz, cy, cx = center_zyx
            volume = base + np.float32(0.045) * ((x-cx)**2 + (y-cy)**2 + (z-cz)**2 <= 2**2)
            selected = ids[ids % 2 == phase]
            projections[selected] = projector.project(volume, selected)
            if scan_id == "training" and phase == 0:
                reconstructed = projector.fdk(projector.project(volume, ids), ids)
                peak = np.asarray(np.unravel_index(np.argmax(reconstructed), shape))
                marker_error = float(np.linalg.norm(peak - center_zyx))
                assert marker_error <= 1, (peak, center_zyx)
            if split == "test":
                target = folder / f"reference_{phase}.npy"
                save_array(target, volume)
                reference.append({"phase_id": phase, "grid": grid, "units": "mm^-1", "path": target.name})
        save_array(folder / f"{scan_id}.npy", projections)
        measured[scan_id] = {"patient_id": scan_id, "scan_id": scan_id, "domain": "phantom", "split": split,
                             "n_phases": 2, "projections": f"{scan_id}.npy", "grid": grid, "geometry": geo,
                             "views": "test_views.csv" if split == "test" else "views.csv"}
    write_json(folder / "train.json", {"scans": [measured["training"], measured["validation"]]})
    write_json(folder / "test.json", {"scans": [measured["testing"]]})
    write_json(folder / "reference.json", {"scans": [{"scan_id": "testing", "phases": reference}]})
    return folder, {"cylinder_roi_mean": cylinder_mean, "cylinder_target": 0.012,
                    "cylinder_relative_error": abs(cylinder_mean / 0.012 - 1), "marker_error_voxels": marker_error}


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_equal(left, right)
    else:
        assert first == second, (first, second)


def run(output):
    folder, physical = phantom(output)

    def cli(*arguments):
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        subprocess.run([sys.executable, "-m", "dpi_moco", *map(str, arguments)], env=env, check=True)

    prepared = output / "prepared"
    cli("prepare", "--manifest", folder / "train.json", "--out", prepared)
    model_paths, checks = {}, {}
    for stage, name in (("restore", "G.pt"), ("register", "R.pt"), ("denoise", "O.pt")):
        root = output / stage
        continuous, resumed = root / "continuous", root / "resumed"
        common = ["train", stage, "--prepared", prepared, "--device", "cpu"]
        options = ["--epochs", 2, "--max-batches", 1, "--deterministic", "--log-every", 1]
        options += (["--restore", model_paths["restore"]] if stage == "register"
                    else ["--batch-size", 2, "--patch-size", 32])
        cli(*common, "--out", continuous, *options)
        cli(*common, "--out", resumed, *options, "--stop-after-epoch", 1)
        ckpt1 = resumed / "checkpoints" / "epoch_0001.pt"
        assert ckpt1.is_file()
        save_array(resumed / "checkpoints" / ".epoch_9999.pt.partial.tmp", np.zeros(1))
        with SummaryWriter(str(resumed / "tensorboard")) as writer:
            writer.add_scalar("progress/epoch", 999, 3)
        cli(*common, "--out", resumed, "--resume")
        model_paths[stage] = resumed / name
        left, right = [load_checkpoint(path / "checkpoints" / "epoch_0002.pt") for path in (continuous, resumed)]
        for key in ("models", "optimizers", "schedulers", "random", "summary", "epoch", "global_step"):
            assert_equal(left[key], right[key])
        events = EventAccumulator(str(resumed / "tensorboard"), size_guidance={"scalars": 0}).Reload()
        steps = [point.step for point in events.Scalars("progress/epoch")]
        assert steps == [1, 2], steps
        assert "validation/input_prediction_target" in events.Tags()["images"]
        rates = [point.value for point in events.Scalars("learning_rate/model")]
        np.testing.assert_allclose(rates, [0.0001 if stage == "register" else 0.001, 0.00001], rtol=1e-6)
        checks[stage] = {"epoch_checkpoints": [1, 2], "global_step": right["global_step"],
                         "state_matches_continuous": True, "tensorboard_steps": steps, "learning_rate": rates}
    predicted = output / "predicted"
    cli("infer", "--manifest", folder / "test.json", "--out", predicted, "--device", "cpu", "--keep-intermediates",
        "--keep-fields", "--motion-qc", "--motion-ablation",
        *(arg for stage, path in model_paths.items() for arg in (f"--{stage}", path)))
    cli("evaluate", "--pred", predicted, "--reference", folder / "reference.json", "--stages", "x", "b", "d", "dpri",
        "d_noisy", "d_from_x", "d_from_b", "d_zero_motion")
    predictions = read_json(predicted / "predictions.json")["scans"][0]["phases"]
    assert [row["phase_id"] for row in predictions] == [0, 1]
    for row in predictions:
        assert list(np.load(predicted / row["path"], mmap_mode="r").shape) == row["grid"]["shape_zyx"]
    write_json(output / "validation.json", {"software": software(torch.device("cuda")), "physics": physical,
               "training": checks, "prediction_phases": len(predictions), "evaluation": str(predicted / "metrics.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Empty validation run directory")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --out directory")
    run(output)
