"""Output validation catches missing or mismatched simulation artifacts."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from uniform_20_cycles.validate import validate_result


def _fixture(tmp_path: Path, *, bad_gt_spacing: bool = False):
    schedule = tmp_path / "mapping.csv"
    with schedule.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("projection_index", "cycle", "phase_label", "angle_deg"))
        writer.writerows(((0, 1, 1, 0), (1, 1, 2, 75), (2, 1, 2, 150), (3, 1, 1, 225)))
    recon = tmp_path / "recon"
    gt = tmp_path / "gt"
    recon.mkdir()
    gt.mkdir()
    np.save(recon / "projections.npy", np.ones((4, 2, 2), np.float32))
    for directory, stems in ((recon, ("prior_image", "phase_01_reconstruction", "phase_02_reconstruction")),
                             (gt, ("phase_01_gt", "phase_02_gt"))):
        for stem in stems:
            data = np.ones((2, 2, 2), np.float32)
            tifffile.imwrite(directory / f"{stem}.tif", data)
            spacing = 2 if bad_gt_spacing and stem == "phase_02_gt" else 1
            nib.save(nib.Nifti1Image(data, np.diag([spacing, 1, 1, 1])),
                     directory / f"{stem}.nii.gz")
    return schedule, recon, gt


def test_validation_reports_complete_small_series(tmp_path: Path):
    mapping, recon, gt = _fixture(tmp_path)
    result = validate_result(mapping, recon, gt, n_projections=4, n_cycles=1,
                             phases=(1, 2))
    assert result["projection_shape"] == [4, 2, 2]
    assert result["phase_view_counts"] == {"1": 2, "2": 2}
    assert result["volume_shape"] == [2, 2, 2]


def test_validation_rejects_mismatched_gt_grid(tmp_path: Path):
    mapping, recon, gt = _fixture(tmp_path, bad_gt_spacing=True)
    with pytest.raises(ValueError, match="spacing"):
        validate_result(mapping, recon, gt, n_projections=4, n_cycles=1,
                        phases=(1, 2))


def test_validation_command_writes_report(tmp_path: Path):
    mapping, recon, gt = _fixture(tmp_path)
    project = Path(__file__).resolve().parents[1]
    report = tmp_path / "report.json"
    completed = subprocess.run([
        sys.executable, str(project / "validate.py"),
        "--mapping-csv", str(mapping), "--recon-dir", str(recon),
        "--gt-dir", str(gt), "--n-projections", "4", "--n-cycles", "1",
        "--n-phases", "2", "--report", str(report),
    ], cwd=project, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "passed"
