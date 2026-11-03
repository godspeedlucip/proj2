"""Observable contracts of the uniform-cycle validation project."""

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uniform_20_cycles import uniform_recon as recon
from uniform_20_cycles.uniform_cycle import prepare_schedule
from uniform_20_cycles.utils.phase_mapping import build_projection_phase_mapping


def test_600_views_have_20_even_cycles_and_no_missing_or_duplicate_index(tmp_path: Path):
    cycle_csv, mapping_csv = prepare_schedule(tmp_path)
    rows = list(csv.DictReader(mapping_csv.open(encoding="utf-8", newline="")))
    assert len(rows) == 600
    assert [int(row["projection_index"]) for row in rows] == list(range(600))
    assert [int(row["cycle"]) for row in rows[::30]] == list(range(1, 21))
    assert [int(row["phase_label"]) for row in (rows[0], rows[15], rows[30])] == [1, 10, 1]
    assert [float(row["angle_deg"]) for row in (rows[0], rows[1], rows[599])] == [0, 0.5, 299.5]
    assert len(list(csv.DictReader(cycle_csv.open(encoding="utf-8", newline="")))) == 20
    assert len(build_projection_phase_mapping(cycle_csv, 10, min_phase=1, max_phase=10)) == 600
    indices = recon.phase_indices(cycle_csv, tuple(range(1, 11)))
    assert sorted(int(i) for group in indices.values() for i in group) == list(range(600))
    assert all(int(row["phase_label"]) == phase for phase, group in indices.items()
               for index in group for row in (rows[int(index)],))


def test_non_integral_or_odd_cycle_width_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="divisible"):
        prepare_schedule(tmp_path, n_projections=601, n_cycles=20)
    with pytest.raises(ValueError, match="even"):
        prepare_schedule(tmp_path, n_projections=60, n_cycles=4)


def test_simulation_saves_each_projection_in_acquisition_order(tmp_path: Path, monkeypatch):
    series = tmp_path / "Heart_with_contrast_back"
    paths = {1: series / "IM00001_ct.nii.gz", 2: series / "IM00002_ct.nii.gz"}
    header = SimpleNamespace(get_zooms=lambda: (1.0, 1.0, 1.0))
    monkeypatch.setattr(recon.MedicalImageTool, "read", lambda p: SimpleNamespace(
        data=np.full((2, 2, 2), 1 if p == paths[1] else 2, np.float32),
        metadata={"header": header},
    ))
    monkeypatch.setattr(recon.TigreTool, "set_geometry", lambda self, **kw: SimpleNamespace(
        offOrigin=np.zeros(3, np.float32)))
    monkeypatch.setattr(recon.TigreTool, "set_angles", lambda self, *a: np.arange(4))
    monkeypatch.setattr(recon, "calibrated_volume", lambda hu, *a: hu)
    monkeypatch.setattr(recon.TigreTool, "project_tigre", lambda vol, geo, angles:
                        np.full((len(angles), 2, 2), vol[0, 0, 0], np.float32))
    monkeypatch.setattr(recon, "reconstruct_prior", lambda p, *a, **kw:
                        (np.zeros((1, 2, 2), np.float32), (1.0, 1.0, 1.0)))
    out = tmp_path / "output"
    recon.process_series(
        series, paths, out, {1: np.array([0, 2]), 2: np.array([1, 3])},
        np.poly1d([1]), 0.0, 0.0,
        settings=recon.SeriesSettings(n_projections=4, n_detector_rows=2,
            n_detector_cols=2, padding_row_pixels=0, padding_col_pixels=0,
            fov_output_pixels=2),
    )
    saved = np.load(out / "projections.npy", mmap_mode="r")
    assert saved.shape == (4, 2, 2)
    assert saved[:, 0, 0].tolist() == [1, 2, 1, 2]


def test_prepare_command_creates_reusable_schedule(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(project / "run.py"), "prepare", "--schedule-dir", str(tmp_path)],
        cwd=project, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "cycle.csv").is_file()
    assert (tmp_path / "projection_phase_mapping.csv").is_file()


def test_list_resolves_relative_input_paths_from_callers_directory(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    series = tmp_path / "data" / "Test" / "Heart_with_contrast_back"
    series.mkdir(parents=True)
    for phase in range(1, 11):
        (series / f"IM{phase:05d}_ct.nii.gz").touch()
    template = (project / "settings" / "round6_contrast_recon_settings.csv").read_text(
        encoding="utf-8").splitlines()
    settings = tmp_path / "settings.csv"
    settings.write_text(template[0] + "\n" + template[1].replace(
        "Control_1_7569/Heart_with_contrast_back", "Test/Heart_with_contrast_back") + "\n",
        encoding="utf-8",
    )
    result = subprocess.run([
        sys.executable, str(project / "run.py"), "list", "--round-root", "data",
        "--settings-csv", "settings.csv", "--schedule-dir", "schedule",
        "--series", "Test/Heart_with_contrast_back",
    ], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
