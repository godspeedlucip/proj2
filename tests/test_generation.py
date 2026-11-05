"""Contracts for the standalone get_data branch."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import generate_round1_contrast_gt as gt
from generate_round1_contrast_recon import SeriesSettings, calibrated_volume
from utils.tool import TigreTool


def test_calibration_excludes_padding_without_air_threshold():
    hu = np.array([[[-2048.0, -1000.0, 0.0]]], np.float32)
    result = calibrated_volume(
        hu, np.poly1d([1.0, 0.01]), air_threshold=0.5,
        air_value=0.7, apply_air_threshold=False,
    )
    assert result[0, 0].tolist() == pytest.approx([0.0, 0.01, 0.03])


def test_each_phase_gt_reprojects_all_scan_angles(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    paths = {}
    for phase, hu_value in ((1, 0.0), (2, 1000.0)):
        path = source / f"IM{phase:05d}_ct.nii.gz"
        data = np.full((3, 6, 6), hu_value, np.float32)
        data[:, 3, 3] = -2048.0
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        paths[phase] = path

    settings = SeriesSettings(
        voxel_spacing_multiplier=1, n_detector_rows=3, n_detector_cols=2,
        d_detector_row_mm=1, d_detector_col_mm=1, dso_mm=1, dsd_mm=1,
        n_projections=4, fov_output_pixels=4, fov_mask_shrink_pixels=0,
        padding_row_pixels=0, padding_col_pixels=0,
    )
    projected = []
    reconstructed = []

    def fake_project(volume, geometry, angles):
        projected.append((float(volume[0, 0, 0]), angles.tolist()))
        return np.full((len(angles), 3, 2), volume[0, 0, 0], np.float32)

    def fake_reconstruct(projections, geometry, angles, tool, **options):
        reconstructed.append((projections.shape, angles.tolist(), options))
        return np.full((3, 4, 4), projections[0, 0, 0] * 10, np.float32), (1, 0.5, 0.5)

    monkeypatch.setattr(TigreTool, "set_geometry", lambda self, **kwargs: SimpleNamespace(offOrigin=None))
    monkeypatch.setattr(TigreTool, "set_angles", lambda self, *args: np.arange(4, dtype=np.float32))
    monkeypatch.setattr(TigreTool, "project_tigre", fake_project)
    monkeypatch.setattr(gt, "reconstruct_prior", fake_reconstruct)

    output = tmp_path / "gt"
    gt.process_series(source, paths, output, np.poly1d([1.0, 0.0]), 0.0, 0.0, settings)

    assert [item[0] for item in projected] == pytest.approx([0.02, 0.04])
    assert [item[1] for item in projected] == [[0, 1, 2, 3]] * 2
    assert [item[0] for item in reconstructed] == [(4, 3, 2)] * 2
    assert [item[1] for item in reconstructed] == [[0, 1, 2, 3]] * 2
    image = np.asarray(nib.load(str(output / "phase_01_gt.nii.gz")).dataobj)
    mask = np.asarray(nib.load(str(output / "phase_01_gt_valid_mask.nii.gz")).dataobj)
    assert np.max(image) == pytest.approx(0.2)
    assert set(np.unique(mask)) == {0, 1}
    assert np.all(image[mask == 0] == 0)
