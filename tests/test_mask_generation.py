"""Segmentation masks retain classes and align with the GT grid."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from generate_round1_contrast_gt import make_phase_gt
from generate_round1_contrast_mask import make_phase_mask, process_series, source_masks
from generate_round1_contrast_recon import SeriesSettings


def _settings(*, wide: bool = False) -> SeriesSettings:
    return SeriesSettings(
        voxel_spacing_multiplier=1, n_detector_rows=3, n_detector_cols=6,
        d_detector_row_mm=1, d_detector_col_mm=1, dso_mm=1, dsd_mm=1,
        fov_output_pixels=4, fov_source_pixels=6 if wide else 0,
        fov_mask_shrink_pixels=0, padding_row_pixels=0, padding_col_pixels=0,
    )


def test_mask_preserves_class_ids_on_wide_gt_grid():
    source = np.zeros((3, 8, 8), np.int16)
    source[:, 2:4, 2:4] = 2
    source[:, 4:6, 4:6] = 5
    settings = _settings(wide=True)

    mask, spacing = make_phase_mask(source, (1, 1, 1), settings)
    gt, gt_spacing = make_phase_gt(
        np.zeros(source.shape, np.float32), (1, 1, 1), settings,
        np.poly1d([1, 0]), 0, 0,
    )

    assert mask.shape == gt.shape == (3, 4, 4)
    assert spacing == gt_spacing == pytest.approx((1, 1.5, 1.5))
    assert set(np.unique(mask)) == {0, 2, 5}


def test_only_annotated_phase_is_saved(tmp_path: Path):
    series = tmp_path / "Heart_with_contrast_back"
    series.mkdir()
    ct_paths = {}
    for phase in (1, 2):
        path = series / f"IM{phase:05d}_ct.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((3, 8, 8), np.float32), np.eye(4)), str(path))
        ct_paths[phase] = path
    source_mask = np.zeros((3, 8, 8), np.int16)
    source_mask[:, 3:5, 3:5] = 2
    nib.save(nib.Nifti1Image(source_mask, np.eye(4)), str(series / "IM00001_mask.nii.gz"))

    assert tuple(source_masks(series, ct_paths)) == (1,)
    output = tmp_path / "out"
    result = process_series(series, ct_paths, output, _settings())

    assert result["status"] == "complete"
    assert tuple(result["phases"]) == ("1",)
    assert (output / "phase_01_mask.tif").is_file()
    assert (output / "phase_01_mask.nii.gz").is_file()
    assert not (output / "phase_02_mask.nii.gz").exists()
