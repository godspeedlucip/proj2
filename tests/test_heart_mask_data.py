import numpy as np
import pytest

from utils.read_2d_npz_wy import npz_reader_2d_data_with_prior


def sample_file(tmp_path, name="sample", *, mask=None, quality=None):
    fields = {
        "input": np.ones((4, 4), dtype=np.float32),
        "label": np.full((4, 4), 2, dtype=np.float32),
        "prior": np.full((4, 4), 3, dtype=np.float32),
    }
    if mask is not None:
        fields["mask"] = mask
    if quality is not None:
        fields["mask_quality"] = (
            quality if isinstance(quality, np.ndarray) else np.array(quality, dtype=np.int8)
        )
    path = tmp_path / f"{name}.npz"
    np.savez(path, **fields)
    return path


def list_file(tmp_path, sample, *, trailing_newline=True):
    path = tmp_path / "samples.txt"
    path.write_text(str(sample) + ("\n" if trailing_newline else ""), encoding="utf-8")
    return path


def test_legacy_reader_loads_last_path_without_newline(tmp_path):
    sample = sample_file(tmp_path)
    dataset = npz_reader_2d_data_with_prior(str(list_file(tmp_path, sample, trailing_newline=False)))
    item = dataset[0]
    assert set(item) == {"input", "label", "prior"}
    assert item["input"].shape == (1, 4, 4)


def test_masked_reader_treats_missing_quality_as_unknown(tmp_path):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    sample = sample_file(tmp_path, mask=mask)
    dataset = npz_reader_2d_data_with_prior(
        str(list_file(tmp_path, sample)), training=False, require_mask=True
    )
    item = dataset[0]
    assert set(item) == {"input", "label", "prior", "mask", "mask_quality"}
    assert item["mask"].shape == (1, 4, 4)
    assert item["mask"].sum().item() == 4
    assert item["mask_quality"].item() == -1
    assert item["input"].mean().item() == 50
    assert item["label"].mean().item() == 100
    assert item["prior"].mean().item() == 150
    assert dataset[0]["input"].equal(item["input"])


@pytest.mark.parametrize("quality", [0, 1])
def test_masked_reader_preserves_known_quality_and_binary_mask(tmp_path, quality):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    sample = sample_file(tmp_path, mask=mask, quality=quality)
    dataset = npz_reader_2d_data_with_prior(
        str(list_file(tmp_path, sample)), training=True, require_mask=True
    )
    item = dataset[0]
    assert item["mask_quality"].item() == quality
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.ones((4, 4), dtype=np.uint8),
        np.ones((3, 4), dtype=np.uint8),
        np.full((4, 4), 2, dtype=np.uint8),
        np.full((4, 4), np.nan, dtype=np.float32),
    ],
)
def test_masked_reader_rejects_bad_mask_with_path(tmp_path, mask):
    sample = sample_file(tmp_path, mask=mask)
    dataset = npz_reader_2d_data_with_prior(
        str(list_file(tmp_path, sample)), require_mask=True
    )
    with pytest.raises(ValueError, match="sample.npz"):
        dataset[0]


def test_masked_reader_rejects_missing_mask_with_path(tmp_path):
    sample = sample_file(tmp_path)
    dataset = npz_reader_2d_data_with_prior(
        str(list_file(tmp_path, sample)), require_mask=True
    )
    with pytest.raises(ValueError, match="sample.npz"):
        dataset[0]


@pytest.mark.parametrize("quality", [np.array(1.0), np.array([1]), np.array(2)])
def test_masked_reader_rejects_non_integer_or_invalid_quality(tmp_path, quality):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    sample = sample_file(tmp_path, mask=mask, quality=quality)
    dataset = npz_reader_2d_data_with_prior(
        str(list_file(tmp_path, sample)), require_mask=True
    )
    with pytest.raises(ValueError, match="sample.npz"):
        dataset[0]


def test_mask_quality_can_come_from_two_column_masked_list(tmp_path):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    sample = sample_file(tmp_path, mask=mask)
    list_path = tmp_path / "masked.tsv"
    list_path.write_text(f"{sample}\t1\n", encoding="utf-8")
    dataset = npz_reader_2d_data_with_prior(str(list_path), require_mask=True)
    assert dataset[0]["mask_quality"].item() == 1


def test_conflicting_npz_and_list_quality_fails_with_sample_path(tmp_path):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    sample = sample_file(tmp_path, mask=mask, quality=0)
    list_path = tmp_path / "masked.tsv"
    list_path.write_text(f"{sample}\t1\n", encoding="utf-8")
    dataset = npz_reader_2d_data_with_prior(str(list_path), require_mask=True)
    with pytest.raises(ValueError, match="sample.npz"):
        dataset[0]


def test_validation_reader_exposes_case_id_from_phase_slice_name(tmp_path):
    sample = sample_file(tmp_path, name="case17_p2_slice8")
    dataset = npz_reader_2d_data_with_prior(
        str(list_file(tmp_path, sample)), training=False, return_case_id=True
    )
    assert dataset[0]["case_id"] == "case17"


def test_validation_reader_accepts_explicit_case_id_for_other_names(tmp_path):
    sample = sample_file(tmp_path)
    list_path = tmp_path / "valid.tsv"
    list_path.write_text(f"{sample}\t\tpatient42\n", encoding="utf-8")
    dataset = npz_reader_2d_data_with_prior(
        str(list_path), training=False, return_case_id=True
    )
    assert dataset[0]["case_id"] == "patient42"


def test_validation_reader_requires_case_id_when_name_is_not_parseable(tmp_path):
    sample = sample_file(tmp_path)
    dataset = npz_reader_2d_data_with_prior(
        str(list_file(tmp_path, sample)), training=False, return_case_id=True
    )
    with pytest.raises(ValueError, match="case_id"):
        dataset[0]
