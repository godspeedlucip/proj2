import hashlib
import json
import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
import SimpleITK as sitk

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


DATASET = {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'heart': 1},
           'file_ending': '.mha'}


def _plans():
    return {
        'dataset_name': 'Dataset001_Paired',
        'plans_name': 'nnUNetPlans',
        'transpose_forward': [2, 0, 1],
        'transpose_backward': [1, 2, 0],
        'image_reader_writer': 'SimpleITKIO',
        'foreground_intensity_properties_per_channel': {
            '0': {'mean': 50.0, 'std': 10.0, 'percentile_00_5': 20.0, 'percentile_99_5': 80.0}
        },
        'configurations': {
            '3d_fullres': {
                'data_identifier': 'nnUNetPlans_3d_fullres',
                'spacing': [1.0, 2.0, 1.0],
                'patch_size': [8, 8, 8],
                'batch_size': 2,
                'use_mask_for_norm': [False],
                'normalization_schemes': ['CTNormalization'],
                'architecture': {
                    'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
                    'arch_kwargs': {'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]]},
                },
                'resampling_fn_data': 'resample_data_or_seg_to_shape',
                'resampling_fn_data_kwargs': {'is_seg': False, 'order': 1},
                'resampling_fn_seg': 'resample_data_or_seg_to_shape',
                'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 0},
            }
        },
    }


def _write_volume(path, array, origin=(0.0, 0.0, 0.0), direction=None):
    volume = sitk.GetImageFromArray(array)
    volume.SetSpacing((1.0, 1.0, 2.0))
    volume.SetOrigin(origin)
    if direction is not None:
        volume.SetDirection(direction)
    sitk.WriteImage(volume, str(path))
    return str(path)


def _write_case(folder, case_id='scan01', patient_id='patient01', values=(40, 60),
                phase5_origin=(0.0, 0.0, 0.0), flip_phase5=False, file_ending='.mha'):
    record = {'case_id': case_id, 'patient_id': patient_id}
    for phase, value, foreground_x in zip(('phase1', 'phase5'), values, (2, 4)):
        image = np.zeros((6, 5, 8), dtype=np.float32)
        image[1:5, 1:4, 1:6] = value
        label = np.zeros_like(image, dtype=np.uint8)
        label[2, 2, foreground_x] = 1
        origin = (0.0, 0.0, 0.0) if phase == 'phase1' else phase5_origin
        direction = None
        if phase == 'phase5' and flip_phase5:
            image = image[..., ::-1].copy()
            label = label[..., ::-1].copy()
            origin = (7.0, 0.0, 0.0)
            direction = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        record[phase] = {
            'image': _write_volume(folder / f'{case_id}_{phase}_ct{file_ending}', image, origin, direction),
            'label': _write_volume(folder / f'{case_id}_{phase}_seg{file_ending}', label, origin, direction),
        }
    return record


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def _prepared_inputs(folder, split):
    records = [
        _write_case(folder, 'train_scan', 'patient_a', (40, 60)),
        _write_case(folder, 'val_scan', 'patient_b', (1000, 1100)),
        _write_case(folder, 'test_scan', 'patient_c', (-1000, -1100)),
    ]
    return (
        _write_json(folder / 'manifest.json', records),
        _write_json(folder / 'splits.json', [{}, split]),
        _write_json(folder / 'plans_input.json', _plans()),
        _write_json(folder / 'dataset_input.json', DATASET),
    )


def test_manifest_resolves_phases_by_name_and_accepts_image_only_records(tmp_path):
    from nnunetv2.preprocessing.paired_tam import read_manifest

    record = _write_case(tmp_path)
    manifest = _write_json(tmp_path / 'manifest.json', [{
        'case_id': 'scan01', 'patient_id': 'patient01',
        'phase5': {'image': 'scan01_phase5_ct.mha'},
        'phase1': {'image': 'scan01_phase1_ct.mha'},
    }])
    resolved = read_manifest(manifest)['scan01']

    assert resolved['phase1']['image'] == record['phase1']['image']
    assert resolved['phase5']['image'] == record['phase5']['image']
    with pytest.raises(ValueError, match='scan01.*phase1.*label'):
        read_manifest(manifest, require_labels=True)


@pytest.mark.parametrize('invalid_pair', ['duplicate_case', 'duplicate_image', 'missing_image'])
def test_manifest_rejects_ambiguous_or_missing_inputs(tmp_path, invalid_pair):
    from nnunetv2.preprocessing.paired_tam import read_manifest

    record = _write_case(tmp_path)
    records = [record]
    if invalid_pair == 'duplicate_case':
        records.append(record)
    elif invalid_pair == 'duplicate_image':
        record['phase5']['image'] = record['phase1']['image']
    else:
        record['phase5']['image'] = str(tmp_path / 'absent.mha')

    with pytest.raises((ValueError, FileNotFoundError), match='scan01'):
        read_manifest(_write_json(tmp_path / 'manifest.json', records), require_labels=True)


def test_common_grid_uses_physical_coordinates_and_shared_image_preprocessing(tmp_path):
    from nnunetv2.preprocessing.paired_tam import preprocess_case

    record = _write_case(tmp_path, flip_phase5=True)
    plans = PlansManager(_plans())
    configuration = plans.get_configuration('3d_fullres')

    data, seg, properties = preprocess_case(record, plans, configuration, DATASET, with_labels=True)
    prediction_data, prediction_seg, _ = preprocess_case(record, plans, configuration, DATASET)

    assert data.shape == seg.shape == (2, 5, 4, 3)
    np.testing.assert_allclose(data[0], -1)
    np.testing.assert_allclose(data[1], 1)
    np.testing.assert_array_equal(np.argwhere(seg[0] == 1), [[1, 1, 1]])
    np.testing.assert_array_equal(np.argwhere(seg[1] == 1), [[3, 1, 1]])
    np.testing.assert_array_equal(prediction_data, data)
    assert prediction_seg is None
    assert properties['spacing'] == (2.0, 1.0, 1.0)
    assert properties['phase_order'] == ['phase1', 'phase5']
    assert properties['shape_before_cropping'] == (8, 6, 5)
    assert properties['bbox_used_for_cropping'] == [[1, 6], [1, 5], [1, 4]]
    assert properties['shape_after_cropping_and_before_resampling'] == (5, 4, 3)
    assert properties['original_geometries']['phase5']['origin'] == (7.0, 0.0, 0.0)
    assert properties['original_geometries']['phase5']['direction'][0] == -1.0
    assert properties['common_geometry']['size'] == (8, 5, 6)
    assert {tuple(point) for point in properties['class_locations'][1]} == {(0, 1, 1, 1), (0, 3, 1, 1)}


def test_case_rejects_uncovered_crop_without_reading_labels(tmp_path):
    from nnunetv2.preprocessing.paired_tam import preprocess_case

    record = _write_case(tmp_path, phase5_origin=(2.0, 0.0, 0.0))
    for phase in ('phase1', 'phase5'):
        Path(record[phase].pop('label')).unlink()
    plans = PlansManager(_plans())

    with pytest.raises(ValueError, match='scan01.*phase5.*field of view'):
        preprocess_case(record, plans, plans.get_configuration('3d_fullres'), DATASET)


def test_case_interpolates_ct_on_an_offset_grid_and_keeps_hard_labels(tmp_path):
    from nnunetv2.preprocessing.paired_tam import preprocess_case

    origin = (0.25, 0.0, 0.0)
    record = _write_case(tmp_path)
    image = np.zeros((6, 5, 7), dtype=np.int16)
    image[1:5, 1:4, 2:5] = (30, 50, 70)
    label = np.zeros_like(image, dtype=np.uint8)
    label[2, 2, 4] = 1
    _write_volume(record['phase5']['image'], image, origin)
    _write_volume(record['phase5']['label'], label, origin)
    plans = PlansManager(_plans())

    data, seg, _ = preprocess_case(record, plans, plans.get_configuration('3d_fullres'), DATASET, with_labels=True)

    assert data[1, 2, 1, 1] == -0.5
    np.testing.assert_array_equal(np.argwhere(seg[1] == 1), [[3, 1, 1]])


def test_preprocess_reads_nifti_from_a_unicode_directory(tmp_path):
    from nnunetv2.preprocessing.paired_tam import preprocess_case

    record = _write_case(tmp_path, file_ending='.nii.gz')
    source = tmp_path / '心脏 CT'
    source.mkdir()
    for phase in ('phase1', 'phase5'):
        for role, path in record[phase].items():
            target = source / Path(path).name
            copyfile(path, target)
            record[phase][role] = str(target)
    plans = PlansManager(_plans())
    dataset = {**DATASET, 'file_ending': '.nii.gz'}

    data, seg, _ = preprocess_case(record, plans, plans.get_configuration('3d_fullres'), dataset, with_labels=True)

    assert data.shape == seg.shape == (2, 5, 4, 3)
    assert np.count_nonzero(seg == 1) == 2


def test_annotated_sampling_locations_exclude_voxels_ignored_in_both_phases(tmp_path):
    from nnunetv2.preprocessing.paired_tam import preprocess_case

    record = _write_case(tmp_path)
    for phase, foreground_x in (('phase1', 2), ('phase5', 4)):
        label = np.full((6, 5, 8), fill_value=2, dtype=np.uint8)
        label[2, 2, foreground_x] = 1
        label[2, 2, 3] = 0
        _write_volume(record[phase]['label'], label)
    plans = PlansManager(_plans())
    dataset = {**DATASET, 'labels': {**DATASET['labels'], 'ignore': 2}}

    _, _, properties = preprocess_case(record, plans, plans.get_configuration('3d_fullres'), dataset, with_labels=True)

    points = properties['class_locations'][(-1, 0, 1)]
    assert {tuple(point) for point in points} == {(0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 1, 1)}


@pytest.mark.parametrize('problem', ['geometry', 'values', 'foreground'])
def test_case_rejects_invalid_label_before_training(tmp_path, problem):
    from nnunetv2.preprocessing.paired_tam import preprocess_case

    record = _write_case(tmp_path)
    label = np.zeros((6, 5, 8), dtype=np.float32)
    origin = (0.1, 0.0, 0.0) if problem == 'geometry' else (0.0, 0.0, 0.0)
    label[2, 2, 0 if problem == 'foreground' else 3] = 1.5 if problem == 'values' else 1
    _write_volume(record['phase5']['label'], label, origin)
    plans = PlansManager(_plans())

    with pytest.raises(ValueError, match='scan01.*phase5.*' + problem):
        preprocess_case(record, plans, plans.get_configuration('3d_fullres'), DATASET, with_labels=True)


def test_case_rejects_foreground_lost_from_source_grid(tmp_path):
    from nnunetv2.preprocessing.paired_tam import preprocess_case

    record = _write_case(tmp_path, phase5_origin=(-1.0, 0.0, 0.0))
    label = np.zeros((6, 5, 8), dtype=np.int16)
    label[2, 2, 0] = 1
    label[2, 2, 4] = 1
    _write_volume(record['phase5']['label'], label, (-1.0, 0.0, 0.0))
    plans = PlansManager(_plans())

    with pytest.raises(ValueError, match='scan01.*phase5.*foreground'):
        preprocess_case(record, plans, plans.get_configuration('3d_fullres'), DATASET, with_labels=True)


def test_prepare_freezes_pooled_train_stats_and_publishes_nnunet_dataset(tmp_path):
    from nnunetv2.preprocessing.paired_tam import prepare_dataset
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy

    split = {'train': ['train_scan'], 'val': ['val_scan'], 'test': ['test_scan']}
    output = tmp_path / 'prepared'
    prepare_dataset(*_prepared_inputs(tmp_path, split), output, fold=1)

    fingerprint = json.loads((output / 'dataset_fingerprint.json').read_text(encoding='utf-8'))
    prepared_plans = json.loads((output / 'plans.json').read_text(encoding='utf-8'))
    prepared_splits = json.loads((output / 'splits_final.json').read_text(encoding='utf-8'))
    stats = fingerprint['foreground_intensity_properties_per_channel']
    assert stats['0']['mean'] == 50
    assert stats['0']['std'] == 10
    assert prepared_plans['foreground_intensity_properties_per_channel'] == stats
    assert prepared_splits[1] == split
    assert fingerprint['fold'] == 1
    assert (output / 'paired_manifest.json').is_file()
    assert (output / 'dataset.json').is_file()

    dataset = nnUNetDatasetNumpy(str(output / 'nnUNetPlans_3d_fullres'))
    assert dataset.identifiers == ['test_scan', 'train_scan', 'val_scan']
    data, seg, _, properties = dataset.load_case('val_scan')
    assert isinstance(data, np.memmap)
    assert isinstance(seg, np.memmap)
    assert data.shape == seg.shape == (2, 5, 4, 3)
    np.testing.assert_allclose(data, 0.99, atol=1e-6)
    assert properties['phase_order'] == ['phase1', 'phase5']


def test_prepared_provenance_rejects_changed_validation_mask(tmp_path):
    from nnunetv2.preprocessing.paired_tam import prepare_dataset, read_manifest, validate_sources

    split = {'train': ['train_scan'], 'val': ['val_scan'], 'test': ['test_scan']}
    output = tmp_path / 'prepared'
    prepare_dataset(*_prepared_inputs(tmp_path, split), output, fold=1)
    records = read_manifest(output / 'paired_manifest.json', require_labels=True)
    fingerprint = json.loads((output / 'dataset_fingerprint.json').read_text(encoding='utf-8'))
    validate_sources(records, fingerprint)

    labels = np.zeros((6, 5, 8), dtype=np.uint8)
    labels[2, 2, 2] = 1
    label_path = Path(records['val_scan']['phase5']['label'])
    original_stat = label_path.stat()
    sha256 = fingerprint['source_files']['val_scan']['phase5']['label']['sha256']
    assert sha256 == hashlib.sha256(label_path.read_bytes()).hexdigest()
    _write_volume(label_path, labels)
    os.utime(label_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000))
    assert label_path.stat().st_size == original_stat.st_size

    with pytest.raises(ValueError, match='val_scan.*phase5.*label.*changed'):
        validate_sources(records, fingerprint)
    assert fingerprint['foreground_intensity_properties_per_channel']['0']['mean'] == 50


@pytest.mark.parametrize('problem', ['patient_overlap', 'missing_case'])
def test_prepare_rejects_split_leakage_or_incomplete_partition(tmp_path, problem):
    from nnunetv2.preprocessing.paired_tam import prepare_dataset

    split = {'train': ['train_scan'], 'val': ['val_scan'], 'test': ['test_scan']}
    if problem == 'missing_case':
        split['test'] = []
    inputs = _prepared_inputs(tmp_path, split)
    if problem == 'patient_overlap':
        records = json.loads(inputs[0].read_text(encoding='utf-8'))
        records[1]['patient_id'] = records[0]['patient_id']
        _write_json(inputs[0], records)

    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='patient_a' if problem == 'patient_overlap' else 'test_scan'):
        prepare_dataset(*inputs, output, fold=1)

    assert not output.exists()


def test_prepare_rejects_case_ids_with_colliding_npy_files(tmp_path):
    from nnunetv2.preprocessing.paired_tam import prepare_dataset

    split = {'train': ['train_scan'], 'val': ['train_scan_seg'], 'test': ['test_scan']}
    inputs = _prepared_inputs(tmp_path, split)
    records = json.loads(inputs[0].read_text(encoding='utf-8'))
    records[1]['case_id'] = 'train_scan_seg'
    _write_json(inputs[0], records)

    with pytest.raises(ValueError, match='train_scan.*train_scan_seg.*filename'):
        prepare_dataset(*inputs, tmp_path / 'prepared', fold=1)


def test_prepare_leaves_no_partial_dataset_when_later_case_has_bad_geometry(tmp_path):
    from nnunetv2.preprocessing.paired_tam import prepare_dataset

    split = {'train': ['train_scan'], 'val': ['val_scan'], 'test': ['test_scan']}
    inputs = _prepared_inputs(tmp_path, split)
    label = np.zeros((6, 5, 8), dtype=np.uint8)
    _write_volume(tmp_path / 'val_scan_phase5_seg.mha', label, origin=(0.0, 1.0, 0.0))
    output = tmp_path / 'prepared'
    output.mkdir()

    with pytest.raises(ValueError, match='val_scan.*phase5.*geometry'):
        prepare_dataset(*inputs, output, fold=1)

    assert not list(output.iterdir())
    assert not list(tmp_path.glob('.paired-tam-*'))
