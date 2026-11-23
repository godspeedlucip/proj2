"""Data preparation for paired phase1/phase5 CT using one 3d_fullres plan."""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, get_bbox_from_mask

from nnunetv2.preprocessing.cropping.cropping import create_nonzero_mask
from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
from nnunetv2.utilities.paired_tam_contract import CONFIGURATION, PHASES, validate_plans
from nnunetv2.utilities.paired_tam_io import read_image
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def _load_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def _save_json(path, value):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _safe_file_stem(value):
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and not value.endswith('.') and not re.search(r'[<>:"/\\|?*\x00-\x1f]', value)
            and not re.fullmatch(r'(CON|PRN|AUX|NUL|(?:COM|LPT)[1-9¹²³])', value.split('.')[0], re.I))


def read_manifest(path, require_labels=False):
    """Resolve a JSON list of explicit pairs, keeping keys in case-id order."""
    path = Path(path).resolve()
    entries = _load_json(path)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'Pair manifest {path} must contain a nonempty list of case records.')
    records, case_ids, images = {}, set(), {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f'Pair manifest {path} must contain objects.')
        case_id, patient_id = entry.get('case_id'), entry.get('patient_id')
        if not _safe_file_stem(case_id):
            raise ValueError(f'Invalid case_id {case_id!r}; supply a safe filename stem.')
        if case_id.casefold() in case_ids:
            raise ValueError(f'Duplicate case_id {case_id!r}.')
        if not isinstance(patient_id, str) or not patient_id or patient_id != patient_id.strip():
            raise ValueError(f'{case_id}: patient_id must be a nonempty string without surrounding whitespace.')
        case_ids.add(case_id.casefold())
        record = {'case_id': case_id, 'patient_id': patient_id}
        for phase in PHASES:
            phase_entry = entry.get(phase)
            if not isinstance(phase_entry, dict):
                raise ValueError(f'{case_id} {phase}: missing phase object.')
            record[phase] = {}
            for role in ('image', 'label'):
                value = phase_entry.get(role)
                required = role == 'image' or require_labels
                if value is None and not required:
                    continue
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'{case_id} {phase}: missing {role} path.')
                resolved = (path.parent / value).resolve()
                if required and not resolved.is_file():
                    raise FileNotFoundError(f'{case_id} {phase}: {role} file does not exist: {resolved}')
                if role == 'image':
                    if resolved in images:
                        raise ValueError(f'{case_id} {phase}: image file is already used by {images[resolved]}.')
                    images[resolved] = f'{case_id} {phase}'
                record[phase][role] = str(resolved)
        records[case_id] = record
    return records


def validate_split(splits, records, fold):
    """Return a complete case partition for the selected patient-level fold."""
    if (type(fold) is not int or fold < 0 or not isinstance(splits, list) or fold >= len(splits)
            or not isinstance(splits[fold], dict)):
        raise ValueError(f'Fold {fold!r} is missing from the splits list.')
    split, cases, patients = {}, {}, {}
    for group in ('train', 'val', 'test'):
        members = splits[fold].get(group)
        if not isinstance(members, list) or any(not isinstance(case_id, str) for case_id in members):
            raise ValueError(f'Fold {fold}: {group} must be a list of case IDs.')
        for case_id in members:
            if case_id not in records:
                raise ValueError(f'Fold {fold}: unknown case {case_id!r} in {group}.')
            if case_id in cases:
                raise ValueError(f'Fold {fold}: case {case_id!r} occurs more than once across the splits.')
            cases[case_id] = group
            patient_id = records[case_id]['patient_id']
            if patients.setdefault(patient_id, group) != group:
                raise ValueError(f'Fold {fold}: patient {patient_id!r} spans {patients[patient_id]} and {group}.')
        split[group] = members
    if missing := sorted(set(records) - cases.keys()):
        raise ValueError(f'Fold {fold}: unassigned cases {missing}.')
    if any(not members for members in split.values()):
        raise ValueError(f'Fold {fold}: train, val and test must each contain at least one case.')
    return split


def _source_paths(records):
    for case_id, record in records.items():
        for phase in PHASES:
            for role in ('image', 'label'):
                yield case_id, phase, role, Path(record[phase][role])


def _file_stat(stat):
    return {name: getattr(stat, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def _source_signature(path, context, previous=None, *, require_same_stat=False):
    try:
        path_stat = _file_stat(path.stat())
        with path.open('rb') as stream:
            # Some Windows builds report creation time for path stat and change time for fstat.
            handle_stat = _file_stat(os.fstat(stream.fileno()))
            signature = {'path': path_stat, 'handle': handle_stat}
            if previous is not None and signature == previous['stat']:
                if path_stat != _file_stat(path.stat()):
                    raise ValueError(f'{context}: source changed during verification: {path}')
                return previous
            if require_same_stat:
                raise ValueError(f'{context}: source changed during preparation: {path}')
            digest = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
            if handle_stat != _file_stat(os.fstat(stream.fileno())) or path_stat != _file_stat(path.stat()):
                raise ValueError(f'{context}: source changed while hashing: {path}')
    except OSError as error:
        raise ValueError(f'{context}: unable to verify source file {path}: {error}') from error
    return {'sha256': digest.hexdigest(), 'stat': signature}


def _capture_sources(records):
    sources = {}
    for case_id, phase, role, path in _source_paths(records):
        context = f'{case_id} {phase} {role}'
        sources.setdefault(case_id, {}).setdefault(phase, {})[role] = _source_signature(path, context)
    return sources


def _validate_sources(records, sources, *, require_same_stat=False):
    if not isinstance(sources, dict) or set(sources) != set(records):
        raise ValueError('Prepared fingerprint must identify the original source files for every case.')
    for case_id, phase, role, path in _source_paths(records):
        context = f'{case_id} {phase} {role}'
        try:
            previous = sources[case_id][phase][role]
            valid_signature = isinstance(previous['stat'], dict) and re.fullmatch('[0-9a-f]{64}', previous['sha256'])
        except (KeyError, TypeError):
            valid_signature = False
        if not valid_signature:
            raise ValueError(f'{context}: prepared fingerprint is missing a valid source signature.')
        current = _source_signature(path, context, previous, require_same_stat=require_same_stat)
        if current['sha256'] != previous['sha256']:
            raise ValueError(f'{context}: source content changed after preparation: {path}')


def validate_sources(records, fingerprint):
    """Check that the manifest's original files still match the prepared fold."""
    _validate_sources(records, fingerprint.get('source_files'))


def _geometry(image):
    return {'size': image.GetSize(), 'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(), 'direction': image.GetDirection()}


def _read_volume(path, context, role):
    try:
        image = read_image(path)
    except (OSError, RuntimeError) as error:
        raise ValueError(f'{context}: could not read {role} volume {path}: {error}') from error
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f'{context}: {role} must be a scalar 3D volume.')
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing())
    if (not all(image.GetSize()) or not np.all(np.isfinite(spacing) & (spacing > 0))
            or not np.all(np.isfinite(image.GetOrigin()))
            or not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-4)):
        raise ValueError(f'{context}: invalid {role} geometry (size, spacing, origin or direction).')
    if role == 'image':
        image = sitk.Cast(image, sitk.sitkFloat32)
        if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
            raise ValueError(f'{context}: image contains non-finite CT intensities.')
    return image


def _read_label(path, image, label_manager, context):
    label = _read_volume(path, context, 'label')
    if not image.IsSameImageGeometryAs(label):
        raise ValueError(f'{context}: label geometry differs from its image.')
    allowed = set(label_manager.all_labels)
    if label_manager.has_ignore_label:
        allowed.add(label_manager.ignore_label)
    values = np.unique(sitk.GetArrayViewFromImage(label))
    if unknown := set(values.tolist()) - allowed:
        raise ValueError(f'{context}: label values {unknown} are outside the dataset.json label map.')
    return sitk.Cast(label, sitk.sitkInt8 if max(allowed) <= 127 else sitk.sitkInt16)


def _label_manager(plans_manager, dataset_json):
    label_manager = plans_manager.get_label_manager(dataset_json)
    labels = label_manager.all_labels
    if labels != list(range(len(labels))) or max(labels + [label_manager.ignore_label or 0]) > 32767:
        raise ValueError('Paired TAM requires consecutive class IDs starting at zero and fitting in int16.')
    if any(type(value) is not int for value in dataset_json['labels'].values()):
        raise ValueError('Paired TAM requires integer values in the dataset.json label map.')
    return label_manager


def _resample_to_grid(image, reference, interpolator):
    if image.IsSameImageGeometryAs(reference):
        return image
    return sitk.Resample(image, reference, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0)


def _foreground(labels, ignore_label):
    mask = labels > 0
    if ignore_label is not None:
        mask &= labels != ignore_label
    return mask


def _check_foreground_coverage(label, aligned, reference, crop, label_manager, context):
    native_fg = _foreground(sitk.GetArrayViewFromImage(label), label_manager.ignore_label)
    if label.IsSameImageGeometryAs(reference):
        native_covered = np.count_nonzero(native_fg[crop]) == np.count_nonzero(native_fg)
    else:
        # Check in the source grid too, so labels outside phase1's FOV cannot disappear unnoticed.
        crop_mask = np.zeros(reference.GetSize()[::-1], dtype=np.uint8)
        crop_mask[crop] = 1
        mask_image = sitk.GetImageFromArray(crop_mask)
        mask_image.CopyInformation(reference)
        source_mask = _resample_to_grid(mask_image, label, sitk.sitkNearestNeighbor)
        native_covered = sitk.GetArrayViewFromImage(source_mask)[native_fg].all()
    aligned_fg = _foreground(aligned, label_manager.ignore_label)
    if not native_covered or np.count_nonzero(aligned_fg[crop]) != np.count_nonzero(aligned_fg):
        raise ValueError(f'{context}: foreground label voxels fall outside the common image crop.')


def _class_locations(seg, label_manager):
    locations = {}
    for label in label_manager.foreground_labels:
        union = np.any(seg == label, axis=0, keepdims=True)
        locations[label] = DefaultPreprocessor._sample_foreground_locations(union, [True])[True]
    if label_manager.has_ignore_label:
        annotated = np.any(seg != label_manager.ignore_label, axis=0, keepdims=True)
        locations[(-1, *label_manager.all_labels)] = DefaultPreprocessor._sample_foreground_locations(
            annotated, [True]
        )[True]
    return locations


def _preprocess_case(record, plans_manager, config, label_manager):
    case_id = record['case_id']
    images = [_read_volume(record[phase]['image'], f'{case_id} {phase}', 'image') for phase in PHASES]
    reference = images[0]
    common_geometry = _geometry(reference)
    data = np.empty((len(PHASES), *reference.GetSize()[::-1]), dtype=np.float32)
    for index, image in enumerate(images):
        aligned = _resample_to_grid(image, reference, sitk.sitkLinear)
        data[index] = sitk.GetArrayViewFromImage(aligned)

    # Determine the crop from the two images before looking at either label.
    union = create_nonzero_mask(data)
    if not union.any():
        raise ValueError(f'{case_id} {"/".join(PHASES)}: images have an empty common-grid intensity support.')
    crop = bounding_box_to_slice(get_bbox_from_mask(union))
    del union
    for phase, image in zip(PHASES, images):
        # The coverage mask comes from the source grid, independent of CT voxel values.
        source_fov = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
        source_fov.CopyInformation(image)
        common_fov = _resample_to_grid(source_fov, reference, sitk.sitkNearestNeighbor)
        if not sitk.GetArrayViewFromImage(common_fov)[crop].all():
            raise ValueError(f'{case_id} {phase}: field of view does not cover the common image crop.')

    transpose = plans_manager.transpose_forward
    bbox = [[crop[i].start, crop[i].stop] for i in transpose]
    original_spacing = common_geometry['spacing'][::-1]
    current_spacing = [original_spacing[i] for i in transpose]
    axes = (0, *(i + 1 for i in transpose))
    data = data[(slice(None), *crop)].transpose(axes).copy()
    properties = {
        'case_id': case_id,
        'patient_id': record['patient_id'],
        'phase_order': list(PHASES),
        'original_geometries': {phase: _geometry(image) for phase, image in zip(PHASES, images)},
        'common_geometry': common_geometry,
        'spacing': original_spacing,
        'shape_before_cropping': tuple(common_geometry['size'][::-1][i] for i in transpose),
        'bbox_used_for_cropping': bbox,
        'shape_after_cropping_and_before_resampling': data.shape[1:],
    }
    seg = None
    if label_manager is not None:
        labels = []
        for phase, image in zip(PHASES, images):
            context = f'{case_id} {phase}'
            label = _read_label(record[phase]['label'], image, label_manager, context)
            aligned = _resample_to_grid(label, reference, sitk.sitkNearestNeighbor)
            aligned_array = sitk.GetArrayViewFromImage(aligned)
            _check_foreground_coverage(label, aligned_array, reference, crop, label_manager, context)
            labels.append(aligned_array[crop].copy())
        seg = np.stack(labels).transpose(axes)

    try:
        stats = plans_manager.foreground_intensity_properties_per_channel['0']
        mean, std, lower, upper = (stats[key] for key in ('mean', 'std', 'percentile_00_5', 'percentile_99_5'))
        valid_stats = np.isfinite([mean, std, lower, upper]).all() and std >= 0 and lower <= upper
    except (KeyError, TypeError):
        valid_stats = False
    if not valid_stats:
        raise ValueError('Paired TAM requires frozen finite training CT intensity statistics in plans.')
    normalizer = CTNormalization(use_mask_for_norm=config.use_mask_for_norm[0], intensityproperties=stats)
    for frame in data:
        normalizer.run(frame)

    target_spacing = config.spacing
    new_shape = np.maximum(compute_new_shape(data.shape[1:], current_spacing, target_spacing), 1)
    data = np.asarray(config.resampling_fn_data(data, new_shape, current_spacing, target_spacing), dtype=np.float32)
    if seg is not None:
        seg = np.asarray(config.resampling_fn_seg(seg, new_shape, current_spacing, target_spacing), dtype=seg.dtype)
        properties['class_locations'] = _class_locations(seg, label_manager)
    return data, seg, properties


def preprocess_case(record, plans_manager, configuration_manager, dataset_json, with_labels=False):
    """Apply the same plans-driven paired image transform in training and prediction."""
    validate_plans(plans_manager, configuration_manager, dataset_json)
    label_manager = _label_manager(plans_manager, dataset_json) if with_labels else None
    return _preprocess_case(record, plans_manager, configuration_manager, label_manager)


def _estimate_intensities(records, train_cases, label_manager):
    rng = np.random.default_rng(1234)
    per_phase_limit = max(1, min(10_000, 10_000_000 // (len(train_cases) * len(PHASES))))
    samples = []
    for case_id in sorted(train_cases):
        for phase in PHASES:
            record = records[case_id][phase]
            context = f'{case_id} {phase}'
            image = _read_volume(record['image'], context, 'image')
            label = _read_label(record['label'], image, label_manager, context)
            foreground = _foreground(sitk.GetArrayViewFromImage(label), label_manager.ignore_label)
            values = sitk.GetArrayViewFromImage(image)[foreground]
            if len(values) > per_phase_limit:
                values = rng.choice(values, per_phase_limit, replace=False)
            if len(values):
                samples.append(values)
    if not samples:
        raise ValueError('Cannot estimate paired CT intensity statistics: no annotated training foreground.')
    sample = np.concatenate(samples).astype(np.float64)
    lower, median, upper = np.percentile(sample, (0.5, 50, 99.5))
    return {'0': {'mean': float(sample.mean()), 'std': float(sample.std()), 'median': float(median),
                  'min': float(sample.min()), 'max': float(sample.max()),
                  'percentile_00_5': float(lower), 'percentile_99_5': float(upper)}}


def prepare_dataset(manifest_path, splits_path, plans_path, dataset_json_path, output_directory, fold=0):
    """Prepare one fold and publish its files together in an independent output directory."""
    records = read_manifest(manifest_path, require_labels=True)
    split = validate_split(_load_json(splits_path), records, fold)
    case_ids = {case_id.casefold(): case_id for case_id in records}
    for case_id in records:
        if other := case_ids.get(f'{case_id.casefold()}_seg'):
            raise ValueError(f'Case IDs {case_id!r} and {other!r} produce the same NumPy filename.')
    dataset_json = _load_json(dataset_json_path)
    plans_manager = PlansManager(_load_json(plans_path))
    config = plans_manager.get_configuration(CONFIGURATION)
    validate_plans(plans_manager, config, dataset_json)
    label_manager = _label_manager(plans_manager, dataset_json)
    if not _safe_file_stem(config.data_identifier):
        raise ValueError('The 3d_fullres data_identifier must be a safe directory name.')

    output = Path(output_directory).absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError(f'Choose an empty paired preprocessing directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = _capture_sources(records)
    stats = _estimate_intensities(records, split['train'], label_manager)
    plans_manager.plans['foreground_intensity_properties_per_channel'] = stats
    training_cases, spacings, shapes_after_crop, crop_ratios = set(split['train']), [], [], []
    with TemporaryDirectory(prefix='.paired-tam-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        data_directory = stage / config.data_identifier
        data_directory.mkdir(parents=True)
        for case_id, record in sorted(records.items()):
            data, seg, properties = _preprocess_case(record, plans_manager, config, label_manager)
            nnUNetDatasetNumpy.save_case(data, seg, properties, str(data_directory / case_id))
            # Publish memory-map files with the compressed source so concurrent runs never unpack into shared data.
            np.save(data_directory / f'{case_id}.npy', data, allow_pickle=False)
            np.save(data_directory / f'{case_id}_seg.npy', seg, allow_pickle=False)
            if case_id in training_cases:
                spacings.append(properties['spacing'])
                shape = properties['shape_after_cropping_and_before_resampling']
                shapes_after_crop.append([shape[i] for i in plans_manager.transpose_backward])
                crop_ratios.append(np.prod(shape) / np.prod(properties['shape_before_cropping']))
        _validate_sources(records, sources, require_same_stat=True)
        fingerprint = {
            'fold': fold,
            'phase_order': list(PHASES),
            'source_files': sources,
            'foreground_intensity_properties_per_channel': stats,
            'spacings': spacings,
            'shapes_after_crop': shapes_after_crop,
            'median_relative_size_after_cropping': float(np.median(crop_ratios)),
        }
        _save_json(stage / 'dataset_fingerprint.json', fingerprint)
        _save_json(stage / 'dataset.json', dataset_json)
        _save_json(stage / 'plans.json', plans_manager.plans)
        _save_json(stage / 'paired_manifest.json', list(records.values()))
        _save_json(stage / 'splits_final.json', [None] * fold + [split])
        if output.exists():
            output.rmdir()
        stage.rename(output)
