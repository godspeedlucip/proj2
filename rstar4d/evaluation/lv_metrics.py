"""Left ventricular boundary and ED/ES volume metrics on the source grid."""
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from .image_metrics import image_metrics
from ..utils.io import read_jsonl, resolve, sha256


_COORDS = np.indices((5, 5, 5)) - 2
_BALL_2 = (_COORDS ** 2).sum(axis=0) <= 4


def boundary_band(mask, valid):
    mask = np.asarray(mask, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if mask.shape != valid.shape or mask.ndim != 3:
        raise ValueError('LV mask and valid FOV must share a ZYX grid')
    inside = mask & ~binary_erosion(mask, _BALL_2, border_value=1)
    outside = ~mask & binary_dilation(mask, _BALL_2, border_value=0)
    return (inside | outside) & valid


def load_annotations(path, sources):
    path = Path(path).resolve()
    result = defaultdict(dict)
    hashes = {}
    roles = set()
    for row in read_jsonl(path):
        case, phase = row['case_id'], row['phase_label']
        source = sources.get(case)
        if (source is None or row.get('source_sha256') != source['sha256']
                or type(phase) is not int or phase not in source['metadata']['phase_labels'] or phase in result[case]):
            raise ValueError(f'Unknown, mismatched or duplicate LV annotation {case}/{phase}')
        role = row.get('role')
        if role is not None:
            if role not in ('ED', 'ES') or (case, role) in roles:
                raise ValueError(f'Invalid or duplicate LV phase role {case}/{role}')
            roles.add((case, role))
        mask_path = resolve(row['path'], path.parent)
        if mask_path not in hashes:
            hashes[mask_path] = sha256(mask_path)
        if hashes[mask_path] != row['sha256']:
            raise ValueError(f'LV mask checksum differs for {case}/{phase}')
        result[case][phase] = dict(row, path=mask_path)
    return result


def load_extractor(path, device):
    import torch

    # Python opens Unicode paths on Windows without the TorchScript narrow-path loader.
    with Path(path).open('rb') as handle:
        extractor = torch.jit.load(handle, map_location=device).eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    def predict(volume):
        image = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None].to(device)
        with torch.inference_mode():
            prediction = extractor(image)
        if prediction.shape == image.shape:
            prediction = prediction[0, 0]
        if prediction.shape != image.shape[2:]:
            raise ValueError('Frozen extractor must return a binary mask on the input ZYX grid')
        return _binary_mask(prediction.cpu().numpy(), volume.shape)

    return predict


def _binary_mask(values, shape):
    if values.shape != shape or not np.isin(values, (0, 1)).all():
        raise ValueError('Expected a binary LV mask on the original ZYX grid')
    return values.astype(bool)


def _volume_ml(mask, valid, spacing):
    if not mask.any():
        return None
    if np.any(mask & ~valid):
        return None
    return float(mask.sum() * np.prod(spacing) / 1000)


def evaluate_lv(pred, source, metadata, annotations, config, extractor):
    phases = []
    for index, label in enumerate(metadata['phase_labels']):
        annotation = annotations.get(label)
        if annotation is None:
            phases.append({'phase_label': label, 'boundary_status': 'missing_annotation', 'mae': None})
            continue
        role = annotation.get('role')
        with h5py.File(annotation['path'], 'r') as f:
            mask = _binary_mask(f[annotation['dataset']][:], pred.shape[1:])
        valid = np.asarray(source['valid_mask'][index], dtype=bool)
        band = boundary_band(mask, valid)
        reference = volume = None
        if 'target' not in source:
            status = 'missing_target'
            metrics = {'mae': None, 'valid_voxels': int(band.sum())}
        elif not band.any():
            status = 'empty_boundary_roi'
            metrics = {'mae': None, 'valid_voxels': 0}
        else:
            status = 'ok'
            sl = slice(index, index + 1)
            metrics = image_metrics(pred[sl], source['target'][sl], valid[None], config['window'],
                                    band[None], config['ssim_window'], config['sigma'])
        extraction_status = None
        if role:
            spacing = metadata['output_spacing_zyx_mm']
            reference = _volume_ml(mask, valid, spacing)
            if extractor is None:
                extraction_status = 'missing_extractor'
            else:
                try:
                    predicted_mask = extractor(np.asarray(pred[index]))
                    volume = _volume_ml(_binary_mask(predicted_mask, valid.shape), valid, spacing)
                    extraction_status = ('empty_prediction' if not predicted_mask.any()
                                         else 'outside_valid_fov' if volume is None else 'ok')
                except (RuntimeError, ValueError) as exc:
                    extraction_status = 'extractor_failed'
                    metrics['extractor_error'] = str(exc)
        phases.append({'phase_label': label, 'role': role, 'boundary_status': status, **metrics,
                       'reference_volume_ml': reference, 'prediction_volume_ml': volume,
                       'extraction_status': extraction_status})
    roles = {item['role']: item for item in phases if item.get('role')}
    motion = {'status': 'missing_ed_es_reference', 'absolute_error_ml': None}
    if set(roles) == {'ED', 'ES'} and all(row['reference_volume_ml'] is not None for row in roles.values()):
        gt = roles['ED']['reference_volume_ml'] - roles['ES']['reference_volume_ml']
        motion.update(reference_delta_ml=gt, status='prediction_unavailable')
        if all(row['prediction_volume_ml'] is not None for row in roles.values()):
            estimated = roles['ED']['prediction_volume_ml'] - roles['ES']['prediction_volume_ml']
            motion.update(status='ok', prediction_delta_ml=estimated, absolute_error_ml=abs(estimated - gt))
    selected = [row['mae'] for row in phases if row['mae'] is not None]
    return {'phases': phases, 'boundary_mae': float(np.mean(selected)) if selected else None, 'motion': motion}


def summarize_lv(cases):
    values = defaultdict(lambda: defaultdict(list))
    boundaries, extraction, motion = Counter(), Counter(), Counter()
    coverage = defaultdict(lambda: Counter(scans=0, available_phases=0, annotated_phases=0, evaluated_phases=0))
    for case in cases:
        animal = str(case['animal_id'])
        lv = case['lv']
        values[animal]['boundary_mae'].append(lv['boundary_mae'])
        values[animal]['motion_error_ml'].append(lv['motion']['absolute_error_ml'])
        motion[lv['motion']['status']] += 1
        coverage[animal]['scans'] += 1
        for phase in lv['phases']:
            status = phase['boundary_status']
            boundaries[status] += 1
            coverage[animal]['available_phases'] += 1
            coverage[animal]['annotated_phases'] += int(status != 'missing_annotation')
            coverage[animal]['evaluated_phases'] += int(status == 'ok')
            if phase.get('extraction_status'):
                extraction[phase['extraction_status']] += 1

    def mean(items):
        items = [item for item in items if item is not None]
        return float(np.mean(items)) if items else None

    animals = {animal: {metric: mean(items) for metric, items in metrics.items()}
               for animal, metrics in sorted(values.items())}
    metrics = {}
    for metric in ('boundary_mae', 'motion_error_ml'):
        selected = [row[metric] for row in animals.values() if row[metric] is not None]
        metrics[metric] = {'n_animals': len(selected), 'mean': mean(selected)}
    return {'animals': animals, 'animal_equal': metrics, 'coverage': coverage,
            'boundary_status_counts': boundaries, 'extraction_status_counts': extraction,
            'motion_status_counts': motion}
