"""Versioned summaries of the views actually used for sparse FDK."""
import h5py
import numpy as np

from ..utils.io import read_json


ENCODING = 'angles_v1'


def angle_summary(angles_rad, state_label, phase_labels, *, selected_indices=None,
                  selected_phase=None, total_views=None):
    angles = np.asarray(angles_rad, dtype=np.float64)
    state = np.asarray(state_label)
    if angles.ndim != 1 or angles.shape != state.shape or not len(angles) or not np.isfinite(angles).all():
        raise ValueError('Invalid angles_rad/state_label arrays')
    if not np.isin(state, phase_labels).all():
        raise ValueError('Acquisition contains an unrecognized phase')
    denominator = len(angles) if total_views is None else int(total_views)
    if denominator < len(angles):
        raise ValueError('Invalid total view count')
    trajectory = np.unwrap(angles)
    if selected_indices is not None:
        selected = np.asarray(selected_indices)
        if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in 'iu'
                or np.any(selected < 0) or np.any(selected >= len(angles)) or len(np.unique(selected)) != len(selected)):
            raise ValueError('Invalid selected projection indices')
        if selected_phase not in phase_labels or np.any(state[selected] != selected_phase):
            raise ValueError('Selected views must belong to the requested phase')
    result = []
    for label in phase_labels:
        indices = np.flatnonzero(state == label)
        if selected_indices is not None and label == selected_phase:
            indices = np.sort(selected)
        if not len(indices):
            raise ValueError(f'No acquired views for phase {label}')
        bins = np.histogram(np.mod(angles[indices], 2*np.pi), bins=12, range=(0, 2*np.pi))[0]
        gap = np.abs(np.diff(trajectory[indices])).max(initial=0.) / (2*np.pi)
        result.append([*(bins / len(indices)), len(indices) / denominator, gap])
    return np.asarray(result, dtype=np.float32)


def load_geometry(record, phase_labels, *, allow_diagnostic_swap=True):
    try:
        if record.get('angle_summary'):
            saved = read_json(record['angle_summary'])
            if (saved['encoding'] != ENCODING or saved['phase_labels'] != list(phase_labels)
                    or not record.get('projections_sha256')
                    or saved['projections_sha256'] != record['projections_sha256']):
                raise ValueError('Angle summary provenance mismatch')
            if saved.get('paired_sha256',record['sha256'])!=record['sha256']:
                raise ValueError('Angle summary belongs to a different paired volume')
            if saved.get('diagnostic_swap') and not allow_diagnostic_swap:
                raise ValueError('Swapped angle summaries are reserved for inference diagnostics')
            result = np.asarray(saved['values'],dtype=np.float32)
            if result.shape != (len(phase_labels), 14) or not np.isfinite(result).all():
                raise ValueError('Invalid angle summary shape or values')
            return result
        with h5py.File(record['projections'], 'r') as source:
            return angle_summary(source['angles_rad'][:], source['state_label'][:], phase_labels)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f'{record["case_id"]}: geometry requires valid acquired angles: {exc}') from exc
