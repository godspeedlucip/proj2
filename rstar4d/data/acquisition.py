"""Versioned summaries of the views actually used for sparse FDK."""
import h5py
import numpy as np

from ..utils.io import canonical_hash, read_json


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
            layout = record.get('cardiac_layout')
            if layout:
                if (saved.get('layout') != layout['layout'] or saved.get('actual_indices') != layout['actual_indices']
                        or saved.get('phase') != layout['phase']
                        or saved.get('source_paired_sha256') != layout['reference_paired_sha256']
                        or saved.get('projections_sha256') != layout['source_projections_sha256']):
                    raise ValueError('Angle summary does not match the measured layout provenance')
                if not saved.get('diagnostic_swap') and (saved.get('encoded_indices') != layout['actual_indices']
                        or ('correct_angles_sha256' in layout and canonical_hash(saved['values']) != layout['correct_angles_sha256'])):
                    raise ValueError('Angle summary does not encode the selected projection indices')
            result = np.asarray(saved['values'],dtype=np.float32)
            if result.shape != (len(phase_labels), 14) or not np.isfinite(result).all():
                raise ValueError('Invalid angle summary shape or values')
            return result
        layout = record.get('cardiac_layout',{})
        # validate_manifest hashes the raw sidecar before training or inference.
        # Bind that verified file to the acquisition that produced the paired FDK.
        if layout and (not record.get('projections_sha256')
                       or record['projections_sha256']!=layout['source_projections_sha256']):
            raise ValueError('Raw projection source differs from the measured layout')
        with h5py.File(record['projections'], 'r') as source:
            result = angle_summary(source['angles_rad'][:],source['state_label'][:],phase_labels,
                                   selected_indices=layout.get('actual_indices'),selected_phase=layout.get('phase'))
        if 'correct_angles_sha256' in layout and canonical_hash(result.tolist())!=layout['correct_angles_sha256']:
            raise ValueError('Raw angles do not encode the measured selected projection indices')
        return result
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f'{record["case_id"]}: geometry requires valid acquired angles: {exc}') from exc
