"""Pair A/B/C/D scans and aggregate phase-level differences by animal."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import read_json, read_jsonl


def _load(path):
    path = Path(path)
    summary = read_json(path / 'summary.json')
    lv = summary.get('lv')
    if not lv or not lv.get('annotation_sha256'):
        raise ValueError(f'{path}: LV-enabled evaluation required')
    settings = summary['metric_config']
    protocol = {name: lv[name] for name in ('annotation_sha256', 'extractor_sha256', 'boundary_unit')}
    protocol.update({name: settings[name] for name in ('window', 'ssim_window', 'sigma')})
    rows = read_jsonl(path / 'cases.jsonl')
    cases = {row['case_id']: row for row in rows}
    if not cases or len(rows) != len(cases):
        raise ValueError(f'{path}: empty or duplicate cases')
    return cases, protocol, lv


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _delta(a, b):
    return a - b if a is not None and b is not None else None


def _compare_scan(candidate, reference):
    for key in ('animal_id', 'source_sha256', 'inference_protocol', 'temporal_topology'):
        if candidate[key] != reference[key]:
            raise ValueError(f'{candidate["case_id"]}: paired scan {key} differs')
    a, b = candidate['lv'], reference['lv']
    if [phase['phase_label'] for phase in a['phases']] != [phase['phase_label'] for phase in b['phases']]:
        raise ValueError(f'{candidate["case_id"]}: paired phase order differs')
    phases = [_delta(x['mae'], y['mae']) for x, y in zip(a['phases'], b['phases'])]
    global_a = candidate['regions'].get('global', {}).get('prediction', {})
    global_b = reference['regions'].get('global', {}).get('prediction', {})
    return {
        'boundary_mae': _mean(phases),
        'boundary_phases': sum(value is not None for value in phases),
        'ordered_endpoint_mae': _mean([phases[0], phases[-1]]) if candidate['temporal_topology']=='ordered_states' else None,
        'motion_error_ml': _delta(a['motion']['absolute_error_ml'], b['motion']['absolute_error_ml']),
        'motion_signed_bias_ml': _delta(a['motion']['signed_bias_ml'], b['motion']['signed_bias_ml']),
        'global_mae': _delta(global_a.get('mae'), global_b.get('mae')),
        'global_ssim': _delta(global_a.get('ssim'), global_b.get('ssim')),
    }


def _pair(candidate, reference, seed):
    by_animal = defaultdict(list)
    if candidate.keys() != reference.keys():
        raise ValueError('A/B/C/D reports must include the same scans')
    for case, row in sorted(candidate.items()):
        by_animal[str(row['animal_id'])].append(_compare_scan(row, reference[case]))
    metrics = ('boundary_mae', 'ordered_endpoint_mae', 'motion_error_ml',
               'motion_signed_bias_ml', 'global_mae', 'global_ssim')
    animals = {animal: {'scans': len(scans),
                        'boundary_phases': sum(scan['boundary_phases'] for scan in scans),
                        'motion_scans': sum(scan['motion_error_ml'] is not None for scan in scans),
                        **{metric: _mean(scan[metric] for scan in scans) for metric in metrics}}
               for animal, scans in sorted(by_animal.items())}
    rng = np.random.default_rng(seed)
    summary = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in animals.values() if row[metric] is not None])
        interval = None
        if len(values) > 1:
            draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            interval = np.quantile(draws, [.025, .975]).tolist()
        benefits = values > 0 if metric=='global_ssim' else values < 0
        summary[metric] = {'paired_animals': len(values), 'mean_delta': _mean(values),
                           'benefited_animals': int(benefits.sum()) if metric!='motion_signed_bias_ml' else None,
                           'animal_bootstrap_95_interval': interval}
    return {'animals': animals, 'summary': summary}


def compare_cycn(reports, motion_tolerance_ml=0., motion_tolerance_source=None, seed=2026):
    if set(reports) != {'A', 'B', 'C', 'D'}:
        raise ValueError('Supply A, B, C and D evaluations')
    if not np.isfinite(motion_tolerance_ml) or motion_tolerance_ml < 0:
        raise ValueError('Motion tolerance must be finite and nonnegative')
    if motion_tolerance_ml and not motion_tolerance_source:
        raise ValueError('A positive motion tolerance requires its prespecified source')
    loaded = {arm: _load(path) for arm, path in reports.items()}
    protocol = loaded['A'][1]
    if any(entry[1] != protocol for entry in loaded.values()):
        raise ValueError('Metric window, LV annotation or frozen extractor differs')
    comparisons = {f'{a}_minus_{b}': _pair(loaded[a][0], loaded[b][0], seed)
                   for a, b in (('B','A'), ('D','B'), ('D','C'))}
    primary = [comparisons[f'D_minus_{reference}']['summary'] for reference in ('B', 'C')]
    motion = [entry['motion_error_ml'] for entry in primary]
    boundary = [entry['boundary_mae'] for entry in primary]
    # Two distinct animals are the minimum for an animal-level resampling interval.
    assessable = all(item['paired_animals'] >= 2 for item in motion)
    motion_status = ('undetermined' if not assessable else
                     'met' if all(item['mean_delta'] <= motion_tolerance_ml for item in motion) else 'exceeded')
    observed = all(item['mean_delta'] is not None and item['mean_delta'] < 0 for item in boundary)
    bounded = all(item['animal_bootstrap_95_interval'] and item['animal_bootstrap_95_interval'][1] < 0
                  for item in boundary)
    return {'seed': seed, 'protocol': protocol, 'motion_tolerance_ml': motion_tolerance_ml,
            'motion_tolerance_source': motion_tolerance_source or 'prespecified no increase in mean absolute motion error',
            'motion_minimum_paired_animals': 2, 'motion_status': motion_status,
            'boundary_status': 'supported' if bounded else 'directional' if observed else 'not_met',
            'adoption_supported': bounded and motion_status=='met',
            'arms': {arm: {'lv': lv, 'cases': [
                {key: row.get(key) for key in ('case_id', 'checkpoint_sha256', 'resource', 'full_comparison')}
                for row in cases.values()]} for arm, (cases, _, lv) in loaded.items()},
            'comparisons': comparisons}
