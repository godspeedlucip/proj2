"""Paired animal-level comparisons for the STAM experiment."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import read_json, read_jsonl


def _read_report(path):
    path = Path(path)
    summary = read_json(path/'summary.json')
    lv = summary.get('lv')
    if not lv or not lv.get('annotation_sha256'):
        raise ValueError(f'LV-enabled evaluation required: {path}')
    protocol = {key: lv[key] for key in ('annotation_sha256', 'extractor_sha256', 'boundary_unit', 'motion_error_unit')}
    metric_config = summary['metric_config']
    protocol.update({key: metric_config[key] for key in ('window', 'ssim_window', 'sigma')})
    rows = read_jsonl(path/'cases.jsonl')
    cases = {row['case_id']: row for row in rows}
    if not cases or len(cases) != len(rows):
        raise ValueError(f'Empty or duplicate cases in {path}')
    return cases, protocol


def _paired_case(candidate, reference):
    for field in ('animal_id', 'source_sha256', 'inference_protocol'):
        if candidate[field] != reference[field]:
            raise ValueError(f'{candidate["case_id"]}: candidate/reference {field} mismatch')
    a, b = candidate['lv'], reference['lv']
    if [row['phase_label'] for row in a['phases']] != [row['phase_label'] for row in b['phases']]:
        raise ValueError(f'{candidate["case_id"]}: phase labels mismatch')
    phases = [(x['mae'] - y['mae']) for x, y in zip(a['phases'], b['phases'])
              if x['mae'] is not None and y['mae'] is not None]
    motion = [entry['motion']['absolute_error_ml'] for entry in (a, b)]
    return {
        'boundary_mae': float(np.mean(phases)) if phases else None,
        'boundary_phases': len(phases),
        'motion_error_ml': motion[0] - motion[1] if all(value is not None for value in motion) else None,
    }


def compare_stam(candidates, references, seeds, motion_tolerance_ml=0.):
    if not seeds or len(set(seeds)) != len(seeds) or len(candidates) != len(references) or len(candidates) != len(seeds):
        raise ValueError('Supply one candidate and reference per distinct seed')
    if not np.isfinite(motion_tolerance_ml) or motion_tolerance_ml < 0:
        raise ValueError('Motion tolerance must be finite and nonnegative')
    per_animal = defaultdict(dict)
    protocol = fixed_sources = None
    for seed, cand, ref in zip(seeds, candidates, references):
        candidates_by_case, current = _read_report(cand)
        references_by_case, reference_protocol = _read_report(ref)
        if current != reference_protocol or (protocol is not None and current != protocol):
            raise ValueError('Evaluation, annotation or frozen extractor protocol differs')
        protocol = current
        sources = {case: (row['animal_id'], row['source_sha256'], row['inference_protocol'])
                   for case, row in candidates_by_case.items()}
        if sources.keys() != references_by_case.keys() or (fixed_sources is not None and sources != fixed_sources):
            raise ValueError('Scans or inference protocols differ across runs')
        fixed_sources = sources
        by_animal = defaultdict(list)
        for case, row in sorted(candidates_by_case.items()):
            by_animal[str(row['animal_id'])].append(_paired_case(row, references_by_case[case]))
        for animal, scans in by_animal.items():
            per_animal[animal][str(seed)] = {
                'scans': len(scans),
                'boundary_phases': sum(row['boundary_phases'] for row in scans),
                'motion_scans': sum(row['motion_error_ml'] is not None for row in scans),
            }
            for metric in ('boundary_mae', 'motion_error_ml'):
                values = [row[metric] for row in scans if row[metric] is not None]
                per_animal[animal][str(seed)][metric] = float(np.mean(values)) if values else None

    rng = np.random.default_rng(2026)
    animal_rows = {animal: {'seeds': by_seed} for animal, by_seed in sorted(per_animal.items())}
    summary = {}
    for metric in ('boundary_mae', 'motion_error_ml'):
        for row in animal_rows.values():
            values = [entry[metric] for entry in row['seeds'].values()]
            row[metric] = float(np.mean(values)) if all(value is not None for value in values) else None
        values = np.asarray([row[metric] for row in animal_rows.values() if row[metric] is not None])
        interval = None
        if len(values) > 1:
            samples = rng.choice(values, size=(10000, len(values)), replace=True)
            interval = np.quantile(samples.mean(axis=1), [.025, .975]).tolist()
        summary[metric] = {'mean_delta': float(values.mean()) if len(values) else None,
                           'paired_animals': len(values), 'improved_animals': int((values < 0).sum()),
                           'animal_bootstrap_95_interval': interval}
    boundary = summary['boundary_mae']['mean_delta']
    motion = summary['motion_error_ml']['mean_delta']
    return {'difference': 'candidate_minus_reference', 'seeds': seeds, 'animals': animal_rows,
            'summary': summary, 'protocol': protocol, 'motion_tolerance_ml': motion_tolerance_ml,
            'observed_criteria_met': (None if boundary is None or motion is None
                                      else boundary < 0 and motion <= motion_tolerance_ml)}
