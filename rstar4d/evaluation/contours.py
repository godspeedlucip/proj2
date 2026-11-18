"""Evaluation of independently delineated contours on the common output grid."""
from collections import defaultdict
from contextlib import nullcontext
import csv
import json
from math import isfinite

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

from ..data.cardiac_annotations import STRUCTURES, VERSION, trusted_surface
from ..utils.io import read_jsonl, resolve, sha256
from ..data.contracts import sample_id


CONTOUR_METRICS = ('surface_dice','hd95','assd','band_mae')


def surface_metrics(pred_surface, ref_surface, tolerance_vox):
    if not np.any(ref_surface):
        return {'status':'no_reference','surface_dice':None,'hd95':None,'assd':None}
    if not np.any(pred_surface):
        return {'status':'empty_prediction','surface_dice':0.,'hd95':None,'assd':None}
    directed = (distance_transform_edt(~ref_surface)[pred_surface],
                distance_transform_edt(~pred_surface)[ref_surface])
    symmetric = np.concatenate(directed)
    return {'status':'ok','surface_dice':float((symmetric<=tolerance_vox).mean()),
            'hd95':max(float(np.percentile(dist,95)) for dist in directed),'assd':float(symmetric.mean())}


def contour_inputs(path):
    result = {}
    protocol = None
    for row in read_jsonl(path):
        key = (sample_id(row),row['phase_label'],row['structure'])
        if key in result or row['structure'] not in STRUCTURES:
            raise ValueError(f'Invalid or duplicate contour key: {key}')
        current = row['delineation']
        if current.get('method')=='frozen_segmenter':
            if not current.get('weights_sha256') or 'threshold' not in current or not current.get('version'):
                raise ValueError('Frozen segmenter requires version, weights SHA256 and threshold')
        elif current.get('method')=='blinded':
            if not current.get('batch'):
                raise ValueError('Blinded delineation requires a batch ID')
        else:
            raise ValueError('Expected independent delineation provenance')
        if protocol is not None and current!=protocol:
            raise ValueError('All contours in one evaluation use the same declared protocol')
        protocol = current
        row['path'] = resolve(row['path'],path.parent)
        result[key] = row
    return result,protocol


def _closed_mask(mask, trust):
    interior = mask[1:-1,1:-1,1:-1]
    return bool(mask.any() and interior.sum()==mask.sum() and not (binary_dilation(mask) & ~trust).any())


def evaluate_case_contours(record, prediction, pred_dataset, source, inputs, cfg):
    meta = record['metadata']
    tolerances = cfg['surface_dice_tolerance_vox']
    if any(type(tolerances[name]) not in (int,float) or tolerances[name]<0 or not isfinite(tolerances[name]) for name in STRUCTURES):
        raise ValueError('Freeze a nonnegative surface Dice tolerance in output voxels for both structures')
    half_width = cfg['half_width_vox']
    if not isfinite(half_width) or half_width<=0:
        raise ValueError('Evaluation half-width must be positive')
    annotations = h5py.File(record['annotations'],'r') if record.get('annotations') else nullcontext()
    results = []
    seen = set()
    hashes = {}
    with annotations as ref:
        if ref is not None:
            info = json.loads(ref.attrs['metadata_json'])
            if (info['paired_sha256']!=record['sha256'] or info['distance_unit']!='output_voxel'
                    or info['version']!=VERSION or info['phase_labels']!=meta['phase_labels']):
                raise ValueError(f'{record["case_id"]}: incompatible reference contour sidecar')
        for t,label in enumerate(meta['phase_labels']):
            for k,structure in enumerate(STRUCTURES):
                key = (sample_id(record),label,structure)
                supplied = inputs.get(key)
                pred_mask = None
                if supplied is not None:
                    seen.add(key)
                    path = supplied['path']
                    if path not in hashes:
                        hashes[path] = sha256(path)
                    if (supplied['case_id']!=record['case_id'] or prediction['sha256']!=supplied['prediction_sha256'] or hashes[path]!=supplied['sha256']
                            or hashes[path]==record.get('annotations_sha256')):
                        raise ValueError(f'{key}: independent contour source/hash mismatch')
                    with h5py.File(path,'r') as file:
                        values = file[supplied['dataset']][:]
                    if values.shape!=pred_dataset.shape[1:] or not np.isin(values,[0,1]).all():
                        raise ValueError(f'{key}: contour must be binary ZYX on the output grid')
                    pred_mask = values.astype(bool)
                volume = None
                band_mae = None
                metrics = {'status':'no_reference','surface_dice':None,'hd95':None,'assd':None}
                if ref is not None and ref['annotation_available'][t,k]:
                    trust = ref['domain'][t,k].astype(bool)
                    reference = ref['surface'][t,k].astype(bool)
                    if pred_mask is None:
                        metrics['status'] = 'missing_prediction'
                    else:
                        pred_surface = trusted_surface(pred_mask,trust)
                        metrics = surface_metrics(pred_surface,reference,tolerances[structure])
                        declaration = info['declarations'].get(str((label,structure)),{})
                        if declaration.get('closed') is True and _closed_mask(pred_mask,trust):
                            volume = int(pred_mask.sum())
                    if 'target' in source:
                        band = trust & (ref['distance'][t,k]<=half_width)
                        if band.any():
                            band_mae = float(np.abs(pred_dataset[t][band]-source['target'][t][band]).mean())
                declaration = info['declarations'].get(str((label,structure)),{}) if ref is not None else {}
                reference_volume = None
                if declaration.get('closed') is True and ref is not None and 'region' in ref:
                    region = ref['region'][t,k].astype(bool)
                    if _closed_mask(region,ref['domain'][t,k].astype(bool)):
                        reference_volume = int(region.sum())
                results.append({'case_id':record['case_id'],'sample_id':sample_id(record),'layout':record.get('layout'),
                                'animal_id':record['animal_id'],'phase_label':label,
                                'structure':structure,'distance_unit':'output_voxel','tolerance_vox':tolerances[structure],
                                'spacing_zyx_mm':meta['output_spacing_zyx_mm'],'band_mae':band_mae,
                                'prediction_volume_voxels':volume,'reference_volume_voxels':reference_volume,**metrics})
    return results,seen


def summarize_contours(rows):
    groups = defaultdict(list)
    trajectories = defaultdict(list)
    for row in rows:
        groups[str(row['animal_id']),row['structure']].append(row)
        if row['reference_volume_voxels'] is not None:
            trajectories[sample_id(row),row['structure']].append(row)
    volumes = []
    for (identifier,structure), phases in sorted(trajectories.items()):
        first = phases[0]
        for previous,current in zip(phases,phases[1:]):
            def delta(key):
                return current[key]-previous[key] if current[key] is not None and previous[key] is not None else None
            volumes.append({'sample_id':identifier,'case_id':first['case_id'],'layout':first.get('layout'),
                            'animal_id':first['animal_id'],'structure':structure,
                            'from_phase':previous['phase_label'],'to_phase':current['phase_label'],
                            'reference_delta_voxels':delta('reference_volume_voxels'),
                            'prediction_delta_voxels':delta('prediction_volume_voxels')})
    animals = []
    for (animal,structure), group in sorted(groups.items()):
        states = defaultdict(list)
        for row in group:
            states[row['case_id'],row['phase_label']].append(row)
        item = {'animal_id':animal,'structure':structure,
                'reference_states':sum(any(r['status']!='no_reference' for r in layouts) for layouts in states.values()),
                'reference_observations':sum(r['status']!='no_reference' for r in group),
                'empty_predictions':sum(r['status']=='empty_prediction' for r in group),
                'missing_predictions':sum(r['status']=='missing_prediction' for r in group)}
        for metric in CONTOUR_METRICS:
            state_means = []
            observed_cases = set()
            observations = 0
            for (case,phase),layouts in states.items():
                values = [row[metric] for row in layouts if row[metric] is not None]
                if values:
                    state_means.append(float(np.mean(values)))
                    observations += len(values)
                    observed_cases.add(case)
            item[metric] = float(np.mean(state_means)) if state_means else None
            item[metric+'_states'] = len(state_means)
            item[metric+'_cases'] = len(observed_cases)
            item[metric+'_observations'] = observations
        animals.append(item)
    return animals,volumes


def write_csv(path, rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
