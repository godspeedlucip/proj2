"""Paired prior retrieval diagnostics on explicitly selected validation blocks."""
from pathlib import Path

import numpy as np
import torch

from ..models import model_from_checkpoint
from ..training.checkpoint import load_checkpoint
from ..training.trainer import select_device
from ..utils.io import atomic_json, read_jsonl, resolve, sha256


def _summarize(a, b, tolerance):
    target = a['target_valid'] & b['target_valid']
    single = target & (a['coarse_count']==1) & (b['coarse_count']==1)
    fine = single & (a['fine_count']>0) & (b['fine_count']>0)
    equal = fine & (a['fine_count']==b['fine_count'])
    scale = torch.maximum(a['value_rms'],b['value_rms']).clamp_min(1e-12)
    comparable = equal & ((a['value_rms']-b['value_rms']).abs()<=tolerance*scale)
    points = int(comparable.sum())
    counts = {'target_sites':target.numel(),'target_valid':int(target.sum()),
              'single_coarse_both':int(single.sum()),'positive_fine_both':int(fine.sum()),
              'equal_fine_count':int(equal.sum()),'value_amplitude_matched':points,'comparable':points}
    record = {'status':'ok' if points else 'no_comparable_single_candidate',
              'points':points,'filter_counts':counts}
    if points:
        for name,values in zip(('matched','mismatched'),(a,b)):
            record[name] = {key:float(values[key][comparable].mean()) for key in
                            ('raw','relative','gate_mean','value_rms','delta_rms')}
        record['delta_rms_difference'] = record['mismatched']['delta_rms']-record['matched']['delta_rms']
    return record


def diagnose(model, arrays, device, value_relative_tolerance, *, phase_labels=None):
    if not 0<=value_relative_tolerance<=1:
        raise ValueError('Prior diagnostic tolerance must be within [0,1]')
    def tensor(name,var_type=torch.float32):
        return torch.as_tensor(arrays[name],device=device,dtype=var_type)[None]
    dynamic = tensor('dynamic').unsqueeze(1)
    phases = dynamic.shape[2]
    if 'phase_labels' in arrays:
        stored = arrays['phase_labels'].tolist()
        if phase_labels is not None and list(phase_labels)!=stored:
            raise ValueError('Prior diagnostic phase labels disagree with manifest')
        phase_labels = stored
    if phase_labels is not None and (len(phase_labels)!=phases or len(set(phase_labels))!=phases):
        raise ValueError('Prior diagnostic requires one unique label per phase')
    context = {'source_fov':tensor('source_fov',torch.bool),'prior_fov':tensor('prior_fov',torch.bool)}
    if model.prior_fusion.mode=='geometry':
        context['geometry'] = tensor('geometry')
    result = []
    for name in ('matched','mismatched'):
        records = []
        def observe(phase, values):
            records.append({key:value.detach().cpu() for key,value in values.items()})
        static = tensor(name).unsqueeze(1)
        with torch.no_grad():
            model.prior_fusion(dynamic,static,{**context,'prior_observer':observe})
        result.append({key:torch.cat([r[key] for r in records],dim=0) for key in records[0]})
    a,b = result
    record = _summarize(a,b,value_relative_tolerance)
    record['phases'] = []
    for index in range(phases):
        pair = [{key:value[index] for key,value in condition.items()} for condition in (a,b)]
        record['phases'].append({'phase_index':index,'phase_label':None if phase_labels is None else phase_labels[index],
                                 **_summarize(*pair,value_relative_tolerance)})
    return record


def diagnose_manifest(checkpoint, manifest, output, device='cpu', tolerance=.1):
    if not 0<=tolerance<=1:
        raise ValueError('Prior diagnostic tolerance must be within [0,1]')
    device = select_device(device)
    model = model_from_checkpoint(load_checkpoint(checkpoint,require_training=False)).to(device).eval()
    if getattr(model,'prior_fusion',None) is None:
        raise ValueError('Prior diagnostic requires a checkpoint with P enabled')
    entries = []
    groups = {}
    path = Path(manifest).resolve()
    for row in read_jsonl(path):
        source = resolve(row['path'],path.parent)
        digest = sha256(source)
        if digest!=row['sha256']:
            raise ValueError(f'{row["case_id"]}: paired prior diagnostic hash mismatch')
        with np.load(source,allow_pickle=False) as arrays:
            measured = diagnose(model,arrays,device,tolerance,phase_labels=row.get('phase_labels'))
        entries.append({'case_id':row['case_id'],'animal_id':str(row['animal_id']),'source_sha256':digest,**measured})
        if measured['status']=='ok':
            groups.setdefault(str(row['animal_id']),[]).append(measured['delta_rms_difference'])
    animal = {key:float(np.mean(values)) for key,values in sorted(groups.items())}
    atomic_json(output,{'checkpoint_sha256':sha256(checkpoint),'value_relative_tolerance':tolerance,'cases':entries,
                        'animal_delta_rms_differences':animal,'valid_animals':len(animal),
                        'mean_paired_difference':float(np.mean(list(animal.values()))) if animal else None})
    return output
