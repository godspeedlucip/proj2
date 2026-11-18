from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import h5py
import numpy as np
from .image_metrics import image_metrics
from .projection_metrics import projection_metrics
from ..training.tensorboard import write_evaluation_events
from ..data.contracts import sample_id, validate_manifest
from .contours import (CONTOUR_METRICS, contour_inputs, evaluate_case_contours,
                       summarize_contours, surface_metrics, write_csv)
from ..utils.io import read_jsonl, atomic_json, atomic_jsonl, resolve, sha256, canonical_hash


def stats(values):
    values=[v for v in values if v is not None]
    return {'n':len(values),'mean':float(np.mean(values)) if values else None,
            'sd':float(np.std(values,ddof=1)) if len(values)>1 else None}


def _case_and_animal_stats(case_values):
    animals = defaultdict(list)
    for (animal,case),value in case_values.items():
        animals[animal].append(value)
    return {'case_equal':stats(list(case_values.values())),
            'animal_equal':stats([float(np.mean(values)) for values in animals.values()])}


def image_summary(rows):
    states = defaultdict(list)
    for row in rows:
        states[str(row['animal_id']),row['case_id'],row['phase_label']].append(row)
    result = {}
    for metric,denominator in [('mae','valid_voxels'),('rmse','valid_voxels'),('ssim','ssim_centers')]:
        case_numerators = defaultdict(float)
        case_denominators = defaultdict(float)
        valid_states = observations = 0
        for (animal,case,phase),layouts in states.items():
            measured = [r for r in layouts if r[metric] is not None and r[denominator]>0]
            if not measured:
                continue
            # Average the per-layout sums and counts before pooling anatomical phases.
            # Replicating a layout therefore adds no extra voxels/windows to the phase weight.
            for row in measured:
                weight = row[denominator]/len(measured)
                value = row[metric]**2 if metric=='rmse' else row[metric]
                case_numerators[animal,case] += weight*value
                case_denominators[animal,case] += weight
            valid_states += 1
            observations += len(measured)
        case_values = {key:total/case_denominators[key] for key,total in case_numerators.items()}
        if metric=='rmse':
            case_values = {key:float(np.sqrt(value)) for key,value in case_values.items()}
        result[metric] = {**_case_and_animal_stats(case_values),'valid_states':valid_states,
                          'valid_state_layout_pairs':observations}
    return result


def projection_summary(cases):
    seen_cases = set()
    applicable = defaultdict(list)
    skipped = []
    for i,row in enumerate(cases):
        key = (str(row['animal_id']),row.get('case_id',i))
        seen_cases.add(key)
        p = row.get('projection',{})
        if p.get('status')=='ok':
            applicable[key].append(p)
        else:
            skipped.append(p)
    result={'applicable_cases':len(applicable),'skipped_cases':len(seen_cases-applicable.keys()),
        'applicable_samples':sum(map(len,applicable.values())),'skipped_samples':len(skipped),
        'applicable_animals':len({animal for animal,case in applicable}),
        'constant_views':sum(p['constant_views'] for layouts in applicable.values() for p in layouts),
        'units':'MAE/RMSE: dimensionless line integral; relative_l2/NCC: dimensionless',
        'skipped_reasons':{},'metrics':{}}
    for p in skipped:
        reason=p.get('reason','projection evaluation unavailable')
        result['skipped_reasons'][reason]=result['skipped_reasons'].get(reason,0)+1
    for metric in ['mae','rmse','relative_l2','ncc']:
        case_values = {}
        for case,layouts in applicable.items():
            values = [p[metric] for p in layouts if p[metric] is not None]
            if values:
                case_values[case] = float(np.mean(values))
        result['metrics'][metric] = _case_and_animal_stats(case_values)
    return result


def evaluate_predictions(predictions,manifest,config,out,*,contours=None):
    predictions=Path(predictions).resolve()
    pred_manifest=predictions/'manifest.jsonl' if predictions.is_dir() else predictions
    sidecars = [kind for enabled,kind in [(config['projection'],'projections'),(contours,'annotations')] if enabled]
    sources={r['sample_id']:r for r in validate_manifest(manifest,require_target=False,sidecars=sidecars)}
    contour_rows,contour_seen = [],set()
    declared,protocol = contour_inputs(Path(contours).resolve()) if contours else ({},None)
    output=[]; flat=[]; flat_phases=[]; seen_samples=set()
    for prediction in read_jsonl(pred_manifest):
        identifier=sample_id(prediction)
        if identifier in seen_samples: raise ValueError(f'Duplicate prediction sample_id: {identifier}')
        seen_samples.add(identifier)
        if identifier not in sources: raise ValueError('Prediction has no source sample')
        row=sources[identifier]; p=resolve(prediction['path'],pred_manifest.parent)
        if (prediction['case_id']!=row['case_id'] or str(prediction['animal_id'])!=str(row['animal_id'])
                or prediction.get('layout',row.get('layout'))!=row.get('layout')):
            raise ValueError(f'{identifier}: prediction/source identity mismatch')
        if sha256(p)!=prediction['sha256'] or prediction['source_sha256']!=row['sha256']: raise ValueError('Prediction/source checksum mismatch')
        if prediction.get('source_metadata_hash')!=canonical_hash(row['metadata']): raise ValueError('Prediction/source metadata mismatch')
        identity = {'case_id':row['case_id'],'animal_id':row['animal_id'],'sample_id':identifier,'layout':row.get('layout')}
        item={**identity,'intensity_domain':row['metadata']['intensity_domain'],
              'paired_sha256':row['sha256'],'annotations_sha256':row.get('annotations_sha256') if contours else None,
              'regions':{}}
        with h5py.File(p,'r') as pf,h5py.File(row['path'],'r') as source:
            pred=pf['prediction']; meta=row['metadata']
            if pred.shape!=source['cbct'].shape: raise ValueError('Prediction shape mismatch')
            labels=meta['phase_labels']
            item['phases']=[{'phase_label':label,'regions':{}} for label in labels]
            if 'target' not in source:
                item['reason']='no GT'
                for phase in item['phases']:
                    phase['reason']='no GT'
            else:
                regions={'global':None}
                if config['roi_labels']:
                    if 'roi_labels' not in source: item['missing_roi_reason']='ROI labels unavailable'
                    else:
                        for label,name in config['roi_labels'].items(): regions[str(name)]=int(label)
                for name,label in regions.items():
                    roi=None if label is None else LabelView(source['roi_labels'],label)
                    pair={kind:image_metrics(a,source['target'],source['valid_mask'],config['window'],roi,
                                             config['ssim_window'],config['sigma'],phase_labels=labels)
                          for kind,a in [('input',source['cbct']),('prediction',pred)]}
                    item['regions'][name]=pair
                    for kind,metrics in pair.items():
                        columns={**identity,'region':name,'kind':kind,'domain':meta['intensity_domain']}
                        for phase,values in zip(item['phases'],metrics.pop('by_phase')):
                            phase['regions'].setdefault(name,{})[kind]={k:v for k,v in values.items() if k!='phase_label'}
                            flat_phases.append({**columns,**values})
                        flat.append({**columns,**metrics})
            if config['projection']: item['projection']=projection_metrics(pred,meta,row.get('projections'))
            if contours:
                rows,seen=evaluate_case_contours(row,prediction,pred,source,declared,config['contours'])
                contour_rows.extend(rows)
                contour_seen.update(seen)
            if 'cardiac_layout_json' in source.attrs:
                item['cardiac_layout']=json.loads(source.attrs['cardiac_layout_json'])
            if 'metadata_json' in pf.attrs:
                item['inference']=json.loads(pf.attrs['metadata_json']).get('inference',{})
        output.append(item)
    if not output: raise ValueError('No predictions to evaluate')
    summary={'cases':len({(str(r['animal_id']),r['case_id']) for r in output}),
             'samples':len(output),'animals':len({str(r['animal_id']) for r in output}),
             'metrics':{},'metric_config':config}
    regions = defaultdict(list)
    for row in flat_phases:
        regions[row['region'],row['kind'],row['domain']].append(row)
    for (region,kind,domain),selected in sorted(regions.items()):
        summary['metrics'][f'{region}/{kind}/{domain}']=image_summary(selected)
    if config['projection']: summary['projection']=projection_summary(output)
    if contours:
        if set(declared)!=contour_seen:
            raise ValueError(f'Unknown contour prediction keys: {sorted(set(declared)-contour_seen)}')
        animal_rows,volumes=summarize_contours(contour_rows)
        contour_summary={}
        for structure in ('heart_outer','lv'):
            animals=[a for a in animal_rows if a['structure']==structure]
            contour_summary[structure]={metric:{**stats([a[metric] for a in animals]),
                'valid_states':sum(a[metric+'_states'] for a in animals),
                'valid_cases':sum(a[metric+'_cases'] for a in animals),
                'valid_state_layout_pairs':sum(a[metric+'_observations'] for a in animals)} for metric in CONTOUR_METRICS}
            for key in ('reference_states','reference_observations','empty_predictions','missing_predictions'):
                contour_summary[structure][key]=sum(a[key] for a in animals)
            contour_summary[structure]['reference_animals']=sum(a['reference_states']>0 for a in animals)
        summary['contours']={'distance_unit':'output_voxel','delineation':protocol,'structures':contour_summary}
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    atomic_jsonl(out/'cases.jsonl',output); atomic_json(out/'summary.json',summary)
    if contours:
        atomic_jsonl(out/'contours_by_phase.jsonl',contour_rows)
        write_csv(out/'contours_by_phase.csv',contour_rows)
        write_csv(out/'contours_by_animal.csv',animal_rows)
        write_csv(out/'annotated_volume_changes.csv',volumes)
    write_csv(out/'metrics.csv',flat)
    write_csv(out/'metrics_by_phase.csv',flat_phases)
    write_evaluation_events(out, summary)
    return summary


class LabelView:
    def __init__(self,dataset,label): self.dataset=dataset; self.label=label
    def __getitem__(self,index): return np.asarray(self.dataset[index])==self.label
