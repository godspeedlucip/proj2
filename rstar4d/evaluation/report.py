from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from .image_metrics import image_metrics
from .projection_metrics import projection_metrics
from ..data.contracts import validate_manifest
from .contours import (CONTOUR_METRICS, contour_inputs, evaluate_case_contours,
                       summarize_contours, surface_metrics, write_csv)
from ..utils.io import read_jsonl, atomic_json, atomic_jsonl, resolve, sha256, canonical_hash


def stats(values):
    values=[v for v in values if v is not None]
    return {'n':len(values),'mean':float(np.mean(values)) if values else None,
            'sd':float(np.std(values,ddof=1)) if len(values)>1 else None}


def projection_summary(cases):
    applicable=[r for r in cases if r.get('projection',{}).get('status')=='ok']
    skipped=[r for r in cases if r.get('projection',{}).get('status')!='ok']
    result={'applicable_cases':len(applicable),'skipped_cases':len(skipped),
        'applicable_animals':len({str(r['animal_id']) for r in applicable}),
        'constant_views':sum(r['projection']['constant_views'] for r in applicable),
        'units':'MAE/RMSE: dimensionless line integral; relative_l2/NCC: dimensionless',
        'skipped_reasons':{},'metrics':{}}
    for row in skipped:
        reason=row.get('projection',{}).get('reason','projection evaluation unavailable')
        result['skipped_reasons'][reason]=result['skipped_reasons'].get(reason,0)+1
    for metric in ['mae','rmse','relative_l2','ncc']:
        animals=defaultdict(list)
        for row in applicable:
            value=row['projection'][metric]
            if value is not None: animals[str(row['animal_id'])].append(value)
        result['metrics'][metric]={'case_equal':stats([r['projection'][metric] for r in applicable]),
            'animal_equal':stats([float(np.mean(v)) for v in animals.values()])}
    return result


def evaluate_predictions(predictions,manifest,config,out,*,contours=None):
    predictions=Path(predictions).resolve()
    pred_manifest=predictions/'manifest.jsonl' if predictions.is_dir() else predictions
    sidecars = [kind for enabled,kind in [(config['projection'],'projections'),(contours,'annotations')] if enabled]
    sources={r['case_id']:r for r in validate_manifest(manifest,require_target=False,sidecars=sidecars)}
    contour_rows,contour_seen = [],set()
    declared,protocol = contour_inputs(Path(contours).resolve()) if contours else ({},None)
    output=[]; flat=[]
    for prediction in read_jsonl(pred_manifest):
        if prediction['case_id'] not in sources: raise ValueError('Prediction has no source case')
        row=sources[prediction['case_id']]; p=resolve(prediction['path'],pred_manifest.parent)
        if sha256(p)!=prediction['sha256'] or prediction['source_sha256']!=row['sha256']: raise ValueError('Prediction/source checksum mismatch')
        if prediction.get('source_metadata_hash')!=canonical_hash(row['metadata']): raise ValueError('Prediction/source metadata mismatch')
        item={'case_id':row['case_id'],'animal_id':row['animal_id'],'intensity_domain':row['metadata']['intensity_domain'],
              'paired_sha256':row['sha256'],'annotations_sha256':row.get('annotations_sha256') if contours else None,
              'regions':{}}
        with h5py.File(p,'r') as pf,h5py.File(row['path'],'r') as source:
            pred=pf['prediction']; meta=row['metadata']
            if pred.shape!=source['cbct'].shape: raise ValueError('Prediction shape mismatch')
            if 'target' not in source: item['reason']='no GT'
            else:
                regions={'global':None}
                if config['roi_labels']:
                    if 'roi_labels' not in source: item['missing_roi_reason']='ROI labels unavailable'
                    else:
                        for label,name in config['roi_labels'].items(): regions[str(name)]=int(label)
                for name,label in regions.items():
                    roi=None if label is None else LabelView(source['roi_labels'],label)
                    pair={kind:image_metrics(a,source['target'],source['valid_mask'],config['window'],roi,config['ssim_window'],config['sigma'])
                          for kind,a in [('input',source['cbct']),('prediction',pred)]}
                    item['regions'][name]=pair
                    for kind,metrics in pair.items():
                        flat.append({'case_id':row['case_id'],'animal_id':row['animal_id'],'region':name,'kind':kind,
                            'domain':meta['intensity_domain'],**metrics})
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
    summary={'cases':len(output),'metrics':{},'metric_config':config}
    for region,kind,domain in sorted({(r['region'],r['kind'],r['domain']) for r in flat}):
        selected=[r for r in flat if (r['region'],r['kind'],r['domain'])==(region,kind,domain)]
        metrics={}
        for metric in ['mae','rmse','ssim']:
            animals=defaultdict(list)
            for row in selected:
                if row[metric] is not None: animals[str(row['animal_id'])].append(row[metric])
            metrics[metric]={'case_equal':stats([r[metric] for r in selected]),
                'animal_equal':stats([float(np.mean(v)) for v in animals.values()])}
        summary['metrics'][f'{region}/{kind}/{domain}']=metrics
    if config['projection']: summary['projection']=projection_summary(output)
    if contours:
        if set(declared)!=contour_seen:
            raise ValueError(f'Unknown contour prediction keys: {sorted(set(declared)-contour_seen)}')
        animal_rows,volumes=summarize_contours(contour_rows)
        contour_summary={}
        for structure in ('heart_outer','lv'):
            animals=[a for a in animal_rows if a['structure']==structure]
            contour_summary[structure]={metric:stats([a[metric] for a in animals]) for metric in CONTOUR_METRICS}
            for key in ('reference_states','empty_predictions','missing_predictions'):
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
    if flat:
        with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    return summary


class LabelView:
    def __init__(self,dataset,label): self.dataset=dataset; self.label=label
    def __getitem__(self,index): return np.asarray(self.dataset[index])==self.label
