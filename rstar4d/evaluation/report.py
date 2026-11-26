from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from .image_metrics import image_metrics
from .lv_metrics import evaluate_lv, load_annotations, load_extractor, summarize_lv
from .projection_metrics import projection_metrics
from ..data.contracts import validate_manifest
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


def evaluate_predictions(predictions,manifest,config,out,lv_annotations=None,lv_extractor=None,lv_device='cpu'):
    if lv_extractor and not lv_annotations:
        raise ValueError('An LV extractor requires trusted LV annotations')
    predictions=Path(predictions).resolve()
    pred_manifest=predictions/'manifest.jsonl' if predictions.is_dir() else predictions
    sources={r['case_id']:r for r in validate_manifest(manifest,require_target=False)}
    annotations=load_annotations(lv_annotations,sources) if lv_annotations else None
    extractor=load_extractor(lv_extractor,lv_device) if lv_extractor else None
    output=[]; flat=[]
    for prediction in read_jsonl(pred_manifest):
        if prediction['case_id'] not in sources: raise ValueError('Prediction has no source case')
        row=sources[prediction['case_id']]; p=resolve(prediction['path'],pred_manifest.parent)
        if sha256(p)!=prediction['sha256'] or prediction['source_sha256']!=row['sha256']: raise ValueError('Prediction/source checksum mismatch')
        if prediction.get('source_metadata_hash')!=canonical_hash(row['metadata']): raise ValueError('Prediction/source metadata mismatch')
        item={'case_id':row['case_id'],'animal_id':row['animal_id'],'intensity_domain':row['metadata']['intensity_domain'],'regions':{},
              'source_sha256':row['sha256'],'checkpoint_sha256':prediction.get('checkpoint_sha256')}
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
            if annotations is not None:
                protocol=json.loads(pf.attrs.get('metadata_json','{}')).get('inference',{})
                item['temporal_topology']=meta['temporal_topology']
                item['inference_protocol']={key:protocol.get(key) for key in
                    ('mode','tile','overlap','split_name','split_hash','precision','device','model_profile')}
                item['resource']=protocol.get('resource')
                item['full_comparison']=protocol.get('full_comparison')
                item['lv']=evaluate_lv(pred,source,meta,annotations.get(row['case_id'],{}),config,extractor)
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
    if annotations is not None:
        summary['lv']=summarize_lv(output)
        units={'mu':'mm^-1','calibrated_mu':'mm^-1','hu':'HU','legacy':'legacy native units'}
        summary['lv'].update(annotation_sha256=sha256(lv_annotations),
            extractor_sha256=sha256(lv_extractor) if lv_extractor else None,
            boundary_unit=units[output[0]['intensity_domain']],motion_error_unit='mL')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    atomic_jsonl(out/'cases.jsonl',output); atomic_json(out/'summary.json',summary)
    if flat:
        with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    return summary


class LabelView:
    def __init__(self,dataset,label): self.dataset=dataset; self.label=label
    def __getitem__(self,index): return np.asarray(self.dataset[index])==self.label
