"""Export reconstruction, frozen masks and comparable held-out scan metrics."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np
import torch

from experiments.frozen_cardiac_segmenter import Segmenter, VERSION, predict_mask
from rstar4d.data.acquisition import load_geometry
from rstar4d.data.cardiac_annotations import trusted_surface
from rstar4d.data.contracts import read_manifest, validate_split
from rstar4d.evaluation.contours import surface_metrics, write_csv
from rstar4d.evaluation.image_metrics import image_metrics
from rstar4d.inference.sliding_window import predict_volume
from rstar4d.models import model_from_checkpoint
from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.utils.io import atomic_json, read_json, sha256


WINDOW = (0., .05)
STRUCTURES = ('heart_outer', 'lv')
SURFACE_TOLERANCE = 2.


def psnr(mse, window=WINDOW):
    if mse is None or mse == 0:
        return None
    return 10. * math.log10((window[1]-window[0])**2 / mse)


def image_stats(prediction, reference, valid):
    difference = prediction[valid].astype(np.float64) - reference[valid]
    count = int(difference.size)
    if count == 0:
        return {'voxels': count, 'mse': None, 'psnr_db': None, 'mae': None, 'rmse': None}
    if not np.isfinite(difference).all():
        raise ValueError('Nonfinite image values within the valid evaluation region')
    mse = float(np.dot(difference,difference)/count)
    return {'voxels': count, 'mse': mse, 'psnr_db': psnr(mse),
            'mae': float(np.abs(difference).mean()), 'rmse': math.sqrt(mse)}


def mask_stats(prediction, reference, valid):
    p, r = prediction & valid, reference & valid
    a, b = int(np.count_nonzero(p)), int(np.count_nonzero(r))
    intersection = int(np.count_nonzero(p&r))
    return {'dice': 2*intersection/(a+b) if (a+b) else None,
            'intersection_voxels': intersection, 'predicted_voxels': a, 'reference_voxels': b}


def contour_row(prediction, target, predicted_mask, annotation, t, j, region, domain):
    predicted_surface = trusted_surface(predicted_mask, domain)
    reference_surface = annotation['surface'][t,j].astype(bool)
    union = predicted_surface | reference_surface
    if union.any():
        bounds = []
        for axis in range(union.ndim):
            selected = np.flatnonzero(union.any(axis=tuple(other for other in range(union.ndim) if other!=axis)))
            bounds.append(slice(selected[0],selected[-1]+1))
        cropped = tuple(bounds)
        predicted_surface, reference_surface = predicted_surface[cropped], reference_surface[cropped]
    band = domain & (annotation['distance'][t,j]<=SURFACE_TOLERANCE)
    return {**mask_stats(predicted_mask,region,domain),
            **surface_metrics(predicted_surface,reference_surface,SURFACE_TOLERANCE),
            'band_mae':image_stats(prediction,target,band)['mae']}


def image_row(predicted, target, domain, region, source, identity, *, ssim=False):
    if not ssim:
        return {**identity, 'region':region,'source':source,**image_stats(predicted,target,domain),
                'ssim':None,'ssim_centers':None}
    result = image_metrics(predicted[None],target[None],domain[None],window=WINDOW)
    mse = result['rmse']**2 if result['rmse'] is not None else None
    return {**identity,'region':region,'source':source,'voxels':result['valid_voxels'],
            'mse':mse,'psnr_db':psnr(mse),'mae':result['mae'],'rmse':result['rmse'],
            'ssim':result['ssim'],'ssim_centers':result['ssim_centers']}


def write_prediction(path, source, meta, model, ck, device, tile, overlap, row, signature):
    if path.exists():
        with h5py.File(path,'r') as f:
            if json.loads(f.attrs['signature'])!=signature or f['prediction'].shape!=source['cbct'].shape:
                raise ValueError(f'Existing prediction identity mismatch at {path}')
        return
    start = time.monotonic()
    recipe = ck.get('stage2_resolved')
    prior_fov = source['prior_valid_mask'][:] if 'prior_valid_mask' in source else None
    context = {'case_id':row['case_id'],'phase_labels':meta['phase_labels'],
               'source_fov':source['valid_mask'][:],'prior_fov':prior_fov}
    if recipe and recipe['prior']['mode']=='geometry':
        angle_file = 'angle_summary' if row.get('angle_summary') else 'projections'
        if sha256(row[angle_file]) != row[angle_file+'_sha256']:
            raise ValueError(f"{row['case_id']}: changed angle sidecar")
        context['geometry'] = load_geometry(row,meta['phase_labels'],allow_diagnostic_swap=False)
    result = predict_volume(model,source['cbct'][:],source['prior'][:],ck['normalization'],
                            tile=tile,overlap=overlap,device=device,amp=True,
                            context=context,recipe=recipe)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.recon-') as temporary:
        pending = Path(temporary)/path.name
        with h5py.File(pending,'x') as f:
            f.attrs['signature'] = json.dumps(signature)
            f.attrs['metadata_json'] = json.dumps(meta)
            f.attrs['inference_seconds'] = time.monotonic()-start
            chunks = (1, *(min(n, limit) for n, limit in zip(result.shape[1:], (16,128,128))))
            f.create_dataset('prediction',data=result,chunks=chunks,compression='lzf')
        pending.rename(path)


def write_masks(path, prediction, source, meta, model, device, signature):
    if path.exists():
        with h5py.File(path,'r') as f:
            if json.loads(f.attrs['signature'])!=signature or f['heart_outer'].shape!=prediction.shape:
                raise ValueError(f'Existing mask identity mismatch at {path}')
        return
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.masks-') as temporary:
        pending = Path(temporary)/path.name
        with h5py.File(pending,'x') as f:
            f.attrs['signature'] = json.dumps(signature)
            f.attrs['metadata_json'] = json.dumps(meta)
            chunks = (1, *(min(n, limit) for n, limit in zip(prediction.shape[1:], (16,128,128))))
            datasets = [f.create_dataset(structure,shape=prediction.shape,dtype='u1',
                                         chunks=chunks,compression='gzip',compression_opts=1)
                        for structure in STRUCTURES]
            for t in range(len(prediction)):
                label = predict_mask(model,prediction[t],device)
                valid = source['valid_mask'][t].astype(bool)
                for threshold, dataset in enumerate(datasets,1):
                    dataset[t] = (label>=threshold)&valid
        pending.rename(path)


def score_case(row, source, prediction, masks, meta):
    if 'target' not in source:
        return [], []
    volume = []
    segmentation = []
    with (h5py.File(row['annotations'],'r') if row.get('annotations') else nullcontext()) as anno:
        if anno is not None:
            info = json.loads(anno.attrs['metadata_json'])
            if info['paired_sha256'] != row['sha256'] or info['phase_labels']!=meta['phase_labels']:
                raise ValueError('Label cache does not match the paired input')
        for t, label in enumerate(meta['phase_labels']):
            fields = {'case_id':row['case_id'],'phase_label':label,
                      'biological_animal_id':row.get('biological_animal_id')}
            target, pred = source['target'][t], prediction[t]
            baseline = source['cbct'][t]
            fov = source['valid_mask'][t].astype(bool)
            for kind, image in (('prediction',pred),('input',baseline)):
                volume.append(image_row(image,target,fov,'fov',kind,fields,ssim=True))
            if anno is None:
                continue
            for j, structure in enumerate(STRUCTURES):
                if not anno['annotation_available'][t,j]:
                    continue
                domain = anno['domain'][t,j].astype(bool)
                region = anno['region'][t,j].astype(bool)
                segmentation.append({**fields,'structure':structure,
                    'surface_tolerance_output_vox':SURFACE_TOLERANCE,
                    'output_spacing_zyx_mm':meta['output_spacing_zyx_mm'],
                    **contour_row(pred,target,masks[structure][t].astype(bool),anno,t,j,region,domain)})
                for kind, image in (('prediction',pred),('input',baseline)):
                    volume.append(image_row(image,target,region&domain,structure,kind,fields))
    return volume,segmentation


def case_result(row, directory, model, ck, segmenter, device, tile, overlap, hashes):
    case = row['case_id']
    prediction_path = directory/'predictions'/case/'prediction.h5'
    mask_path = directory/'masks'/case/'masks.h5'
    artifact = directory/'evaluation'/f'{case}.json'
    signature = {**hashes,'source':row['sha256'],'angles':row.get('projections_sha256'),
                 'angle_summary':row.get('angle_summary_sha256'),
                 'annotations':row.get('annotations_sha256'),'tile':list(tile),'overlap':list(overlap)}
    if artifact.exists():
        saved = read_json(artifact)
        if saved['signature'] != signature or not prediction_path.is_file() or not mask_path.is_file():
            raise ValueError(f'Existing evaluation has different or missing sources for {case}')
        return saved
    for path in (prediction_path,mask_path,artifact):
        path.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(row['path'],'r') as source:
        meta = json.loads(source.attrs['metadata_json'])
        if 'target' in source and (meta['intensity_domain']!=ck['intensity_domain']
                or meta['calibration_application']!=ck['calibration_application']):
            raise ValueError(f'Unexpected simulation intensity domain for {case}')
        write_prediction(prediction_path,source,meta,model,ck,device,tile,overlap,row,signature)
        with h5py.File(prediction_path,'r') as prediction:
            write_masks(mask_path,prediction['prediction'],source,meta,segmenter,device,signature)
            with h5py.File(mask_path,'r') as masks:
                image, contour = score_case(row,source,prediction['prediction'],masks,meta)
    record = {'case_id':case,'signature':signature,'image':image,'segmentation':contour,
              'prediction':str(prediction_path),'masks':str(mask_path)}
    atomic_json(artifact,record)
    print(f'Completed test output {directory.name} {case}',flush=True)
    return record


def per_case(rows, group, metric):
    by_region = defaultdict(lambda:defaultdict(list))
    for row in rows:
        value = row.get(metric)
        if value is not None:
            key = '/'.join(str(row[key]) for key in group)
            by_region[key][row['case_id']].append(value)
    return {key:{case:(float(np.mean(values)),len(values)) for case,values in sorted(cases.items())}
            for key,cases in sorted(by_region.items())}


def scan_table(rows, group, fields):
    by_scan = {}
    for field in fields:
        for kind, cases in per_case(rows,group,field).items():
            for case, result in cases.items():
                by_scan.setdefault((case,kind),{})[field] = result
    return [{'case_id':case,'region_or_structure':kind,
             **{field:values.get(field,(None,0))[0] for field in fields},
             **{f'{field}_phases':values.get(field,(None,0))[1] for field in fields}}
            for (case,kind),values in sorted(by_scan.items())]


def summarize(rows, group, metric):
    return {key:{'scan_equal_mean':float(np.mean([value for value,_ in cases.values()])),
                 'scans':len(cases),'phases':sum(count for _,count in cases.values())}
            for key,cases in per_case(rows,group,metric).items()}


def run(args):
    device = torch.device(args.device)
    ck = load_checkpoint(args.checkpoint)
    if ck['stage'] != 2:
        raise ValueError('Expected the Stage II validation-best checkpoint')
    model = model_from_checkpoint(ck).to(device).eval()
    segmenter = Segmenter().to(device).eval()
    weights = args.segmenter/'best.pt'
    frozen = read_json(args.segmenter/'frozen.json')
    if (frozen['version']!=VERSION or sha256(weights)!=frozen['weights_sha256']
            or sha256(args.manifest)!=frozen['identity']['manifest_sha256']
            or sha256(args.split)!=frozen['identity']['split_sha256']):
        raise ValueError('Frozen segmenter checkpoint verification failed')
    segmenter.load_state_dict(torch.load(weights,map_location=device,weights_only=True)['model'])
    hashes = {'checkpoint':sha256(args.checkpoint),'mask_network':frozen['weights_sha256']}
    rows = read_manifest(args.manifest)
    split = validate_split(rows,args.split)
    if ck['split']!=split:
        raise ValueError('Evaluation split differs from the Stage II checkpoint')
    test = [row for row in rows if row['animal_id'] in split['test']]
    if not test:
        raise ValueError('No held-out scans')
    trained_sources = {row['case_id']:row for row in ck['data_snapshot']['cases']}
    for row in test:
        saved = trained_sources[row['case_id']]
        if any(row.get(field)!=saved.get(field) for field in
               ('sha256','projections_sha256','angle_summary_sha256','annotations_sha256')):
            raise ValueError(f"Test source changed since training: {row['case_id']}")
    cases = [case_result(row,args.out,model,ck,segmenter,device,args.tile,args.overlap,hashes) for row in test]
    image = [entry for case in cases for entry in case['image']]
    masks = [entry for case in cases for entry in case['segmentation']]
    if not image or {r['structure'] for r in masks}!=set(STRUCTURES):
        raise ValueError('Missing held-out intensity or two-structure mask evidence')
    eval_path = args.out/'evaluation'
    image_fields = ('mse','psnr_db','mae','rmse','ssim')
    contour_fields = ('surface_dice','hd95','assd','band_mae')
    write_csv(eval_path/'image_by_phase.csv',image)
    write_csv(eval_path/'mask_dice_by_phase.csv',masks)
    write_csv(eval_path/'image_by_scan.csv',scan_table(image,('region','source'),image_fields))
    write_csv(eval_path/'mask_dice_by_scan.csv',scan_table(masks,('structure',),('dice',*contour_fields)))
    summary = {'case_count':len(cases),'phase_count':sum(r['region']=='fov' and r['source']=='prediction' for r in image),
               'stage2_best_epoch':ck['completed_epoch'],'window':list(WINDOW),'delineation':frozen,
               'metric_aggregation':'mean within scan, then equal mean across scans',
               'image':{name:summarize(image,('region','source'),name)
                        for name in image_fields},
               'dice':summarize(masks,('structure',),'dice'),
               'surface_tolerance_output_vox':SURFACE_TOLERANCE,
               'contours':{name:summarize(masks,('structure',),name)
                           for name in contour_fields}}
    if args.measured:
        for row in read_manifest(args.measured):
            case_result(row,args.measured_out,model,ck,segmenter,device,args.tile,args.overlap,hashes)
        summary['measured_output'] = str(args.measured_out)
    atomic_json(eval_path/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','split','segmenter','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--measured',type=Path)
    p.add_argument('--measured-out',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--tile',type=int,nargs=3,default=(32,64,64))
    p.add_argument('--overlap',type=int,nargs=3,default=(8,16,16))
    args = p.parse_args()
    if (args.measured is None)!=(args.measured_out is None):
        p.error('Specify both measured paths together')
    torch.set_num_threads(2)
    run(args)
