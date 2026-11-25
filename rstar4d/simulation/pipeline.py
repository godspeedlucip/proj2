from __future__ import annotations
import csv
from dataclasses import asdict, fields
import gc
import json
import os
from pathlib import Path
import traceback
import h5py
import nibabel as nib
import numpy as np
from .reference import SeriesSettings, load_series_settings, load_series_calibration, validate_calibration_round
from .reference import calibrated_volume, make_phase_gt, apply_reconstruction_calibration, TigreTool
from .phase_mapping import build_projection_phase_mapping
from .backend import geometry, project, reconstruct, output_grid, backend_fingerprint
from ..data.contracts import basic_metadata, validate_case, write_manifest, read_manifest
from ..utils.io import read_jsonl, read_json, resolve, safe_id, sha256, canonical_hash, code_fingerprint, atomic_json


def case_settings(row,config):
    source=config.get('settings_csv') or row.get('settings_csv')
    if not source and config.get('settings_root'):
        source=str(Path(config['settings_root'])/f'round{row["round"]}_heart_gt_settings.csv')
    details={}; values=asdict(SeriesSettings())
    if source:
        source=Path(source); table=load_series_settings(source)
        if row['series'] not in table: raise ValueError(f'No settings row for {row["series"]}')
        values.update(asdict(table[row['series']]))
        with source.open(encoding='utf-8-sig',newline='') as f:
            csvrow=next(r for r in csv.DictReader(f) if r['series'].replace('\\','/').strip('/')==row['series'])
        details={'csv':str(source),'row':csvrow,'ignored_columns':sorted(set(csvrow)-set(values)-{'series'})}
    for extra in [config.get('settings',{}),row.get('settings',{})]:
        unknown=set(extra)-set(values)
        if unknown: raise ValueError(f'Unknown geometry settings: {unknown}')
        values.update(extra)
    settings=SeriesSettings(**values); settings.validate()
    labels=sorted(map(int,row['phase_paths']))
    if settings.min_phase not in labels or settings.max_phase not in labels: raise ValueError('Settings phase endpoints absent from source')
    if settings.dsd_mm<=settings.dso_mm: raise ValueError('DSD must exceed DSO')
    return settings,details


def case_calibration(row,config):
    cal=config['calibration']
    if cal['mode']=='none': return np.poly1d([1.,0.]),{'mode':'none','coefficients':None}
    source=cal.get('csv')
    if not source and cal.get('root'): source=str(Path(cal['root'])/f'round{row["round"]}_heart_calibration.csv')
    if not source: raise ValueError('post_fdk requires calibration CSV')
    path=Path(source); validate_calibration_round(path,Path(f'AHFP-Round-{row["round"]}'))
    polynomial,_,_,details=load_series_calibration(path,row['series'])
    details.update(mode='post_fdk',coefficients=polynomial.coeffs.tolist())
    return polynomial,details


def read_raw_manifest(manifest,case_ids=None,limit=None):
    manifest=Path(manifest).resolve(); rows=read_jsonl(manifest); ids=set()
    if not rows: raise ValueError('Empty raw manifest')
    for row in rows:
        safe_id(row['case_id'])
        if row['case_id'] in ids: raise ValueError('Duplicate case ID')
        ids.add(row['case_id'])
    if case_ids:
        requested=set(case_ids)
        if requested-ids: raise ValueError(f'Unknown cases: {sorted(requested-ids)}')
        rows=[r for r in rows if r['case_id'] in requested]
    if limit is not None:
        if not isinstance(limit,int) or limit<1: raise ValueError('limit must be a positive integer')
        rows=rows[:limit]
    for row in rows:
        if not row.get('animal_id'): raise ValueError(f'{row["case_id"]}: explicit animal_id/identity_map required')
        if len(set(map(int,row['phase_paths'])))!=len(row['phase_paths']): raise ValueError('Duplicate normalized phase labels')
        row['phase_paths']={str(int(k)):str(resolve(v,manifest.parent)) for k,v in row['phase_paths'].items()}
        if row.get('settings_csv'): row['settings_csv']=str(resolve(row['settings_csv'],manifest.parent))
        if len(row['phase_paths'])<2: raise ValueError('Need multiple CT phases')
    return rows


def validate_settings(config,manifest):
    rows=read_raw_manifest(manifest,config.get('case_ids'),config.get('limit')); summaries=[]
    if not config.get('cycle_csv'): raise ValueError('cycle_csv is required')
    for row in rows:
        settings,details=case_settings(row,config); _,cal=case_calibration(row,config)
        labels=sorted(map(int,row['phase_paths']))
        mapping=build_projection_phase_mapping(config['cycle_csv'],len(labels),min_phase=settings.min_phase,
            max_phase=settings.max_phase,phase_labels=labels,n_projections=settings.n_projections,angle_span_deg=settings.angle_span_deg)
        if [r.projection_index for r in mapping]!=list(range(settings.n_projections)): raise ValueError('Invalid view mapping')
        counts={str(label):sum(r.phase_label==label for r in mapping) for label in labels}
        if min(counts.values())<2: raise ValueError('FDK phase has fewer than two views')
        first=nib.load(row['phase_paths'][str(labels[0])]); shape=first.shape; spacing=first.header.get_zooms()[:3]
        for path in row['phase_paths'].values():
            other=nib.load(path)
            if other.shape!=shape or not np.allclose(other.header.get_zooms()[:3],spacing,rtol=0,atol=1e-5) or not np.allclose(other.affine,first.affine,rtol=0,atol=1e-5):
                raise ValueError(f'{row["case_id"]}: inconsistent source phase grids')
        summaries.append({'case_id':row['case_id'],'settings':asdict(settings),'source_shape':list(shape),
            'source_spacing':list(map(float,spacing)),'view_counts':counts,'calibration':cal,'settings_source':details})
    return summaries


def _supports_fit(mu,geo,grid):
    lower=np.array(grid['center_zyx_mm'])-np.array(grid['shape_zyx'])*grid['voxel_spacing_zyx_mm']/2
    upper=np.array(grid['center_zyx_mm'])+np.array(grid['shape_zyx'])*grid['voxel_spacing_zyx_mm']/2
    foreground=mu>0
    for axis in range(3):
        indices=np.flatnonzero(foreground.any(axis=tuple(i for i in range(3) if i!=axis)))
        if not len(indices): continue
        limits=(np.array([indices[0]-.5,indices[-1]+.5])-(geo.nVoxel[axis]-1)/2)*geo.dVoxel[axis]+geo.offOrigin[axis]
        if limits[0]<lower[axis]-1e-5 or limits[1]>upper[axis]+1e-5: return False
    return True


def simulate_case(row,config,out,backend):
    case=safe_id(row['case_id']); folder=out/case; partial=out/(case+'.partial')
    settings,settings_details=case_settings(row,config); polynomial,cal=case_calibration(row,config)
    labels=sorted(map(int,row['phase_paths'])); cycle=Path(config['cycle_csv'])
    source_hashes={str(label):sha256(row['phase_paths'][str(label)]) for label in labels}
    effective_config={k:v for k,v in config.items() if k not in ('limit','case_ids')}
    signature={'source_hashes':source_hashes,'settings':asdict(settings),'calibration':cal,'cycle_sha256':sha256(cycle),
        'identity':{k:row.get(k) for k in ['case_id','animal_id','round','series']},
        'config':effective_config,'backend':backend,'implementation':code_fingerprint()}
    cache_key=canonical_hash(signature)
    if folder.exists():
        done=read_json(folder/'complete.json')
        if done['cache_key']!=cache_key: raise FileExistsError(f'{case}: different cached settings/source; choose a new output directory')
        record=done['record']; record['path']=str(folder/'paired.h5')
        if sha256(record['path'])!=record['sha256']: raise ValueError('Cached paired data corrupted')
        if record.get('projections'):
            record['projections']=str(folder/'projections.h5')
            if sha256(record['projections'])!=record['projections_sha256']: raise ValueError('Cached projections corrupted')
        return record
    if partial.exists(): raise FileExistsError(f'Interrupted case remains at {partial}; inspect before retrying')
    partial.mkdir(parents=True)
    try:
        first=nib.load(row['phase_paths'][str(labels[0])]); source_shape=first.shape
        source_spacing=tuple(map(float,first.header.get_zooms()[:3])); geo=geometry(source_shape,source_spacing,settings)
        angles=TigreTool.make_angles(settings.n_projections,0,settings.angle_span_deg)
        mapping=build_projection_phase_mapping(cycle,len(labels),min_phase=settings.min_phase,max_phase=settings.max_phase,
            phase_labels=labels,n_projections=settings.n_projections,angle_span_deg=settings.angle_span_deg)
        state=np.array([m.phase_label for m in mapping],dtype=np.int32)
        if len(state)!=len(angles) or any(np.sum(state==label)<2 for label in labels): raise ValueError('Incomplete/empty phase mapping')
        mixed=np.empty((len(angles),*geo.nDetector),dtype=np.float32)
        grid=output_grid(geo,settings); shape=tuple(grid['shape_zyx']); fits=True
        paired=partial/'paired.h5'
        with h5py.File(paired,'w') as f:
            chunks=(1,min(shape[0],16),min(shape[1],128),min(shape[2],128))
            for name,dtype in [('target',np.float32),('cbct',np.float32),('valid_mask',np.uint8)]:
                f.create_dataset(name,shape=(len(labels),*shape),dtype=dtype,chunks=chunks,compression='gzip',compression_opts=1)
            for t,label in enumerate(labels):
                print(json.dumps({'case':case,'phase':label,'step':'project_gt'},ensure_ascii=False),flush=True)
                image=nib.load(row['phase_paths'][str(label)])
                if image.shape!=source_shape or not np.allclose(image.affine,first.affine,rtol=0,atol=1e-5): raise ValueError('Source phase grid mismatch')
                mu=calibrated_volume(image.dataobj,np.poly1d([1,0]),0,0,apply_air_threshold=False)
                fits=fits and _supports_fit(mu,geo,grid)
                p=project(mu,geo,angles,config['view_batch']); mixed[state==label]=p[state==label]
                gt,spacing=reconstruct(p,geo,angles,settings)
                sampled,mask_spacing,valid=make_phase_gt(image.dataobj,source_spacing,settings,np.poly1d([1,0]),0,0,return_valid_mask=True)
                if gt.shape!=shape or valid.shape!=shape or not np.allclose(spacing,mask_spacing,rtol=0,atol=1e-5): raise ValueError('FDK and GT validity grids disagree')
                gt[valid==0]=0
                if cal['mode']=='post_fdk': gt=apply_reconstruction_calibration(gt,polynomial)
                f['target'][t]=gt; f['valid_mask'][t]=valid
                del mu,p,gt,sampled,valid,image; gc.collect()
            if config['noise']['kind']=='poisson':
                rng=np.random.default_rng(np.random.SeedSequence([config['seed'],int(cache_key[:8],16)]))
                photons=config['noise']['photons']; counts=rng.poisson(photons*np.exp(-mixed.astype(np.float64)))
                mixed=(-np.log(np.maximum(counts,1)/photons)).astype(np.float32)
            prior=np.zeros(shape,dtype=np.float32) if config['prior_mode']=='phase_weighted_fdk' else None
            for t,label in enumerate(labels):
                image,_=reconstruct(mixed[state==label],geo,angles[state==label],settings)
                if prior is not None: prior+=image*(int(np.sum(state==label))/len(state))
                if cal['mode']=='post_fdk': image=apply_reconstruction_calibration(image,polynomial)
                f['cbct'][t]=image
            if prior is None: prior,_=reconstruct(mixed,geo,angles,settings)
            if cal['mode']=='post_fdk': prior=apply_reconstruction_calibration(prior,polynomial)
            f.create_dataset('prior',data=prior,chunks=chunks[1:],compression='gzip',compression_opts=1)
            meta=basic_metadata(len(labels),spacing,phase_labels=labels,
                intensity_domain='calibrated_mu' if cal['mode']=='post_fdk' else 'mu',
                calibration_application=cal['mode'],calibration=cal,source_hashes=source_hashes,
                source_spacing_zyx_mm=list(source_spacing),source_shape_zyx=list(source_shape),
                source_affine=first.affine.tolist(),used_spacing_zyx_mm=geo.dVoxel.tolist(),
                source_offOrigin_zyx_mm=geo.offOrigin.tolist(),geometry=grid,settings=asdict(settings),
                settings_source=settings_details,generation_profile=config['profile'],prior_mode=config['prior_mode'],
                valid_mask_rule='all_interpolation_sources_valid_and_inside_output_fov',
                full_object_fov=bool(config['full_object_fov'] and fits),full_fov_verified_from_source_support=bool(fits),
                noise=config['noise'],simulation_seed=config['seed'],cycle_sha256=sha256(cycle),cache_key=cache_key,
                backend=backend,implementation=signature['implementation'])
            f.attrs['metadata_json']=json.dumps(meta,ensure_ascii=False,allow_nan=False)
        validate_case(paired)
        record={'case_id':case,'animal_id':str(row['animal_id']),'round':row.get('round'),'series':row.get('series'),
            'path':str(folder/'paired.h5'),'sha256':sha256(paired)}
        if config['save_projections']:
            projection_file=partial/'projections.h5'
            with h5py.File(projection_file,'w') as f:
                f.create_dataset('mixed_line_integral',data=mixed,chunks=(1,*geo.nDetector),compression='gzip',compression_opts=1)
                for name,data in {'angles_rad':angles,'state_label':state,'cycle':[m.cycle for m in mapping],
                    'projection_index':[m.projection_index for m in mapping],'branch':[0 if m.branch=='min_to_max' else 1 for m in mapping]}.items(): f.create_dataset(name,data=data)
                f.attrs['geometry_json']=json.dumps(grid); f.attrs['branch_labels']='0=min_to_max,1=max_to_min'
            record.update(projections=str(folder/'projections.h5'),projections_sha256=sha256(projection_file))
        atomic_json(partial/'complete.json',{'cache_key':cache_key,'record':record,'provenance':signature})
        os.replace(partial,folder)
        return record
    except Exception as exc:
        atomic_json(partial/'error.json',{'error':str(exc),'traceback':traceback.format_exc()})
        raise


def simulate_manifest(manifest,config,out):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    context=out/'generation_context.json'; manifest_sha=sha256(manifest)
    if context.exists():
        if read_json(context)['raw_manifest_sha256']!=manifest_sha: raise ValueError('Output belongs to another raw manifest; choose a new output directory')
    elif any(out.iterdir()): raise FileExistsError('Simulation output is not an owned generation directory; cannot mix imported/generated data')
    else: atomic_json(context,{'kind':'simulation','raw_manifest_sha256':manifest_sha})
    status={'complete':False,'config':config,'requested_case_ids':[], 'completed_case_ids':[],
        'request_id':canonical_hash({'manifest':manifest_sha,'config':config})}
    atomic_json(out/'generation_summary.json',status)
    try:
        rows=read_raw_manifest(manifest,config.get('case_ids'),config.get('limit'))
        if not rows: raise ValueError('No cases selected')
        status['requested_case_ids']=[r['case_id'] for r in rows]
        atomic_json(out/'generation_summary.json',status)
        if not config.get('cycle_csv'): raise ValueError('cycle_csv required')
        backend=backend_fingerprint(); status['backend']=backend
        existing=read_manifest(out/'manifest.jsonl') if (out/'manifest.jsonl').exists() else []
        records={r['case_id']:r for r in existing}
        for row in rows:
            records[row['case_id']]=simulate_case(row,config,out,backend)
            write_manifest(out/'manifest.jsonl',[records[k] for k in sorted(records)])
            status['completed_case_ids'].append(row['case_id']); status['case_count']=len(records)
            atomic_json(out/'generation_summary.json',status)
        status['complete']=True; atomic_json(out/'generation_summary.json',status)
    except Exception as exc:
        status['error']=str(exc); atomic_json(out/'generation_summary.json',status)
        raise
    return out/'manifest.jsonl'
