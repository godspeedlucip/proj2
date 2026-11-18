from __future__ import annotations
import json
import os
from pathlib import Path
import h5py
import numpy as np
from ..utils.io import read_json, read_jsonl, atomic_jsonl, resolve, sha256, canonical_hash, safe_id

SCHEMA_VERSION=1


def basic_metadata(t,spacing,**updates):
    meta={'schema_version':SCHEMA_VERSION,'phase_labels':list(range(1,t+1)),
        'intensity_domain':'mu','mu_water':.02,'normalization':{'scale':.02,'offset':0.},
        'coordinate_convention':'pig_source_array','axis_order':'TZYX',
        'output_spacing_zyx_mm':list(spacing),'temporal_topology':'ordered_states',
        'target_kind':'single_phase_full_angle_fdk_v1','calibration_application':'none',
        'calibration':{'mode':'none','coefficients':None},'source_hashes':{},
        'geometry':{},'full_object_fov':False,'generation_profile':'explicit_paired',
        'valid_mask_rule':'explicit_mask'}
    meta.update(updates)
    return meta


def metadata(f):
    if 'metadata_json' not in f.attrs: raise ValueError('HDF5 missing metadata_json')
    return json.loads(f.attrs['metadata_json'])


def validate_metadata(meta,t):
    required={'schema_version','phase_labels','intensity_domain','normalization','coordinate_convention',
        'output_spacing_zyx_mm','temporal_topology','target_kind','calibration_application','calibration',
        'geometry','source_hashes','axis_order','generation_profile','valid_mask_rule'}
    missing=required-set(meta)
    if missing: raise ValueError(f'Missing metadata: {sorted(missing)}')
    if meta['schema_version']!=1: raise ValueError('Unsupported data schema')
    labels=meta['phase_labels']
    if len(labels)!=t or len(set(labels))!=t or any(not isinstance(i,int) for i in labels): raise ValueError('Invalid phase labels')
    if t<2: raise ValueError('At least two phases required')
    if meta['intensity_domain'] not in ('mu','calibrated_mu','hu','legacy'): raise ValueError('Unknown intensity domain')
    if meta['calibration_application'] not in ('none','post_fdk','legacy_pre_fdk'): raise ValueError('Unknown calibration application')
    spacing=meta['output_spacing_zyx_mm']
    if len(spacing)!=3 or any(not np.isfinite(v) or v<=0 for v in spacing): raise ValueError('Invalid output spacing')
    norm=meta['normalization']
    if set(norm)!={'scale','offset'} or norm['scale']<=0 or not np.isfinite(list(norm.values())).all(): raise ValueError('Invalid normalization')
    if meta['temporal_topology'] not in ('ordered_states','cyclic'): raise ValueError('Unknown temporal topology')
    if meta['axis_order']!='TZYX': raise ValueError('Expected explicit TZYX storage order')
    if not meta['coordinate_convention'] or not meta['generation_profile'] or not meta['valid_mask_rule']:
        raise ValueError('Coordinate/generation/mask provenance is required')
    grid=meta['geometry']; required_grid={'shape_zyx','voxel_spacing_zyx_mm','voxel_to_scanner_zyx','center_zyx_mm'}
    if not isinstance(grid,dict) or not required_grid.issubset(grid): raise ValueError('Missing explicit output grid geometry')
    shape=np.asarray(grid['shape_zyx']); voxel=np.asarray(grid['voxel_spacing_zyx_mm'],dtype=float)
    affine=np.asarray(grid['voxel_to_scanner_zyx'],dtype=float); center=np.asarray(grid['center_zyx_mm'],dtype=float)
    if shape.shape!=(3,) or not np.issubdtype(shape.dtype,np.integer) or np.any(shape<1): raise ValueError('Invalid grid shape')
    if voxel.shape!=(3,) or np.any(voxel<=0) or not np.isfinite(voxel).all(): raise ValueError('Invalid grid voxel spacing')
    if affine.shape!=(4,4) or not np.isfinite(affine).all() or not np.allclose(affine[3],[0,0,0,1]) or abs(np.linalg.det(affine[:3,:3]))<1e-12:
        raise ValueError('Invalid voxel-to-scanner affine')
    if not np.allclose(np.linalg.norm(affine[:3,:3],axis=0),voxel,rtol=1e-5,atol=1e-6): raise ValueError('Affine/voxel spacing mismatch')
    if center.shape!=(3,) or not np.allclose((affine@np.r_[(shape-1)/2,1])[:3],center,atol=1e-5): raise ValueError('Affine/grid center mismatch')
    cal=meta['calibration']; mode=meta['calibration_application']
    if not isinstance(cal,dict) or cal.get('mode')!=mode: raise ValueError('Calibration mode/application mismatch')
    if mode=='none':
        if cal.get('coefficients') is not None or meta['intensity_domain']=='calibrated_mu': raise ValueError('Contradictory uncalibrated intensity contract')
    else:
        coefficients=np.asarray(cal.get('coefficients'),dtype=float)
        if coefficients.ndim!=1 or not len(coefficients) or not np.isfinite(coefficients).all(): raise ValueError('Calibration coefficients required')
        if not any(cal.get(k) for k in ['calibration_sha256','row_sha256','provenance']): raise ValueError('Calibration source provenance required')
    if not isinstance(meta['source_hashes'],dict): raise ValueError('source_hashes must be a mapping')
    # Reject JSON NaN and non-serializable values at the data boundary.
    canonical_hash(meta)


def _datasets(f):
    for name in ['cbct','prior','target','valid_mask','roi_labels','prior_valid_mask']:
        if name in f: yield name,f[name]


def validate_case(path,require_target=True,full=True):
    with h5py.File(path,'r') as f:
        required={'cbct','prior','valid_mask'}|({'target'} if require_target else set())
        if not required.issubset(f): raise ValueError(f'Missing HDF5 arrays: {required-set(f)}')
        shape=f['cbct'].shape
        if len(shape)!=4 or min(shape)<1: raise ValueError('cbct must have nonempty [T,Z,Y,X] shape')
        if f['prior'].shape!=shape[1:]: raise ValueError('prior shape mismatch')
        if 'prior_valid_mask' in f and f['prior_valid_mask'].shape!=shape[1:]:
            raise ValueError('prior_valid_mask shape mismatch')
        for name in ['target','valid_mask','roi_labels']:
            if name in f and f[name].shape!=shape: raise ValueError(f'{name} shape mismatch')
        meta=metadata(f); validate_metadata(meta,shape[0])
        if list(meta['geometry']['shape_zyx'])!=list(shape[1:]): raise ValueError('Array/output grid shape mismatch')
        if full:
            valid_count=0
            for name,a in _datasets(f):
                for i in range(a.shape[0]):
                    # Each phase is read in z slabs; no full 4-D materialization.
                    if a.ndim==4:
                        blocks=(a[i,z:z+16] for z in range(0,a.shape[1],16))
                    else: blocks=[a[i]]
                    for block in blocks:
                        if not np.isfinite(block).all(): raise ValueError(f'{path}: nonfinite {name}')
                        if name in ('valid_mask','prior_valid_mask'):
                            if not np.isin(block,[0,1]).all(): raise ValueError('valid_mask must be binary')
                            if name=='valid_mask': valid_count+=int(block.sum())
                        if name=='roi_labels' and (not np.issubdtype(block.dtype,np.integer) or np.any(block<0)):
                            raise ValueError('ROI labels must be nonnegative integers')
            if valid_count==0: raise ValueError('Case has no valid voxels')
        info = {'shape':list(shape),'metadata':meta}
        if 'cardiac_layout_json' in f.attrs:
            info['cardiac_layout'] = json.loads(f.attrs['cardiac_layout_json'])
        return info


def write_case(path,cbct,prior,target,valid,meta,roi_labels=None,prior_valid_mask=None):
    path=Path(path)
    if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(path.name+'.partial')
    if temp.exists(): raise FileExistsError(f'Partial case exists; inspect before retry: {temp}')
    with h5py.File(temp,'w') as f:
        f.attrs['metadata_json']=json.dumps(meta,ensure_ascii=False,allow_nan=False)
        for name,a in [('cbct',cbct),('prior',prior),('target',target),('valid_mask',valid),
                       ('roi_labels',roi_labels),('prior_valid_mask',prior_valid_mask)]:
            if a is None: continue
            dtype=np.uint8 if name in ('valid_mask','prior_valid_mask') else (np.int32 if name=='roi_labels' else np.float32)
            a=np.asarray(a,dtype=dtype)
            chunks=(1,min(a.shape[1],16),min(a.shape[2],128),min(a.shape[3],128)) if a.ndim==4 else (min(a.shape[0],16),min(a.shape[1],128),min(a.shape[2],128))
            f.create_dataset(name,data=a,chunks=chunks,compression='gzip',compression_opts=1)
    validate_case(temp,require_target=target is not None)
    os.replace(temp,path)


def sample_id(row):
    return row.get('sample_id',row['case_id'])


def read_manifest(path):
    path=Path(path).resolve(); rows=read_jsonl(path)
    if not rows: raise ValueError('Empty manifest')
    seen=set()
    owners={}
    for row in rows:
        safe_id(row['case_id'])
        identifier = sample_id(row)
        if not isinstance(identifier,str):
            raise ValueError('sample_id must be a string')
        row['sample_id'] = safe_id(identifier)
        if identifier in seen: raise ValueError(f'Duplicate sample_id: {identifier}')
        seen.add(identifier)
        if not str(row.get('animal_id','')).strip(): raise ValueError('Missing animal_id')
        owner = owners.setdefault(row['case_id'],str(row['animal_id']))
        if owner!=str(row['animal_id']):
            raise ValueError(f'{row["case_id"]}: samples of one case cannot belong to different animals')
        row['path']=str(resolve(row['path'],path.parent))
        for key in ('projections','annotations','angle_summary'):
            if row.get(key): row[key]=str(resolve(row[key],path.parent))
    return rows


def write_manifest(path,rows):
    path=Path(path).resolve(); output=[]
    for item in rows:
        row=dict(item)
        for key in ['path','projections','annotations','angle_summary']:
            if row.get(key):
                target = Path(row[key]).resolve()
                row[key] = (target.as_posix() if target.drive.casefold()!=path.drive.casefold()
                            else Path(os.path.relpath(target,path.parent)).as_posix())
        output.append(row)
    atomic_jsonl(path,output)


def validate_split(rows,split):
    if isinstance(split,(str,Path)): split=read_json(split)
    if set(split)!={'train','val','test'}: raise ValueError('Split requires train/val/test lists of animal IDs')
    all_ids=[]
    for name,values in split.items():
        if not isinstance(values,list): raise ValueError('Each split must be a list')
        all_ids.extend(str(v) for v in values)
    if len(set(all_ids))!=len(all_ids): raise ValueError('Animal leakage/duplicate in split')
    missing={str(r['animal_id']) for r in rows}-set(all_ids)
    if missing: raise ValueError(f'Animals absent from split: {sorted(missing)}')
    return {k:[str(v) for v in values] for k,values in split.items()}


def validate_manifest(path,split=None,require_target=True,full=True,
                      sidecars=('projections','annotations','angle_summary'),prefer_angle_summary=False):
    rows=read_manifest(path)
    for row in rows:
        if not row.get('sha256') or sha256(row['path'])!=row['sha256']: raise ValueError(f'Data hash mismatch: {row["case_id"]}')
        info=validate_case(row['path'],require_target,full)
        layout = info.get('cardiac_layout')
        if layout:
            for key in ('layout','actual_indices','reference_paired_sha256','source_projections_sha256'):
                if key in row and row[key]!=layout[key]:
                    raise ValueError(f'{sample_id(row)}: manifest {key} disagrees with paired layout provenance')
            row['layout'] = layout['layout']
        row.update(info)
        for kind in sidecars:
            if kind=='projections' and prefer_angle_summary and row.get('angle_summary'):
                continue
            if row.get(kind):
                try:
                    verified=sha256(row[kind])==row.get(kind+'_sha256')
                except OSError as exc:
                    raise ValueError(f'{row["case_id"]}: missing or unreadable {kind} file') from exc
                if not verified:
                    raise ValueError(f'{kind.capitalize()} hash mismatch: {row["case_id"]}')
    if split is not None: validate_split(rows,split)
    domains={(r['metadata']['intensity_domain'],r['metadata']['calibration_application'],canonical_hash(r['metadata']['normalization'])) for r in rows}
    if len(domains)>1: raise ValueError('Mixed intensity/calibration/normalization cohorts')
    return rows


def snapshot(rows,split):
    return {'cases':[{**{k:r.get(k) for k in ['case_id','animal_id','sha256','projections_sha256']},
                      **({'sample_id':sample_id(r)} if sample_id(r)!=r['case_id'] else {}),
                      **{k:r[k] for k in ('layout','annotations_sha256','angle_summary_sha256') if k in r}}
                     for r in sorted(rows,key=sample_id)],
            'split':split}
