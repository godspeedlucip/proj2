from __future__ import annotations
import json
import copy
import os
from pathlib import Path
import shutil
import h5py
import numpy as np
import nibabel as nib
import tifffile
from scipy.ndimage import affine_transform
from .contracts import validate_case, write_case, write_manifest, validate_metadata
from ..utils.io import read_jsonl, read_json, resolve, sha256, safe_id, canonical_hash


def resample_pair(arrays,meta,config):
    if set(config)!={'shape','spacing'}: raise ValueError('resample requires only shape and spacing')
    shape=np.asarray(config['shape'],dtype=int); spacing=np.asarray(config['spacing'],dtype=float)
    if shape.shape!=(3,) or spacing.shape!=(3,) or np.any(shape<1) or np.any(spacing<=0): raise ValueError('Invalid common resampling grid')
    old_shape=np.asarray(arrays['cbct'].shape[1:]); old_spacing=np.asarray(meta.get('geometry',{}).get('voxel_spacing_zyx_mm',meta['output_spacing_zyx_mm']))
    step=spacing/old_spacing; offset=(old_shape-1)/2-(shape-1)/2*step
    result={}
    for name,array in arrays.items():
        if array is None: result[name]=None; continue
        volumes=array[None] if array.ndim==3 else array
        sampled=np.stack([affine_transform(np.asarray(v,dtype=np.float32),np.diag(step),offset=offset,
            output_shape=tuple(shape),order=0 if name=='roi_labels' else 1,mode='constant',cval=0,prefilter=False) for v in volumes])
        if name in ('valid_mask','prior_valid_mask'): sampled=(sampled>=1-1e-6).astype(np.uint8)
        if name=='roi_labels': sampled=sampled.astype(np.int32)
        result[name]=sampled[0] if array.ndim==3 else sampled
    if result.get('target') is not None: result['target'][result['valid_mask']==0]=0
    updated=copy.deepcopy(meta); updated['output_spacing_zyx_mm']=spacing.tolist()
    updated['resampling']={'source_shape':old_shape.tolist(),'source_spacing':old_spacing.tolist(),
        'shape':shape.tolist(),'spacing':spacing.tolist(),'source_step':step.tolist(),'source_offset':offset.tolist()}
    updated['generation_profile']=meta['generation_profile']+'__common_grid_v1'
    updated['full_object_fov']=False  # Resampled/cropped support needs a new explicit physical verification.
    if 'voxel_to_scanner_zyx' in updated.get('geometry',{}):
        transform=np.eye(4); transform[:3,:3]=np.diag(step); transform[:3,3]=offset
        updated['geometry']['voxel_to_scanner_zyx']=(np.asarray(meta['geometry']['voxel_to_scanner_zyx'])@transform).tolist()
        updated['geometry']['shape_zyx']=shape.tolist(); updated['geometry']['voxel_spacing_zyx_mm']=spacing.tolist()
        updated['geometry']['center_zyx_mm']=(np.asarray(updated['geometry']['voxel_to_scanner_zyx'])@np.r_[(shape-1)/2,1])[:3].tolist()
    return result,updated


def read_image(source,base):
    if isinstance(source,list): return np.stack([read_image(s,base) for s in source])
    if isinstance(source,dict):
        p=resolve(source['path'],base)
        with h5py.File(p,'r') as f: return np.asarray(f[source['dataset']])
    p=resolve(source,base)
    if p.name.endswith(('.nii','.nii.gz')): return np.asarray(nib.load(str(p)).dataobj,dtype=np.float32)
    if p.suffix.lower() in ('.tif','.tiff'): return tifffile.imread(p)
    if p.suffix.lower()=='.npy': return np.load(p,allow_pickle=False)
    raise ValueError(f'Unsupported image source {p}; HDF5 needs path and dataset')


def import_paired(manifest,out):
    manifest=Path(manifest).resolve(); out=Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Import output must be empty')
    out.mkdir(parents=True,exist_ok=True); rows=[]
    for item in read_jsonl(manifest):
        case=safe_id(item['case_id']); destination=out/(case+'.h5')
        if not item.get('animal_id'): raise ValueError('animal_id is required')
        if 'paired_h5' in item:
            source=resolve(item['paired_h5'],manifest.parent)
            validate_case(source,require_target=item.get('require_target',True))
            if item.get('sha256') and sha256(source)!=item['sha256']: raise ValueError('Source checksum mismatch')
            if item.get('resample'):
                from .contracts import metadata
                with h5py.File(source,'r') as f:
                    arrays={k:np.asarray(f[k]) if k in f else None for k in ['cbct','prior','target','valid_mask','roi_labels','prior_valid_mask']}
                    arrays,meta=resample_pair(arrays,metadata(f),item['resample'])
                write_case(destination,arrays['cbct'],arrays['prior'],arrays['target'],arrays['valid_mask'],meta,
                           arrays['roi_labels'],arrays['prior_valid_mask'])
            else:
                temp=destination.with_suffix('.partial'); shutil.copyfile(source,temp); os.replace(temp,destination)
        else:
            if 'metadata' not in item or 'source_metadata' not in item:
                raise ValueError('Image import requires explicit metadata and source_metadata for cbct/prior/target')
            meta=item['metadata']; meta=read_json(resolve(meta,manifest.parent)) if isinstance(meta,str) else meta
            validate_metadata(meta,len(meta.get('phase_labels',[])))
            contracts=item['source_metadata']; preserved={}
            fields=['phase_labels','axis_order','temporal_topology','output_spacing_zyx_mm','coordinate_convention',
                'geometry','calibration_application','calibration','normalization','intensity_domain','mu_water',
                'generation_profile','target_kind','valid_mask_rule','source_spacing_zyx_mm','used_spacing_zyx_mm',
                'source_shape_zyx','source_affine','source_offOrigin_zyx_mm','settings']
            for name in ['cbct','prior','valid_mask']+(['target'] if 'target' in item else []):
                own=contracts.get(name)
                if isinstance(own,str): own=read_json(resolve(own,manifest.parent))
                if not own: raise ValueError(f'Missing spatial/intensity contract for {name}')
                validate_metadata(own,len(meta['phase_labels']))
                for field in fields:
                    if (field in own)!=(field in meta) or canonical_hash(own.get(field))!=canonical_hash(meta.get(field)):
                        raise ValueError(f'Paired contract mismatch: {name}.{field}')
                preserved[name]=copy.deepcopy(own)
            meta=copy.deepcopy(meta); meta['source_metadata']=preserved
            if 'valid_mask' not in item: raise ValueError('Explicit valid_mask required')
            a=read_image(item['cbct'],manifest.parent); prior=read_image(item['prior'],manifest.parent)
            target=read_image(item['target'],manifest.parent) if 'target' in item else None
            mask=read_image(item['valid_mask'],manifest.parent)
            roi=read_image(item['roi_labels'],manifest.parent) if item.get('roi_labels') else None
            prior_mask=read_image(item['prior_valid_mask'],manifest.parent) if item.get('prior_valid_mask') else None
            if item.get('resample'):
                arrays,meta=resample_pair({'cbct':a,'prior':prior,'target':target,'valid_mask':mask,
                                           'roi_labels':roi,'prior_valid_mask':prior_mask},meta,item['resample'])
                a,prior,target,mask,roi,prior_mask=[arrays[k] for k in ['cbct','prior','target','valid_mask','roi_labels','prior_valid_mask']]
            write_case(destination,a,prior,target,mask,meta,roi,prior_mask)
        row={'case_id':case,'animal_id':str(item['animal_id']),'path':str(destination),'sha256':sha256(destination)}
        for key in ['round','series']: 
            if key in item: row[key]=item[key]
        if item.get('projections'):
            source=resolve(item['projections'],manifest.parent); dest=out/(case+'_projections.h5')
            if item.get('projections_sha256') and sha256(source)!=item['projections_sha256']: raise ValueError('Projection source hash mismatch')
            shutil.copyfile(source,dest); row.update(projections=str(dest),projections_sha256=sha256(dest))
        rows.append(row)
    if not rows: raise ValueError('No paired sources')
    write_manifest(out/'manifest.jsonl',rows)
    return out/'manifest.jsonl'
