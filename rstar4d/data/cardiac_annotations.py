"""Create versioned annotation sidecars on the full FDK output grid."""
import json
import os
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt

from .contracts import validate_manifest, validate_split, write_manifest
from ..utils.io import atomic_json, canonical_hash, read_jsonl, resolve, sha256


STRUCTURES = ('heart_outer','lv')
VERSION = 'cardiac_surface_v1'


def trusted_surface(foreground, trusted):
    foreground = np.asarray(foreground,dtype=bool)
    trusted = np.asarray(trusted,dtype=bool)
    result = np.zeros_like(foreground)
    for axis in range(3):
        lo = [slice(None)]*3
        hi = lo.copy()
        lo[axis] = slice(None,-1)
        hi[axis] = slice(1,None)
        lo, hi = tuple(lo),tuple(hi)
        both = trusted[lo] & trusted[hi]
        result[lo] |= both & foreground[lo] & ~foreground[hi]
        result[hi] |= both & ~foreground[lo] & foreground[hi]
    return result


def output_to_world(meta, declaration):
    if 'output_to_world' in declaration:
        return np.asarray(declaration['output_to_world'],dtype=float)
    required = {'used_spacing_zyx_mm','source_shape_zyx','source_offOrigin_zyx_mm','source_affine'}
    if not required.issubset(meta):
        raise ValueError('An imported grid needs explicit output_to_world for NIfTI annotations')
    spacing = np.asarray(meta['used_spacing_zyx_mm'])
    scanner_to_source = np.eye(4)
    scanner_to_source[:3,:3] = np.diag(1/spacing)
    scanner_to_source[:3,3] = ((np.asarray(meta['source_shape_zyx'])-1)/2
                               - np.asarray(meta['source_offOrigin_zyx_mm'])/spacing)
    return np.asarray(meta['source_affine']) @ scanner_to_source @ np.asarray(meta['geometry']['voxel_to_scanner_zyx'])


def resample_labels(path, out_to_world, shape):
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all() or not np.equal(data,np.floor(data)).all():
        raise ValueError(f'Expected integer 3D label image: {path}')
    transform = np.linalg.solve(image.affine,out_to_world)
    return affine_transform(data,transform[:3,:3],offset=transform[:3,3],output_shape=shape,
                            order=0,mode='constant',cval=0,prefilter=False)


def prepare_annotations(record, inventory_row, declarations, out_path, half_width_vox=2.):
    meta = record['metadata']
    phases = meta['phase_labels']
    shape = tuple(record['shape'][1:])
    effective = {}
    for (label,structure),row in declarations.items():
        row = dict(row)
        if row['available'] and not row.get('mask'):
            candidate = inventory_row.get('mask_paths',{}).get(str(label))
            if candidate:
                row['mask'] = candidate
        for field in ('mask','trusted_domain','correspondence_domain'):
            if row.get(field):
                row[field] = Path(row[field]).resolve()
        effective[label,structure] = row
    declarations = effective
    source_hashes = {str(path): sha256(path) for row in declarations.values() for path in row.values()
                     if isinstance(path,Path)}
    provenance = {'case_id':record['case_id'],'animal_id':str(record['animal_id']),
                  'phase_labels':phases,'structures':list(STRUCTURES),'label_mapping':{'heart_outer':[1,2],'lv':[2]},
                  'distance_unit':'output_voxel','half_width_vox':half_width_vox,'version':VERSION,
                  'grid':meta['geometry'],'paired_sha256':record['sha256'],'source_hashes':source_hashes,
                  'declarations':{str(key):{k:str(v) if isinstance(v,Path) else v for k,v in row.items()}
                                  for key,row in declarations.items()}}
    key = canonical_hash(provenance)
    out_path = Path(out_path)
    if not out_path.exists():
        partial = out_path.with_name(out_path.name+'.partial')
        with h5py.File(record['path'],'r') as paired, h5py.File(partial,'x') as dest:
            dest.attrs['metadata_json'] = json.dumps({**provenance,'cache_key':key},ensure_ascii=False)
            datasets = {}
            for name in ('surface','distance','domain','correspondence','region'):
                datasets[name] = dest.create_dataset(name, shape=(len(phases),2,*shape),
                    dtype=np.float32 if name == 'distance' else np.uint8,
                    chunks=(1,1,*[min(n,k) for n,k in zip(shape,(16,128,128))]), compression='gzip', compression_opts=1)
            available = np.zeros((len(phases),2),dtype=np.uint8)
            for ti,label in enumerate(phases):
                fov = np.asarray(paired['valid_mask'][ti],dtype=bool)
                for si,name in enumerate(STRUCTURES):
                    declaration = declarations.get((label,name))
                    if declaration is None or not declaration['available']:
                        continue
                    mask = declaration.get('mask')
                    matrix = None
                    if mask or declaration.get('trusted_domain') or declaration.get('correspondence_domain'):
                        matrix = output_to_world(meta,declaration)
                    if mask:
                        labels = resample_labels(mask,matrix,shape)
                    elif 'roi_labels' in paired:
                        labels = paired['roi_labels'][ti]
                    else:
                        raise ValueError(f'{record["case_id"]}/{label}/{name}: confirmed annotation needs a mask')
                    trust_path = declaration.get('trusted_domain')
                    domain = fov.copy()
                    if trust_path:
                        domain &= resample_labels(trust_path,matrix,shape) != 0
                    elif declaration.get('whole_fov_confirmed') is not True:
                        raise ValueError(f'{record["case_id"]}/{label}/{name}: explicit trusted domain required')
                    if mask:
                        image = nib.load(str(mask))
                        transform = np.linalg.solve(image.affine,matrix)
                        domain &= affine_transform(np.ones(image.shape,np.uint8),transform[:3,:3],
                            offset=transform[:3,3],output_shape=shape,order=0,prefilter=False) != 0
                    if not np.isin(labels[domain],(0,1,2)).all():
                        raise ValueError(f'{record["case_id"]}/{label}/{name}: unknown label value in trusted domain')
                    region = np.isin(labels,provenance['label_mapping'][name])
                    surface = trusted_surface(region,domain)
                    if not surface.any():
                        continue
                    available[ti,si] = 1
                    datasets['surface'][ti,si] = surface
                    datasets['region'][ti,si] = region & domain
                    datasets['distance'][ti,si] = distance_transform_edt(~surface)
                    datasets['domain'][ti,si] = domain
                    pair_path = declaration.get('correspondence_domain')
                    if pair_path:
                        datasets['correspondence'][ti,si] = domain & (resample_labels(pair_path,matrix,shape)!=0)
            dest.create_dataset('annotation_available',data=available)
            dest.flush()
        os.replace(partial,out_path)
    with h5py.File(out_path,'r') as saved:
        if json.loads(saved.attrs['metadata_json'])['cache_key'] != key:
            raise ValueError(f'{record["case_id"]}: stale annotation cache at {out_path}')
        available = saved['annotation_available'][:]
    return {'annotations':str(out_path.resolve()),'annotations_sha256':sha256(out_path),
            'annotation_available':available.tolist()}


def prepare_manifest(inventory, manifest, annotations, out, half_width_vox=2., split=None):
    if not np.isfinite(half_width_vox) or half_width_vox<=0:
        raise ValueError('Annotation half-width must be positive')
    rows = validate_manifest(manifest, sidecars=())
    if split:
        split = validate_split(rows,split)
    inventories = {row['case_id']:row for row in read_jsonl(inventory)}
    annotation_path = Path(annotations).resolve()
    cases = {row['case_id']:row for row in rows}
    declarations = {case:{} for case in cases}
    for row in read_jsonl(annotation_path):
        case,label,structure = (row[k] for k in ('case_id','phase_label','structure'))
        if (case not in cases or str(row['animal_id']) != str(cases[case]['animal_id'])
                or label not in cases[case]['metadata']['phase_labels'] or structure not in STRUCTURES
                or type(row['available']) is not bool):
            raise ValueError(f'Annotation identity or availability is invalid: {case}/{label}/{structure}')
        if (label,structure) in declarations[case]:
            raise ValueError(f'Duplicate annotation: {case}/{label}/{structure}')
        for field in ('mask','trusted_domain','correspondence_domain'):
            if row.get(field):
                row[field] = resolve(row[field],annotation_path.parent)
        if row['available'] and not row.get('mask'):
            path = inventories.get(case,{}).get('mask_paths',{}).get(str(label))
            if path:
                row['mask'] = resolve(path,Path(inventory).resolve().parent)
        declarations[case][label,structure] = row
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    coverage = []
    for row in rows:
        case = row['case_id']
        declared = declarations[case]
        if declared:
            row.update(prepare_annotations(row,inventories.get(case,{}),declared,out/(case+'.h5'),half_width_vox))
        coverage.append({**{k:row.get(k) for k in ('case_id','animal_id','annotation_available')},
                         'phase_labels':row['metadata']['phase_labels'],
                         'raw_mask_phases':inventories.get(case,{}).get('mask_phase_labels',[])})
    groups = {'all':coverage} if split is None else {
        key:[r for r in coverage if str(r['animal_id']) in animals] for key,animals in split.items()}
    counts = {}
    for key,group in groups.items():
        counts[key] = {'cases':len(group),'animals':len({str(r['animal_id']) for r in group}),
                       'raw_mask_states':sum(len(r['raw_mask_phases']) for r in group),
                       'states':sum(len(r['phase_labels']) for r in group)}
        for index,name in enumerate(STRUCTURES):
            counts[key][name+'_states'] = sum(sum(state[index] for state in (r['annotation_available'] or []))
                                             for r in group)
    atomic_json(out/'coverage.json',{'structures':STRUCTURES,'cases':coverage,'counts':counts})
    write_manifest(out/'manifest.jsonl',rows)
    return out/'manifest.jsonl'
