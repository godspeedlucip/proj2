"""Control one phase's sparse reconstruction using subsets of the same mixed scan."""
from pathlib import Path
import json
import shutil

import h5py
import numpy as np

from .backend import geometry, reconstruct
from .reference import SeriesSettings, apply_reconstruction_calibration
from ..data.acquisition import angle_summary, ENCODING
from ..data.contracts import validate_manifest, write_manifest
from ..utils.io import atomic_json, canonical_hash, sha256


def layout_summaries(angles, labels, phases, phase, indices_a, indices_b):
    if len(indices_a)!=len(indices_b) or len(indices_a)<2:
        raise ValueError('Layouts need equal counts of at least two phase views')
    summaries = [angle_summary(angles,labels,phases,selected_indices=indices,selected_phase=phase)
                 for indices in (indices_a,indices_b)]
    if np.array_equal(summaries[0],summaries[1]):
        raise ValueError('Select two layouts with distinguishable geometry summaries')
    return summaries


def angle_layouts(manifest, case_id, phase, indices_a, indices_b, out):
    rows = validate_manifest(manifest,full=False,sidecars=('projections','annotations'))
    candidates = [item for item in rows if item['case_id']==case_id]
    if len(candidates)>1:
        raise ValueError(f'{case_id}: choose a single original acquisition for layout generation')
    row = next(iter(candidates),None)
    if row is None or not row.get('projections'):
        raise ValueError(f'{case_id}: required mixed projections are missing')
    meta = row['metadata']
    settings = SeriesSettings(**meta['settings'])
    geo = geometry(meta['source_shape_zyx'],meta['source_spacing_zyx_mm'],settings)
    labels = meta['phase_labels']
    target = labels.index(phase)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Layout output directory must be empty: {out}')
    manifests = {}
    joint = []
    with h5py.File(row['projections'],'r') as projections:
        angles = projections['angles_rad'][:]
        state = projections['state_label'][:]
        correct = layout_summaries(angles,state,labels,phase,indices_a,indices_b)
        for i,(name,indices) in enumerate((('A',indices_a),('B',indices_b))):
            position = sorted(indices)
            case_path = out/name
            case_path.mkdir(parents=True)
            paired = case_path/'paired.h5'
            reconstructed,_ = reconstruct(projections['mixed_line_integral'][position],geo,angles[position],settings)
            if meta['calibration_application']=='post_fdk':
                reconstructed = apply_reconstruction_calibration(reconstructed,np.poly1d(meta['calibration']['coefficients']))
            shutil.copyfile(row['path'],paired)
            provenance = {'reference_paired_sha256':row['sha256'],
                'reference_annotations_sha256':row.get('annotations_sha256'),
                'source_projections_sha256':row['projections_sha256'],'phase':phase,
                'actual_indices':position,'layout':name,
                'correct_angles_sha256':canonical_hash(correct[i].tolist())}
            with h5py.File(paired,'r+') as dest:
                dest['cbct'][target] = reconstructed
                dest.attrs['cardiac_layout_json'] = json.dumps(provenance)
            variant = {**row,**provenance,'sample_id':f'{row["sample_id"]}-phase-{phase}-{name}',
                       'cardiac_layout':provenance,'path':str(paired),'sha256':sha256(paired)}
            if row.get('annotations'):
                annotations = case_path/'annotations.h5'
                shutil.copyfile(row['annotations'],annotations)
                with h5py.File(annotations,'r+') as ref:
                    info = json.loads(ref.attrs['metadata_json'])
                    info.update(paired_sha256=variant['sha256'],parent_annotation_sha256=row['annotations_sha256'])
                    ref.attrs['metadata_json'] = json.dumps(info,ensure_ascii=False)
                variant.update(annotations=str(annotations),annotations_sha256=sha256(annotations))
            for swapped in (False,True):
                tag = 'swapped' if swapped else 'correct'
                encoded = 1-i if swapped else i
                summary_path = case_path/f'angles-{tag}.json'
                atomic_json(summary_path,{'encoding':ENCODING,'phase_labels':labels,
                    'values':correct[encoded].tolist(),'projections_sha256':row['projections_sha256'],
                    'paired_sha256':variant['sha256'],
                    'phase':phase,'actual_indices':position,
                    'encoded_indices':sorted((indices_a,indices_b)[encoded]),
                    'common_total_views':len(angles),'source_paired_sha256':row['sha256'],
                    'layout':name,'diagnostic_swap':swapped})
                item = {**variant,'diagnostic_swap':swapped,
                        'angle_summary':str(summary_path),'angle_summary_sha256':sha256(summary_path)}
                path = case_path/f'manifest-{tag}.jsonl'
                write_manifest(path,[item])
                manifests[name+'-'+tag] = str(path)
                if not swapped:
                    joint.append(item)
    joint_path = out/'manifest-joint.jsonl'
    write_manifest(joint_path,joint)
    manifests['joint-correct'] = str(joint_path)
    atomic_json(out/'layouts.json',manifests)
    return manifests
