"""Generate controlled cardiac ablations and summarize paired animal results."""
import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml
import numpy as np

from rstar4d.config import load_config, validate_train
from rstar4d.evaluation.contours import CONTOUR_METRICS, write_csv
from rstar4d.utils.io import atomic_json, read_json, read_jsonl


CONTRASTS = (('MPB','baseline'),('M','baseline'),('MPB','PB'),('MPB','MB'),('MPB','MP'),
             ('P','baseline'),('B','baseline'),('MP','baseline'),('MB','baseline'),('PB','baseline'))


def matrix(base):
    cases = {}
    for m,p,b in product((False,True),repeat=3):
        case = ''.join(name for name,enabled in zip('MPB',(m,p,b)) if enabled) or 'baseline'
        config = deepcopy(base)
        config['extensions']['motion']['mode'] = 'learned' if m else 'off'
        config['extensions']['prior']['mode'] = 'geometry' if p else 'off'
        config['extensions']['boundary']['enabled'] = b
        cases[case] = validate_train(config)
    return cases


def write_matrix(base_yaml, out):
    base = load_config(base_yaml)
    if base['stage2_crop_mode'] not in ('core_with_context','fixed_input'):
        raise ValueError('The eight arms require a shared cardiac crop mode')
    motion = base['extensions']['motion']
    if (motion['surface_weight'] or base['extensions']['boundary']['grad_weight_vox']
            or not motion['struct_weight'] or not motion['smooth_weight']):
        raise ValueError('Use the fixed primary motion and boundary losses for the eight-arm matrix')
    return write_configs(matrix(base),out)


def write_configs(configs, out):
    out.mkdir(parents=True,exist_ok=True)
    files = {}
    for group,config in configs.items():
        path = out/(group+'.yaml')
        path.write_text(yaml.safe_dump(validate_train(config),sort_keys=False),encoding='utf-8')
        files[group] = str(path)
    return files


def write_controls(base_yaml, out, gradient=None, surface=None):
    arms = matrix(load_config(base_yaml))
    controls = {'motion_identity':deepcopy(arms['MPB']),'motion_no_aux':deepcopy(arms['MPB']),
                'prior_image_only':deepcopy(arms['MPB'])}
    controls['motion_identity']['extensions']['motion']['mode'] = 'identity'
    controls['motion_no_aux']['extensions']['motion'].update(struct_weight=0.,smooth_weight=0.)
    controls['prior_image_only']['extensions']['prior']['mode'] = 'image_only'
    if gradient is not None:
        if gradient<=0:
            raise ValueError('The B gradient comparison requires a positive weight')
        controls['boundary_gradient'] = deepcopy(arms['B'])
        controls['boundary_gradient']['extensions']['boundary']['grad_weight_vox'] = gradient
    if surface is not None:
        if surface<=0:
            raise ValueError('The surface comparison requires a positive weight')
        controls['motion_surface'] = deepcopy(arms['MPB'])
        controls['motion_surface']['extensions']['motion']['surface_weight'] = surface
    return write_configs(controls,out)


def _layout(row):
    return row.get('layout') or row.get('cardiac_layout',{}).get('layout')


def _keyed(rows, selected_layout=None):
    result = {}
    for row in rows:
        layout = _layout(row)
        if selected_layout is not None and layout!=selected_layout:
            continue
        # For a declared cross-layout contrast, the selected arms supply the layout dimension.
        key = (str(row.get('animal_id','')),row['case_id'],'' if selected_layout else layout or '',
               row.get('phase_label'))
        if key in result:
            raise ValueError(f'Duplicate evaluation key: {key}')
        result[key] = row
    return result


def _matched_differences(baseline, other, metric, structure, layout_pair):
    new_layout,base_layout = layout_pair or (None,None)
    base = _keyed((r for r in baseline if r['structure']==structure),base_layout)
    new = _keyed((r for r in other if r['structure']==structure),new_layout)
    state_values = defaultdict(list)
    cases = set()
    states = []
    failures = {'missing_key':0,'undefined_value':0}
    for key in sorted(base.keys() | new.keys()):
        if key not in base or key not in new:
            failures['missing_key'] += 1
        elif base[key].get(metric) is None or new[key].get(metric) is None:
            failures['undefined_value'] += 1
        else:
            old,current = base[key],new[key]
            delta = current[metric]-old[metric]
            state_values[key[0],key[1],key[3]].append(delta)
            cases.add(key[:2])
            if key[3] is not None:
                states.append({'animal_id':key[0],'case_id':key[1],'phase_label':key[3],
                               'reference_layout':_layout(old),'candidate_layout':_layout(current),
                               'reference_value':old[metric],'candidate_value':current[metric],'delta':delta})
    animal = defaultdict(list)
    for (name,case,phase),values in state_values.items():
        animal[name].append(float(np.mean(values)))
    differences = {name:float(np.mean(values)) for name,values in animal.items()}
    summary = {'valid_states':sum(phase is not None for _,_,phase in state_values),
            'valid_state_layout_pairs':len(states),'valid_cases':len(cases),'valid_animals':len(differences),
            'mean_delta':float(np.mean(list(differences.values()))) if differences else None,
            'animal_differences':differences,**failures}
    return summary,states


def paired_differences(baseline, other, metric, structure):
    summary,_ = _matched_differences(baseline,other,metric,structure,None)
    return summary


def global_rows(rows, *, by_phase=False):
    result = []
    for row in rows:
        identity = {'case_id':row['case_id'],'animal_id':row['animal_id'],'layout':_layout(row),'structure':'global'}
        for entry in row.get('phases',[]) if by_phase else (row,):
            result.append({**identity,'phase_label':entry.get('phase_label'),
                           **entry.get('regions',{}).get('global',{}).get('prediction',{})})
    return result


def _resolve_contrast(baseline, other, allow_variants, pair):
    if pair is not None:
        if len(pair)!=2 or not all(isinstance(x,str) and x for x in pair):
            raise ValueError('Specify candidate and reference layouts for the contrast')
        for rows,label in zip((other,baseline),pair):
            if not any(_layout(row)==label for row in rows):
                raise ValueError(f'No evaluated samples for requested layout {label}')
        return pair
    if allow_variants:
        left,right = ({_layout(row) for row in rows} for rows in (baseline,other))
        if len(left)==len(right)==1 and left!=right and None not in left|right:
            return (next(iter(right)),next(iter(left)))
        if left!=right:
            raise ValueError('Multiple layouts are present: specify an explicit layout pair')
    return None


def check_common_references(baseline, other, allow_layout_variants=False, layout_pair=None):
    pair = _resolve_contrast(baseline,other,allow_layout_variants,layout_pair)
    new_layout,base_layout = pair or (None,None)
    base,candidate = _keyed(baseline,base_layout),_keyed(other,new_layout)
    for key in base.keys() & candidate.keys():
        a,b = base[key],candidate[key]
        if (a['paired_sha256'],a.get('annotations_sha256')) == (b['paired_sha256'],b.get('annotations_sha256')):
            continue
        if (allow_layout_variants or pair) and all('cardiac_layout' in r for r in (a,b)):
            fields = ('reference_paired_sha256','reference_annotations_sha256','source_projections_sha256','phase')
            left,right = a['cardiac_layout'],b['cardiac_layout']
            if (left.get('reference_paired_sha256') and left.get('source_projections_sha256') and
                    all(field in left and field in right and left[field]==right[field] for field in fields)):
                continue
        raise ValueError(f'{key}: paired groups require the same reference image and annotation provenance')
    return pair


def compare(runs, out, contrasts=CONTRASTS, *, allow_layout_variants=False, layout_pair=None):
    contour_rows = {}
    case_rows = {}
    groups = {}
    protocols = []
    for name in sorted({group for pair in contrasts for group in pair}):
        run_path = runs/name
        target = run_path/'evaluation'
        if not (target/'summary.json').is_file():
            continue
        summary = read_json(target/'summary.json')
        case_rows[name] = read_jsonl(target/'cases.jsonl')
        path = target/'contours_by_phase.jsonl'
        if path.is_file():
            contour_rows[name] = read_jsonl(path)
            # Single-layout reports written before sample IDs stored layout provenance only in cases.jsonl.
            source_layouts = {r.get('sample_id',r['case_id']):_layout(r) for r in case_rows[name]}
            for record in contour_rows[name]:
                record.setdefault('layout',source_layouts[record.get('sample_id',record['case_id'])])
            data = summary['contours']
            if data['delineation'] is not None:
                protocols.append((data['delineation'],summary['metric_config']['contours']))
        log_path = run_path/'train.jsonl'
        epochs = [row for row in read_jsonl(log_path) if row['event']=='epoch'] if log_path.is_file() else []
        groups[name] = {'checkpoint':str(run_path/'checkpoints/best.pt'),'evaluation':summary,
                        'peak_allocated_gib':max((row['peak_allocated_gib'] for row in epochs),default=None),
                        'sampling_sha256_by_epoch':[row.get('sampling_sha256') for row in epochs],
                        'inference_by_case':{r.get('sample_id',r['case_id']):r.get('inference',{}) for r in case_rows[name]}}
        for key in ('initialization','stage2_resolved'):
            path = run_path/(key+'.json')
            if path.is_file():
                groups[name][key] = read_json(path)
    if not groups:
        raise ValueError('No completed evaluation outputs were found')
    if protocols and any(protocol!=protocols[0] for protocol in protocols):
        raise ValueError('Paired contours require a common segmentation/annotation protocol and metric settings')
    summary = []
    animals = []
    phase_summary = []
    phase_details = []
    for new,old in contrasts:
        if new not in groups or old not in groups:
            continue
        pair = check_common_references(case_rows[old],case_rows[new],allow_layout_variants,layout_pair)
        new_layout,old_layout = pair or (None,None)
        old_cases = _keyed(case_rows[old],old_layout)
        new_cases = _keyed(case_rows[new],new_layout)
        complete = all('phases' in old_cases[k] and 'phases' in new_cases[k]
                       for k in old_cases.keys() & new_cases.keys())
        tests = [('global',metric,global_rows(case_rows[old],by_phase=True),
                  global_rows(case_rows[new],by_phase=True)) for metric in ('mae','ssim')]
        for structure,metric in product(('heart_outer','lv'),CONTOUR_METRICS):
            tests.append((structure,metric,contour_rows.get(old,[]),contour_rows.get(new,[])))
        for structure,metric,baseline,candidate in tests:
            item = {'candidate':new,'reference':old,'structure':structure,'metric':metric}
            result,details = _matched_differences(baseline,candidate,metric,structure,pair)
            result['phase_count_status'] = 'known'
            if structure=='global' and not complete:
                result.update(valid_states=None,valid_state_layout_pairs=None,phase_count_status='unknown_legacy')
            phase_summary.append({**item,**{k:v for k,v in result.items() if k!='animal_differences'}})
            phase_details.extend({**item,**row} for row in details)
            if structure=='global':
                case_result,_ = _matched_differences(global_rows(case_rows[old]),global_rows(case_rows[new]),
                                                     metric,structure,pair)
                case_result.update({key:result[key] for key in ('valid_states','valid_state_layout_pairs','phase_count_status')})
                result = case_result
            differences = result.pop('animal_differences')
            summary.append({**item,**result})
            animals.extend({**item,'animal_id':animal,'delta':delta} for animal,delta in differences.items())
    out.mkdir(parents=True,exist_ok=True)
    replay = {tuple(group['sampling_sha256_by_epoch']) for group in groups.values() if group['sampling_sha256_by_epoch']}
    atomic_json(out/'summary.json',{'distance_unit':'output_voxel','groups':groups,'paired':summary,
                                    'layout_pair_candidate_reference':layout_pair,
                                    'paired_phase_summary':phase_summary,'paired_by_phase':phase_details,
                                    'matching_sampling_hashes':len(replay)==1 if replay else None,
                                    'missing_groups':sorted(set(group for pair in contrasts for group in pair)-set(groups))})
    write_csv(out/'paired_summary.csv',summary)
    write_csv(out/'paired_animals.csv',animals)
    write_csv(out/'paired_phase_summary.csv',phase_summary)
    write_csv(out/'paired_by_phase.csv',phase_details)
    return out/'summary.json'


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command',required=True)
    sub = commands.add_parser('write-matrix')
    sub.add_argument('--base',required=True,type=Path)
    sub.add_argument('--out',required=True,type=Path)
    sub = commands.add_parser('write-controls')
    sub.add_argument('--base',required=True,type=Path)
    sub.add_argument('--out',required=True,type=Path)
    sub.add_argument('--grad-weight-vox',type=float)
    sub.add_argument('--surface-weight',type=float)
    sub = commands.add_parser('compare')
    sub.add_argument('--runs',required=True,type=Path)
    sub.add_argument('--out',required=True,type=Path)
    sub.add_argument('--contrast',nargs=2,action='append',metavar=('CANDIDATE','REFERENCE'))
    sub.add_argument('--allow-layout-variants',action='store_true')
    sub.add_argument('--layout-pair',nargs=2,metavar=('CANDIDATE_LAYOUT','REFERENCE_LAYOUT'))
    sub = commands.add_parser('angle-layouts')
    sub.add_argument('--manifest',required=True,type=Path)
    sub.add_argument('--case-id',required=True)
    sub.add_argument('--phase',required=True,type=int)
    sub.add_argument('--indices-a',required=True,type=int,nargs='+')
    sub.add_argument('--indices-b',required=True,type=int,nargs='+')
    sub.add_argument('--out',required=True,type=Path)
    sub = commands.add_parser('diagnose-prior')
    sub.add_argument('--checkpoint',required=True,type=Path)
    sub.add_argument('--manifest',required=True,type=Path)
    sub.add_argument('--out',required=True,type=Path)
    sub.add_argument('--device',default='cpu')
    sub.add_argument('--tolerance',default=.1,type=float)
    args = parser.parse_args()
    if args.command=='write-matrix':
        result = write_matrix(args.base,args.out)
    elif args.command=='write-controls':
        result = write_controls(args.base,args.out,args.grad_weight_vox,args.surface_weight)
    elif args.command=='compare':
        result = compare(args.runs,args.out,args.contrast or CONTRASTS,allow_layout_variants=args.allow_layout_variants,
                         layout_pair=args.layout_pair)
    elif args.command=='angle-layouts':
        from rstar4d.simulation.cardiac_layouts import angle_layouts
        result = angle_layouts(args.manifest,args.case_id,args.phase,args.indices_a,args.indices_b,args.out)
    else:
        from rstar4d.evaluation.prior_diagnostics import diagnose_manifest
        result = diagnose_manifest(args.checkpoint,args.manifest,args.out,args.device,args.tolerance)
    print(result)


if __name__=='__main__':
    main()
