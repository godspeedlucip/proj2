from __future__ import annotations
import csv
import re
from pathlib import Path
import nibabel as nib
import numpy as np
from ..utils.io import atomic_json, atomic_jsonl, read_json, safe_id


def inventory_pig(data_root,out,settings_root=None,identity_map=None):
    data_root=Path(data_root).resolve(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    identities={}
    if identity_map:
        with Path(identity_map).open(encoding='utf-8-sig',newline='') as f:
            identities={r['series_relative_path'].replace('\\','/'):r['animal_id'] for r in csv.DictReader(f)}
    grouped={}
    for p in sorted(data_root.rglob('*_ct.nii.gz')):
        relative=p.relative_to(data_root)
        if not any('with_contrast' in s.lower() for s in relative.parts[:-1]): continue
        match=re.fullmatch(r'IM(\d+)_ct.nii.gz',p.name,re.I)
        if not match: raise ValueError(f'Unexpected phase filename: {p}')
        phases=grouped.setdefault(p.parent,{})
        label=int(match.group(1))
        if label in phases: raise ValueError('Duplicate phase label')
        phases[label]=p
    rows=[]; issues=[]
    for directory,phases in grouped.items():
        relative=directory.relative_to(data_root); labels=sorted(phases)
        round_parts=[part for part in relative.parts if re.fullmatch(r'AHFP-Round-\d+',part,re.I)]
        if len(round_parts)!=1: raise ValueError(f'Cannot determine round for {relative}')
        round_id=int(round_parts[0].split('-')[-1]); round_pos=relative.parts.index(round_parts[0])
        series='/'.join(relative.parts[round_pos+1:]); rel=relative.as_posix()
        ids=set(m.group(1) for part in relative.parts for m in [re.search(r'_(\d{4,})$',part)] if m)
        animal=identities.get(rel) or (next(iter(ids)) if len(ids)==1 else None)
        if animal is None: issues.append({'series_relative_path':rel,'reason':'ambiguous animal_id','candidates':sorted(ids)})
        shapes=[]; spacings=[]; affines=[]
        for label in labels:
            image=nib.load(str(phases[label])); shapes.append(list(image.shape)); spacings.append(list(map(float,image.header.get_zooms()[:3]))); affines.append(image.affine.tolist())
        if any(s!=shapes[0] for s in shapes) or any(not np.allclose(s,spacings[0],rtol=0,atol=1e-5) for s in spacings):
            issues.append({'series_relative_path':rel,'reason':'phase grid mismatch'})
        case='round%d__%s'%(round_id,re.sub(r'[^\w.\-]','__',series))
        row={'case_id':safe_id(case),'animal_id':animal,'round':round_id,'series':series,'series_relative_path':rel,
             'phase_paths':{str(k):str(phases[k]) for k in labels},'phase_labels':labels,'source_shape_zyx':shapes[0],
             'source_spacing_zyx_mm':spacings[0],'source_affine':affines[0]}
        masks = {}
        for path in sorted(directory.iterdir()):
            match = re.fullmatch(r'IM(\d+)_mask.nii.gz', path.name, re.I)
            if match and int(match.group(1)) in phases:
                label = str(int(match.group(1)))
                if label in masks:
                    raise ValueError(f'Duplicate mask for {case} phase {label}')
                masks[label] = str(path)
        row.update(mask_paths=masks,mask_phase_labels=sorted(map(int,masks)),mask_count=len(masks))
        if settings_root:
            p=Path(settings_root)/f'round{round_id}_heart_gt_settings.csv'
            row['settings_csv']=str(p.resolve())
            if not p.exists(): issues.append({'series_relative_path':rel,'reason':'missing settings csv'})
            else:
                with p.open(encoding='utf-8-sig',newline='') as f: matches=[r for r in csv.DictReader(f) if r['series'].replace('\\','/').strip('/')==series]
                if len(matches)!=1: issues.append({'series_relative_path':rel,'reason':'missing/duplicate settings row'})
        rows.append(row)
    if not rows: raise ValueError('No contrast CT series discovered')
    atomic_jsonl(out/'series.jsonl',rows)
    summary={'series_count':len(rows),'animal_count':len({r['animal_id'] for r in rows if r['animal_id']}),
        'mask_count':sum(r['mask_count'] for r in rows),'phase_count':sum(len(r['phase_labels']) for r in rows),
        'rounds':{str(i):{'series':sum(r['round']==i for r in rows),'phase_counts':sorted({len(r['phase_labels']) for r in rows if r['round']==i})} for i in sorted({r['round'] for r in rows})},'issues':issues}
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'split_template.json',{'train':sorted({r['animal_id'] for r in rows if r['animal_id']}),'val':[],'test':[]})
    return summary
