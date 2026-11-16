"""Plan/generate TIGRE measurements using the user's pig simulation contract.

Original study scripts and source CTs are read-only. GT/prior volumes are never
read by generation or training. Generated measurements live in a new case folder.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .data import file_sha256


def acquisition_times(n_views,duration_s):
    if n_views<1 or not np.isfinite(duration_s) or duration_s<=0:
        raise ValueError('positive view count and finite scan duration required')
    return np.arange(n_views,dtype=np.float64)*duration_s/n_views


def hu_to_mu(hu):
    hu=np.asarray(hu,dtype=np.float32)
    if not np.isfinite(hu).all():
        raise ValueError('source HU must be finite')
    result=hu.copy()
    result/=1000.;result+=1.;result*=.02
    np.maximum(result,0.,out=result)
    result[hu==-2048.]=0.
    return result


def grid_spec(shape,spacing,settings):
    shape=np.asarray(shape,dtype=np.int64)
    spacing=np.asarray(spacing,dtype=np.float64)*settings['voxel_spacing_multiplier']
    if shape.shape!=(3,) or spacing.shape!=(3,) or (shape<=0).any() or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('positive finite 3D source grid required')
    shift=np.array([0.,settings['shift_down_voxels']-settings['shift_up_voxels'],
                     settings['shift_right_voxels']-settings['shift_left_voxels']])
    offset=(-shift*spacing).astype(np.float32).astype(np.float64)
    spacing=spacing.astype(np.float32).astype(np.float64)
    size=shape*spacing
    bounds=np.stack((offset-size/2,offset+size/2))[:,::-1]
    return dict(shape_zyx=shape.tolist(),spacing_zyx_mm=spacing.tolist(),size_zyx_mm=size.tolist(),
                offset_zyx_mm=offset.tolist(),bounds_xyz_mm=bounds.tolist())


def evaluation_grid(source,settings):
    """Mirror reconstruct_prior's centered grid, mask, then crop/zoom."""
    detector=np.array([settings['n_detector_rows'],settings['n_detector_cols']])
    pitch=np.array([settings['d_detector_row_mm'],settings['d_detector_col_mm']],dtype=np.float32)
    object_fov=(detector*pitch)*settings['dso_mm']/settings['dsd_mm']
    spacing=np.array(source['spacing_zyx_mm'])
    shape=np.minimum(np.maximum(np.floor(object_fov[[0,1,1]]/spacing).astype(int),1),source['shape_zyx'])
    width=settings['fov_output_pixels']
    if settings.get('fov_source_pixels',0):
        requested=settings['fov_source_pixels']*spacing[1:]
        if (requested>object_fov[1]+1e-6).any():
            raise ValueError('wide GT FOV exceeds detector')
        shape[1:]=width
        spacing[1:]*=settings['fov_source_pixels']/width
    shape=shape.astype(int)
    resize=bool(min(shape[1:])<width)
    output_spacing=spacing.copy()
    if resize:
        output_spacing[1:]*=shape[1:]/width
    return dict(intermediate_shape_zyx=shape.tolist(),spacing_zyx_mm=spacing.tolist(),
                output_shape_zyx=[int(shape[0]),width,width],output_spacing_zyx_mm=output_spacing.tolist(),
                object_fov_vu_mm=object_fov.tolist(),mask_shrink_pixels=settings['fov_mask_shrink_pixels'],
                postprocess='scipy_zoom_order1_grid_mode_false' if resize else 'center_crop',
                center_scanner_zyx_mm=[0.,0.,0.],intensity='raw_mu_mm^-1_no_polynomial')


def study_module(project_root):
    root=Path(project_root).resolve()
    path=root/'generate_round1_contrast_recon.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    module=importlib.import_module('generate_round1_contrast_recon')
    if Path(module.__file__).resolve()!=path:
        raise RuntimeError('another study module is already imported; use a fresh Python process')
    return module


def plan_case(project_root,round_number,series,duration_s=10.):
    import nibabel as nib
    root=Path(project_root).resolve();module=study_module(root)
    round_root=root/'data/pig'/f'AHFP-Round-{int(round_number)}'
    settings_path=root/'data'/f'round{int(round_number)}_contrast_recon_settings.csv'
    cycle_path=root/'data/cycle.csv'
    selected=module.select_series(module.discover_series(round_root),round_root,[series])
    if len(selected)!=1:
        raise ValueError('select one exact relative series path')
    directory,paths=selected[0]
    relative=directory.relative_to(round_root).as_posix()
    settings=asdict(module.load_series_settings(settings_path)[relative])
    first=nib.load(str(next(iter(paths.values()))))
    shape=list(first.shape);spacing=[float(v) for v in first.header.get_zooms()[:3]]
    phase_files={}
    for phase,path in paths.items():
        image=nib.load(str(path))
        if list(image.shape)!=shape or not np.allclose(image.header.get_zooms()[:3],spacing,rtol=0,atol=1e-5):
            raise ValueError(f'phase grid differs: {path}')
        phase_files[str(phase)]=dict(path=str(path.resolve()),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    from utils.phase_mapping import build_projection_phase_mapping
    rows=build_projection_phase_mapping(cycle_path,len(paths),min_phase=settings['min_phase'],
        max_phase=settings['max_phase'],phase_labels=tuple(paths),n_projections=settings['n_projections'],
        angle_span_deg=settings['angle_span_deg'])
    times=acquisition_times(settings['n_projections'],duration_s)
    mapping=[dict(view_index=r.projection_index,phase_label=r.phase_label,cycle=r.cycle,
                  branch=r.branch,angle_deg=r.angle_deg,time_s=float(times[r.projection_index])) for r in rows]
    if [r['view_index'] for r in mapping]!=list(range(settings['n_projections'])):
        raise ValueError('phase mapping must cover each view exactly once')
    source=grid_spec(shape,spacing,settings)
    names=['generate_round1_contrast_gt.py','generate_round1_contrast_recon.py',
           'generate_pig_first_phase_prior.py','utils/tool.py','utils/phase_mapping.py']
    provenance={name:dict(path=str(root/name),sha256=file_sha256(root/name)) for name in names}
    provenance['settings_csv']=dict(path=str(settings_path),sha256=file_sha256(settings_path))
    provenance['cycle_csv']=dict(path=str(cycle_path),sha256=file_sha256(cycle_path))
    return dict(schema='bird_pig_plan_v1',project_root=str(root),round=int(round_number),series=relative,
        phase_files=phase_files,settings=settings,source_grid=source,evaluation_grid=evaluation_grid(source,settings),
        acquisition=dict(duration_s=float(duration_s),n_views=settings['n_projections'],
                         angle_span_deg=settings['angle_span_deg'],time_rule='i*duration/n_views; endpoint excluded',
                         motion_sampling='nearest phase per supplied cycle.csv'),
        mapping=mapping,provenance=provenance,intensity_processing='hu_to_mu_valid_source_only_v1',
        training_inputs=['projections','geometry','chronological_time'],uses_prior_image=False,uses_gt_supervision=False)


def generate_case(plan,outdir,view_indices=None,batch_views=16,gpu=0):
    import nibabel as nib
    from .tigre_backend import make_geometry, project_numpy
    outdir=Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('case directory is not empty; choose a new output directory')
    if batch_views<1:
        raise ValueError('batch_views must be positive')
    n=plan['acquisition']['n_views']
    indices=list(range(n)) if view_indices is None else sorted(set(int(i) for i in view_indices))
    if not indices or min(indices)<0 or max(indices)>=n:
        raise ValueError('invalid selected view indices')
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[plan['mapping'][i] for i in indices]
    s=plan['settings'];grid=plan['source_grid']
    spec=dict(grid=grid,n_detector_vu=[s['n_detector_rows'],s['n_detector_cols']],
              d_detector_vu_mm=[s['d_detector_row_mm'],s['d_detector_col_mm']],dso_mm=s['dso_mm'],dsd_mm=s['dsd_mm'])
    geometry=make_geometry(spec)
    path=outdir/'projections.npy'
    measurements=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(indices),*spec['n_detector_vu']))
    try:
        for phase in sorted({r['phase_label'] for r in rows}):
            meta=plan['phase_files'][str(phase)];source=Path(meta['path'])
            if source.stat().st_size!=meta['size'] or source.stat().st_mtime_ns!=meta['mtime_ns']:
                raise ValueError(f'source changed since planning: {source}')
            image=nib.load(str(source))
            hu=np.asarray(image.dataobj,dtype=np.float32)
            mu=hu_to_mu(hu);del hu
            positions=[j for j,r in enumerate(rows) if r['phase_label']==phase]
            for start in range(0,len(positions),batch_views):
                selected=positions[start:start+batch_views]
                angles=np.deg2rad([rows[j]['angle_deg'] for j in selected]).astype(np.float32)
                block=project_numpy(mu,geometry,angles,gpu=gpu)
                if not np.isfinite(block).all():
                    raise FloatingPointError('TIGRE produced nonfinite measurements')
                measurements[selected]=block
            measurements.flush();del mu,image
            print(f'phase={phase} generated_views={len(positions)}',flush=True)
    finally:
        measurements.flush();del measurements
    with (outdir/'view_mapping.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    case=dict(schema='bird_tigre_case_v1',geometry=spec,acquisition=plan['acquisition'],views=rows,
        projections=dict(path='projections.npy',shape=[len(indices),*spec['n_detector_vu']],sha256=file_sha256(path),
                         producer='TIGRE Ax Siddon',complete_scan=len(indices)==n),
        evaluation_grid=plan['evaluation_grid'],intensity_processing=plan['intensity_processing'],
        source_plan_sha256=file_sha256(outdir/'plan.json'),source_series=plan['series'],source_round=plan['round'],
        uses_prior_image=False,uses_gt_supervision=False)
    (outdir/'case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    return outdir/'case.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','generate'])
    parser.add_argument('--project-root',required=True);parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--series',required=True);parser.add_argument('--duration',type=float,default=10.)
    parser.add_argument('--out',required=True);parser.add_argument('--views',nargs='+',type=int)
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--batch-views',type=int,default=16)
    args=parser.parse_args();plan=plan_case(args.project_root,args.round,args.series,args.duration)
    if args.action=='plan':
        out=Path(args.out)
        if out.exists():raise FileExistsError(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        print(out)
    else:
        print(generate_case(plan,args.out,args.views,args.batch_views,args.gpu))
