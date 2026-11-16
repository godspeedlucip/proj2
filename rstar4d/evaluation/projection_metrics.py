from __future__ import annotations
import h5py
import numpy as np
from ..simulation.reference import TigreTool
from ..simulation.backend import project


def projection_metrics(pred,metadata,projection_path,batch=64):
    if metadata.get('intensity_domain')!='mu' or metadata.get('calibration_application','none')!='none':
        return {'status':'not_applicable','reason':'Physical projection requires uncalibrated mu'}
    if not metadata.get('full_object_fov') or not metadata.get('full_fov_verified_from_source_support'):
        return {'status':'not_applicable','reason':'Complete source support is not verified in output FOV'}
    if not projection_path: return {'status':'not_applicable','reason':'No saved mixed projections'}
    grid=metadata.get('geometry',{})
    required={'shape_zyx','voxel_spacing_zyx_mm','center_zyx_mm','voxel_to_scanner_zyx','DSO','DSD','nDetector','dDetector'}
    if not required.issubset(grid): return {'status':'not_applicable','reason':'Output grid-to-scanner geometry is incomplete'}
    if tuple(grid['shape_zyx'])!=tuple(pred.shape[1:]): raise ValueError('Projection evaluation output shape mismatch')
    affine=np.asarray(grid['voxel_to_scanner_zyx'])
    if not np.allclose(affine[:3,:3],np.diag(grid['voxel_spacing_zyx_mm'])): raise ValueError('Unsupported rotated output grid')
    geo=TigreTool.build_tigre_geometry(pred.shape[1:],grid['voxel_spacing_zyx_mm'],n_detector=grid['nDetector'],
        d_detector=grid['dDetector'],dso=grid['DSO'],dsd=grid['DSD'])
    # Predictions already occupy the output grid. Never add the source offOrigin again.
    geo.offOrigin=np.asarray(grid['center_zyx_mm'],dtype=np.float32)
    squared=0.; absolute=0.; target_squared=0.; n=0; ncc=[]; constant=0
    with h5py.File(projection_path,'r') as f:
        angles=np.asarray(f['angles_rad']); labels=np.asarray(f['state_label'])
        if not set(labels).issubset(metadata['phase_labels']): raise ValueError('Unknown projection state label')
        for t,label in enumerate(metadata['phase_labels']):
            indices=np.flatnonzero(labels==label)
            if not len(indices): raise ValueError('State has no saved projections')
            volume=np.asarray(pred[t],dtype=np.float32)
            for start in range(0,len(indices),batch):
                ix=indices[start:start+batch]; simulated=project(volume,geo,angles[ix],batch)
                measured=np.asarray(f['mixed_line_integral'][ix],dtype=np.float64)
                residual=simulated.astype(np.float64)-measured
                squared+=float(np.square(residual).sum()); absolute+=float(np.abs(residual).sum())
                target_squared+=float(np.square(measured).sum()); n+=residual.size
                for a,b in zip(simulated,measured):
                    a=a.astype(np.float64).ravel(); b=b.ravel(); a=a-a.mean(); b=b-b.mean()
                    denominator=np.linalg.norm(a)*np.linalg.norm(b)
                    if denominator<=1e-15: constant+=1
                    else: ncc.append(float(np.dot(a,b)/denominator))
    return {'status':'ok','mae':absolute/n,'rmse':float(np.sqrt(squared/n)),
        'relative_l2':float(np.sqrt(squared/target_squared)) if target_squared else None,
        'ncc':float(np.mean(ncc)) if ncc else None,'constant_views':constant,'units':'dimensionless line integral'}
