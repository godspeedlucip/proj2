from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import numpy as np
from .reference import TigreTool, reconstruct_prior, SeriesSettings
from ..utils.io import sha256, environment


def require_cuda():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('TIGRE simulation requires a working CUDA GPU; no fake/CPU fallback')
    try: import tigre
    except ImportError as exc: raise RuntimeError('Install/build official CERN TIGRE in this Python environment') from exc


def backend_fingerprint():
    require_cuda()
    result={'environment':environment(),'files':{}}
    for name in ['tigre','tigre.utilities.Ax','tigre.algorithms.single_pass_algorithms','_Ax','_Atb']:
        try:
            spec=importlib.util.find_spec(name)
            if spec and spec.origin and Path(spec.origin).is_file(): result['files'][name]=sha256(spec.origin)
        except (ImportError,ValueError,ModuleNotFoundError): pass
    return result


def geometry(shape,spacing,settings):
    settings.validate()
    if settings.dsd_mm<=settings.dso_mm: raise ValueError('DSD must exceed DSO')
    geo=TigreTool.build_tigre_geometry(shape,np.asarray(spacing)*settings.voxel_spacing_multiplier,
        n_detector=settings.n_detector,d_detector=settings.d_detector_mm,dso=settings.dso_mm,dsd=settings.dsd_mm)
    geo.offOrigin=np.array([0,-(settings.shift_down_voxels-settings.shift_up_voxels)*geo.dVoxel[1],
        -(settings.shift_right_voxels-settings.shift_left_voxels)*geo.dVoxel[2]],dtype=np.float32)
    return geo


def project(volume,geo,angles,batch=64):
    require_cuda(); import tigre
    if batch<1: raise ValueError('view_batch must be positive')
    volume=np.asarray(volume,dtype=np.float32)
    if volume.shape!=tuple(geo.nVoxel) or not np.isfinite(volume).all(): raise ValueError('Invalid projection volume')
    angles=np.asarray(angles,dtype=np.float32)
    output=np.empty((len(angles),*geo.nDetector),dtype=np.float32)
    for start in range(0,len(angles),batch):
        output[start:start+batch]=tigre.Ax(volume,copy.deepcopy(geo),angles[start:start+batch],projection_type='Siddon')
    if not np.isfinite(output).all(): raise FloatingPointError('Nonfinite TIGRE projections')
    return output


def reconstruct(projections,geo,angles,settings):
    require_cuda()
    if len(angles)<2: raise ValueError('FDK requires at least two views')
    output,spacing=reconstruct_prior(np.array(projections,dtype=np.float32,copy=True),copy.deepcopy(geo),
        np.asarray(angles,dtype=np.float32),TigreTool(),padding_row_pixels=settings.padding_row_pixels,
        padding_col_pixels=settings.padding_col_pixels,fov_output_transverse_pixels=settings.fov_output_pixels,
        fov_source_pixels=settings.fov_source_pixels,fov_mask_diameter_shrink_pixels=settings.fov_mask_shrink_pixels)
    if not np.isfinite(output).all(): raise FloatingPointError('Nonfinite TIGRE reconstruction')
    return output,spacing


def output_grid(geo,settings):
    fov=geo.sDetector*geo.DSO/geo.DSD
    shape=np.minimum(np.maximum(np.floor(np.array([fov[0],fov[1],fov[1]])/geo.dVoxel).astype(int),1),geo.nVoxel)
    voxel=np.array(geo.dVoxel,dtype=float); center=np.zeros(3,dtype=float); width=settings.fov_output_pixels
    if settings.fov_source_pixels:
        shape[1:]=width; voxel[1:]*=settings.fov_source_pixels/width
    elif min(shape[1:])<width:
        # scipy.zoom(grid_mode=False) samples endpoints; retain the true sampling affine.
        voxel[1:]*=(shape[1:]-1)/max(width-1,1); shape[1:]=width
    else:
        start=(shape[1:]-width)//2
        center[1:]=(start+(width-1)/2-(shape[1:]-1)/2)*voxel[1:]; shape[1:]=width
    affine=np.eye(4); affine[:3,:3]=np.diag(voxel); affine[:3,3]=center-(shape-1)/2*voxel
    return {'shape_zyx':shape.tolist(),'voxel_spacing_zyx_mm':voxel.tolist(),
        'center_zyx_mm':center.tolist(),'voxel_to_scanner_zyx':affine.tolist(),
        'DSO':float(geo.DSO),'DSD':float(geo.DSD),'nDetector':geo.nDetector.tolist(),'dDetector':geo.dDetector.tolist(),
        'offDetector':[0.,0.],'rotDetector':[0.,0.,0.],'COR':0.,'projector':'Siddon','filter':'ram_lak','parker':False}
