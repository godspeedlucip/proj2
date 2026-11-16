"""TIGRE Siddon Ax + approximate matched Atb, plus bounded-memory INR VJP.

TIGRE is imported lazily. This module never substitutes FDK or a Python ray
integrator when TIGRE is unavailable. The matched Atb is an approximate adjoint.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .data import file_sha256


def make_geometry(spec,shape_zyx=None):
    import tigre
    grid=spec['grid'];shape=np.array(shape_zyx or grid['shape_zyx'],dtype=np.int32)
    if shape.shape!=(3,) or (shape<=0).any():
        raise ValueError('positive 3D grid required')
    g=tigre.geometry();g.mode='cone'
    g.DSO=float(spec['dso_mm']);g.DSD=float(spec['dsd_mm'])
    g.nVoxel=shape;g.sVoxel=np.array(grid['size_zyx_mm'],dtype=np.float32);g.dVoxel=g.sVoxel/shape
    g.nDetector=np.array(spec['n_detector_vu'],dtype=np.int32)
    g.dDetector=np.array(spec['d_detector_vu_mm'],dtype=np.float32);g.sDetector=g.nDetector*g.dDetector
    g.offOrigin=np.array(grid['offset_zyx_mm'],dtype=np.float32)
    g.offDetector=np.zeros(2,np.float32);g.rotDetector=np.zeros(3,np.float32);g.COR=0.;g.accuracy=.5
    return g


def gpu_ids(index):
    from tigre.utilities.gpu import GpuIds
    ids=GpuIds()
    if index not in ids.devices:
        raise ValueError(f'TIGRE GPU {index} not in available devices {ids.devices}')
    ids.devices=[index]
    return ids


def project_numpy(volume,geometry,angles,gpu=0):
    import tigre
    return tigre.Ax(np.ascontiguousarray(volume,dtype=np.float32),geometry,
                    np.ascontiguousarray(angles,dtype=np.float32),projection_type='Siddon',gpuids=gpu_ids(gpu))


class _TigreAx(torch.autograd.Function):
    @staticmethod
    def forward(ctx,volume,geometry,angles,gpu):
        if volume.dtype!=torch.float32 or volume.ndim!=3:
            raise ValueError('TIGRE expects a float32 [Z,Y,X] volume')
        ctx.geometry=copy.deepcopy(geometry);ctx.angles=np.array(angles,dtype=np.float32,copy=True)
        ctx.gpu=gpu;ctx.device=volume.device
        p=project_numpy(volume.detach().cpu().numpy(),ctx.geometry,ctx.angles,gpu)
        return torch.as_tensor(p,device=volume.device)

    @staticmethod
    def backward(ctx,gradient):
        import tigre
        g=tigre.Atb(np.ascontiguousarray(gradient.detach().cpu().numpy(),dtype=np.float32),
            ctx.geometry,ctx.angles,backprojection_type='matched',gpuids=gpu_ids(ctx.gpu))
        return torch.as_tensor(g,device=ctx.device),None,None,None


def tigre_ax(volume,geometry,angles,gpu=0):
    return _TigreAx.apply(volume,geometry,np.asarray(angles,dtype=np.float32),gpu)


def grid_points(start,stop,shape_zyx,device):
    nz,ny,nx=shape_zyx
    index=torch.arange(start,stop,device=device)
    return torch.stack(((index%nx+.5)/nx,((index//nx)%ny+.5)/ny,(index//(nx*ny)+.5)/nz),-1).float()


def materialize(model,time,shape_zyx,chunk,device):
    if chunk<1:raise ValueError('positive chunk required')
    count=int(np.prod(shape_zyx));values=torch.empty((2,count),device=device)
    with torch.no_grad():
        for start in range(0,count,chunk):
            stop=min(start+chunk,count)
            result=model(grid_points(start,stop,shape_zyx,device),time)
            values[0,start:stop]=result['mu'][:,0];values[1,start:stop]=result['mu_tp'][:,0]
    return values.reshape(2,*shape_zyx)


def volume_vjp(model,time,shape_zyx,gradient,chunk,device):
    if chunk<1 or tuple(gradient.shape)!=(2,*shape_zyx):
        raise ValueError('invalid volume gradient or chunk')
    count=int(np.prod(shape_zyx));gradient=gradient.detach().reshape(2,count)
    for start in range(0,count,chunk):
        stop=min(start+chunk,count)
        result=model(grid_points(start,stop,shape_zyx,device),time)
        torch.autograd.backward((result['mu'],result['mu_tp']),
            (gradient[0,start:stop,None],gradient[1,start:stop,None]))


def validate_case_metadata(m):
    """Reject physically inconsistent redundant fields before projection or export."""
    if m.get('schema')!='bird_tigre_case_v1' or m.get('intensity_processing')!='hu_to_mu_valid_source_only_v1':
        raise ValueError('unsupported case/intensity contract')
    if m.get('uses_prior_image') is not False or m.get('uses_gt_supervision') is not False:
        raise ValueError('this trainer requires the no-prior/no-GT case contract')
    def vector(value,shape,positive=False):
        a=np.asarray(value,dtype=np.float64)
        if a.shape!=shape or not np.isfinite(a).all() or (np.abs(a)>np.finfo(np.float32).max).any():
            raise ValueError('geometry must have correct shape and finite float32 range')
        if positive and (a<=0).any():raise ValueError('geometry lengths must be positive')
        return a
    def dimensions(value,count):
        if not isinstance(value,list) or len(value)!=count or any(type(n) is not int or n<1 or n>np.iinfo(np.int32).max for n in value):
            raise ValueError('geometry dimensions must be positive int32 values')
        return np.asarray(value,dtype=np.int64)
    g=m['geometry'];grid=g['grid']
    dims=dimensions(grid['shape_zyx'],3)
    spacing=vector(grid['spacing_zyx_mm'],(3,),True)
    size=vector(grid['size_zyx_mm'],(3,),True)
    offset=vector(grid['offset_zyx_mm'],(3,))
    bounds=vector(grid['bounds_xyz_mm'],(2,3))
    if not np.allclose(dims*spacing,size,rtol=1e-6,atol=1e-5):
        raise ValueError('shape * spacing must equal physical size')
    expected=np.stack((offset-size/2,offset+size/2))[:,::-1]
    if not np.allclose(bounds,expected,rtol=1e-6,atol=1e-5):
        raise ValueError('bounds must agree with size and offOrigin')
    detector=dimensions(g['n_detector_vu'],2)
    vector(g['d_detector_vu_mm'],(2,),True)
    dso,dsd=vector([g['dso_mm'],g['dsd_mm']],(2,),True)
    if dsd<=dso:raise ValueError('DSD must exceed DSO')
    a=m['acquisition'];n=a['n_views'];duration=a['duration_s'];span=a['angle_span_deg']
    if type(n) is not int or n<4 or not np.isfinite([duration,span]).all() or min(duration,span)<=0:
        raise ValueError('invalid acquisition sampling')
    views=m['views'];ids=[v['view_index'] for v in views]
    if not ids or any(type(i) is not int or not 0<=i<n for i in ids) or ids!=sorted(set(ids)):
        raise ValueError('global view IDs must be unique, chronological and in range')
    times=np.asarray([v['time_s'] for v in views],np.float64)
    angles=np.asarray([v['angle_deg'] for v in views],np.float64)
    if not np.isfinite(times).all() or not np.allclose(times,np.asarray(ids)*duration/n,rtol=0,atol=1e-7):
        raise ValueError('time must equal global view index * duration / n_views')
    if not np.isfinite(angles).all() or not np.allclose(angles,np.asarray(ids)*span/n,rtol=0,atol=1e-6):
        raise ValueError('angle must equal global view index * span / n_views')
    if m['projections'].get('complete_scan') is not (len(ids)==n):
        raise ValueError('complete_scan flag differs from view coverage')
    if m['projections']['shape']!=[len(ids),*detector.tolist()]:
        raise ValueError('projection dimensions differ from views/detector')


class TigreCase:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.metadata=json.loads(self.path.read_text(encoding='utf-8'))
        m=self.metadata
        validate_case_metadata(m)
        self.projection_path=(self.path.parent/m['projections']['path']).resolve()
        if file_sha256(self.projection_path)!=m['projections']['sha256']:
            raise ValueError('projection SHA-256 mismatch')
        self.projections=np.load(self.projection_path,mmap_mode='r',allow_pickle=False)
        if self.projections.dtype!=np.float32 or list(self.projections.shape)!=m['projections']['shape']:
            raise ValueError('projection shape/dtype mismatch')
        self.views=m['views'];self.duration=float(m['acquisition']['duration_s'])
        if len(self.views)!=len(self.projections) or not np.isfinite(self.duration) or self.duration<=0:
            raise ValueError('invalid acquisition duration/view count')
        times=np.array([v['time_s'] for v in self.views]);ids=[v['view_index'] for v in self.views]
        if (not np.isfinite(times).all() or (times<0).any() or (times>=self.duration).any()
            or (np.diff(times)<=0).any() or ids!=sorted(set(ids))):
            raise ValueError('times and global view IDs must be unique and chronological')
        if list(self.projections.shape[1:])!=m['geometry']['n_detector_vu']:
            raise ValueError('detector geometry mismatch')
        for block in self.projections:
            if not np.isfinite(block).all():raise ValueError('nonfinite projections')
        self.signature=file_sha256(self.path)

    def __len__(self):
        return len(self.views)
