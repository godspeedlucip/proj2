"""Reproducible real-GPU checks for the installed TIGRE operator pair."""
from __future__ import annotations
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from .tigre_backend import make_geometry,project_numpy,tigre_ax,gpu_ids


def check(out,gpu=0):
    import tigre
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    spec=dict(grid=dict(shape_zyx=[12,12,12],size_zyx_mm=[12.,12.,12.],offset_zyx_mm=[0.,0.,0.]),
              dso_mm=100.,dsd_mm=150.,n_detector_vu=[20,20],d_detector_vu_mm=[1.,1.])
    geo=make_geometry(spec);angles=np.array([.13,.71,1.4],np.float32)
    rng=np.random.default_rng(2026)
    x=rng.random((12,12,12),dtype=np.float32);y=rng.random((3,20,20),dtype=np.float32)
    ax=project_numpy(x,geo,angles,gpu)
    aty=tigre.Atb(y,geo,angles,backprojection_type='matched',gpuids=gpu_ids(gpu))
    lhs=float(np.sum(ax.astype(np.float64)*y));rhs=float(np.sum(x.astype(np.float64)*aty))
    error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
    device=f'cuda:{gpu}'
    leaf=torch.tensor(x,device=device,requires_grad=True)
    (tigre_ax(leaf,geo,angles,gpu)*torch.tensor(y,device=device)).sum().backward()
    gradient=leaf.grad.detach().cpu().numpy()
    report=dict(python=platform.python_version(),torch=str(torch.__version__),pytigre=version('pytigre'),
                gpu=torch.cuda.get_device_name(gpu),geometry=spec,angles_rad=angles.tolist(),
                ax_dot_y=lhs,x_dot_atb_y=rhs,relative_inner_product_error=error,
                wrapper_gradient_max_abs_error=float(np.max(np.abs(gradient-aty))),
                all_finite=bool(np.isfinite(ax).all() and np.isfinite(gradient).all()),
                interpretation='matched Atb is approximate; this is not an exact adjoint/finite-difference proof')
    if not report['all_finite'] or report['wrapper_gradient_max_abs_error']>1e-6:
        raise AssertionError(report)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();check(args.out,args.gpu)
