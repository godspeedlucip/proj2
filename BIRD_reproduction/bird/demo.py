"""Small synthetic nonperiodic 3D Gaussian scene, NOT the paper's XCAT dataset."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .geometry import sample_rays
from .io import npz_output_path


def phantom(x,t):
    # Smooth anatomy, nonperiodic translation and independent intensity variation.
    shift=.12*t+.025*torch.sin(5*t)
    center=torch.cat((.42+shift,.5+.025*torch.cos(3*t),.5+0*t),-1)
    scale=x.new_tensor([.24,.3,.32])
    body=.018*torch.exp(-2*((x-.5)/scale).square().sum(-1,keepdim=True))
    heart=.025*torch.exp(-2*((x-center)/.13).square().sum(-1,keepdim=True))
    vessel_center=center+x.new_tensor([.13,0.,.03])
    vessel=(.015+.012*t)*torch.exp(-2*((x-vessel_center)/x.new_tensor([.035,.04,.19])).square().sum(-1,keepdim=True))
    return body+heart+vessel


def create_demo(path,views=24,detector=16,samples=96):
    if min(views,detector,samples)<2:
        raise ValueError('demo dimensions must be >=2')
    path=npz_output_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    angles=np.arange(views)*2*np.pi/views
    radial=np.stack((np.cos(angles),np.sin(angles),np.zeros(views)),-1)
    sources=160*radial;centers=-80*radial
    u=np.stack((-np.sin(angles),np.cos(angles),np.zeros(views)),-1)*(180/detector)
    v=np.tile([0.,0.,180/detector],(views,1))
    times=np.linspace(0,1,views)
    bounds=np.array([[-50.,-50.,-50.],[50.,50.,50.]],dtype=np.float32)
    row,col=np.meshgrid(np.arange(detector),np.arange(detector),indexing='ij')
    values=[]
    with torch.no_grad():
        for view in range(views):
            targets=centers[view]+(col.reshape(-1)-(detector-1)/2)[:,None]*u[view]+(row.reshape(-1)-(detector-1)/2)[:,None]*v[view]
            origins=np.broadcast_to(sources[view],targets.shape).copy()
            xyz,ds,_=sample_rays(torch.tensor(origins,dtype=torch.float32),torch.tensor(targets-origins,dtype=torch.float32),torch.tensor(bounds),samples)
            flat=xyz.reshape(-1,3)
            t=torch.full((len(flat),1),float(times[view]))
            value=phantom(flat,t).reshape(-1,samples).sum(-1)*ds
            values.append(value.reshape(detector,detector).numpy())
    np.savez_compressed(path,projections=np.stack(values).astype(np.float32),sources_mm=sources.astype(np.float32),
                        detector_centers_mm=centers.astype(np.float32),detector_u_mm=u.astype(np.float32),
                        detector_v_mm=v.astype(np.float32),times_s=times,bounds_mm=bounds,time_bounds_s=np.array([0.,1.]))
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/demo_data.npz');parser.add_argument('--views',type=int,default=24)
    parser.add_argument('--detector',type=int,default=16);parser.add_argument('--samples',type=int,default=96)
    args=parser.parse_args();torch.set_num_threads(4)
    print(create_demo(args.out,args.views,args.detector,args.samples))
