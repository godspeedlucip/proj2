"""World-mm ray integration; networks see normalized coordinates."""
import torch
from torch.utils.checkpoint import checkpoint


def sample_rays(origins,directions,bounds,samples):
    if samples<1:
        raise ValueError('samples must be positive')
    if origins.ndim!=2 or origins.shape[-1]!=3 or directions.shape!=origins.shape or bounds.shape!=(2,3):
        raise ValueError('invalid ray/bounds shapes')
    lengths=directions.norm(dim=-1,keepdim=True)
    if bool((lengths<=0).any()):
        raise ValueError('zero ray direction')
    direction=directions/lengths
    parallel=direction.abs()<1e-10
    safe=torch.where(parallel,torch.ones_like(direction),direction)
    a=(bounds[0]-origins)/safe
    b=(bounds[1]-origins)/safe
    lo=torch.minimum(a,b)
    hi=torch.maximum(a,b)
    lo=torch.where(parallel,torch.full_like(lo,-torch.inf),lo)
    hi=torch.where(parallel,torch.full_like(hi,torch.inf),hi)
    outside=(parallel & ((origins<bounds[0]) | (origins>bounds[1]))).any(-1)
    near=lo.amax(-1).clamp_min(0)
    far=hi.amin(-1)
    valid=(far>near)&~outside
    near=torch.where(valid,near,torch.zeros_like(near))
    far=torch.where(valid,far,near)
    ds=(far-near)/samples
    fractions=torch.arange(samples,device=origins.device,dtype=origins.dtype)+.5
    distance=near[:,None]+ds[:,None]*fractions[None,:]
    world=origins[:,None,:]+distance[:,:,None]*direction[:,None,:]
    points=(world-bounds[0])/(bounds[1]-bounds[0])
    points=torch.where(valid[:,None,None],points,torch.full_like(points,.5))
    return points,ds,valid


def render_rays(model,origins,directions,times,bounds,samples,ray_chunk=64,checkpoint_chunks=False):
    if len(origins)<1 or ray_chunk<1:
        raise ValueError('nonempty rays and positive ray_chunk required')
    final,topology=[],[]
    def query(x,t):
        result=model(x,t)
        return torch.cat((result['mu'],result['mu_tp']),-1)
    for start in range(0,len(origins),ray_chunk):
        stop=start+ray_chunk
        xyz,ds,_=sample_rays(origins[start:stop],directions[start:stop],bounds,samples)
        n=len(xyz)
        time=times[start:stop].reshape(n,1,1).expand(n,samples,1).reshape(-1,1)
        xyz=xyz.reshape(-1,3)
        if checkpoint_chunks and torch.is_grad_enabled():
            values=checkpoint(query,xyz,time,use_reentrant=False)
        else:
            values=query(xyz,time)
        integral=values.reshape(n,samples,2).sum(1)*ds[:,None]
        final.append(integral[:,0])
        topology.append(integral[:,1])
    return torch.cat(final),torch.cat(topology)
