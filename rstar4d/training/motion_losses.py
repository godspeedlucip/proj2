"""Reference-guided motion losses in full-resolution output voxel coordinates."""
from functools import lru_cache

import h5py
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from ..models.sampling_grid import FEATURE_STRIDE, feature_mask, flow_to_points, grid_zyx, pull3d


def reference_descriptor(images, valid):
    """One-voxel finite differences with one-sided support at true FOV edges."""
    squared = torch.zeros_like(images,dtype=torch.float32)
    for axis in (-3,-2,-1):
        size = images.shape[axis]-1
        delta = images.float().diff(dim=axis)
        adjacent = valid.narrow(axis,0,size) & valid.narrow(axis,1,size)
        before = torch.zeros_like(squared)
        after = torch.zeros_like(squared)
        count = torch.zeros_like(squared)
        before.narrow(axis,1,size).copy_(delta*adjacent)
        after.narrow(axis,0,size).copy_(delta*adjacent)
        count.narrow(axis,1,size).add_(adjacent)
        count.narrow(axis,0,size).add_(adjacent)
        squared += ((before+after)/count.clamp_min(1)).square()
    return squared.sqrt()


def _domain_index(domain):
    if not domain.any():
        return None
    boundary = domain & ~binary_erosion(domain,structure=np.ones((3,3,3),bool))
    coordinates = np.argwhere(boundary)
    return cKDTree(coordinates)


@lru_cache(maxsize=6)
def _source_index(path, phase):
    with h5py.File(path,'r') as file:
        return _domain_index(np.asarray(file['valid_mask'][phase],dtype=bool))


def outside_distance_vox(points_global, source_domain):
    index = source_domain if isinstance(source_domain,cKDTree) else _domain_index(np.asarray(source_domain,dtype=bool))
    if index is None:
        raise ValueError('Nearest source point requires a nonempty FOV')
    _, nearest = index.query(points_global.detach().cpu().numpy())
    q = points_global.new_tensor(index.data[nearest])
    return torch.linalg.vector_norm(points_global-q,dim=-1)


def _variance(x, mean, pool):
    return (pool(x.square())-mean.square()).clamp_min(0.)


def _motion_index(supervision, context, b, s):
    if 'source_path' in supervision:
        return _source_index(supervision['source_path'][b],int(context['phase_indices'][b,s]))
    return _domain_index(context['source_fov'][b,s].detach().cpu().numpy())


def _boundary_distance(points, support, origin, get_index):
    distances = torch.zeros_like(support,dtype=torch.float32)
    if not bool(support.all()):
        distances[~support] = outside_distance_vox(points[~support]+origin,get_index())
    return distances


def motion_terms(aux, supervision, context, motion, recipe):
    fields = aux['flow_vox'].float()
    valid = context['source_fov']
    reference = supervision.get('motion_reference', {})
    source_valid = reference.get('valid',valid)
    radius = recipe['lncc_radius_vox']
    window = tuple(2*r+1 for r in radius)
    zero = fields.sum()*0.
    structural, smooth, surface = zero, zero, zero
    counts = dict(windows=0,outside=0,low_contrast=0,cross_core=0,smooth_pairs=0,surface_pairs=0,
                  surface_points=0,surface_outside=0)
    if motion['struct_weight']:
        descriptors = reference_descriptor(reference.get('target',supervision['target'])[:,0],source_valid)

    for b in range(len(fields)):
        bounds = context['core_bounds'][b]
        origin = context['origin'][b]
        lo = [max(0,start-r) for (start,_),r in zip(bounds.tolist(),radius)]
        hi = [min(n,stop+r) for n,(_,stop),r in zip(valid.shape[-3:],bounds.tolist(),radius)]
        shape = tuple(end-start for start,end in zip(lo,hi))
        patch = tuple(slice(start,end) for start,end in zip(lo,hi))
        offset = reference['offset'][b] if reference else origin.new_zeros(3)
        shift = offset.tolist()
        reference_patch = tuple(slice(start+d,end+d) for start,end,d in zip(lo,hi,shift))
        points = grid_zyx(shape,fields.device) + fields.new_tensor(lo)
        feature_points = grid_zyx(fields.shape[-3:],fields.device) * fields.new_tensor(FEATURE_STRIDE)
        core_features = ((feature_points>=bounds[:,0]) & (feature_points<bounds[:,1])).all(dim=-1)
        surface_dirs = {}
        surface_reach = None
        if motion['surface_weight'] and 'correspondence' in supervision:
            reach = tuple(int(np.ceil(v)) for v in motion.get('max_displacement_vox',(8,8,8)))
            source_regions = reference.get('correspondence',supervision['correspondence'])[b]
            source_distance = reference.get('distance',supervision['distance'])[b]
            surface_reach = F.max_pool3d(source_regions.flatten(0,1)[:,None].float(),
                                         tuple(2*r+1 for r in reach),stride=1,padding=reach)
            input_slices = tuple(slice(d,d+n) for d,n in zip(shift,valid.shape[-3:]))
            surface_reach = surface_reach.reshape_as(source_regions)[(...,*input_slices)].bool()
        for j,(t,s) in enumerate(aux['pairs'].tolist()):
            get_index = lambda: _motion_index(supervision,context,b,s)
            if not bool(valid[b,s].any()) and get_index() is None:
                continue
            flow = fields[b:b+1,j]
            if motion['struct_weight'] and all(length>=kernel for length,kernel in zip(shape,window)):
                def pool(x):
                    return F.avg_pool3d(x[None,None],window,stride=1)[0,0]
                target = descriptors[b,t][reference_patch]
                target_valid = valid[b,t][patch]
                target_mean = pool(target)
                target_variance = _variance(target,target_mean,pool)
                centers = points[tuple(slice(r,n-r) for n,r in zip(shape,radius))]
                eligible = (((centers>=bounds[:,0]) & (centers<bounds[:,1])).all(dim=-1)
                            & (pool(target_valid.float())==1.)
                            & (target_mean>recipe['edge_threshold']) & (target_variance>recipe['variance_threshold']))
                count = int(eligible.sum())
                if count:
                    warped = points + flow_to_points(flow,points[None])[0]
                    sampled,support = pull3d(descriptors[b:b+1,s:s+1],(warped+offset)[None],source_valid[b:b+1,s])
                    sampled,support = sampled[0,0],support[0]
                    source_mean = pool(sampled)
                    source_variance = _variance(sampled,source_mean,pool)
                    source_ok = pool(support.float())==1.
                    texture_ok = source_variance > recipe['variance_threshold']
                    corr = ((pool(target*sampled)-target_mean*source_mean)
                            / (target_variance.clamp_min(1e-12)*source_variance.clamp_min(1e-12)).sqrt()).clamp(-1.,1.)
                    cost = torch.where(source_ok & texture_ok,1.-corr,2.)
                    penalty = pool(_boundary_distance(warped,support,origin,get_index))*.05
                    structural = structural + (cost+penalty)[eligible].sum()
                    counts['windows'] += count
                    counts['outside'] += int((eligible & ~source_ok).sum())
                    counts['low_contrast'] += int((eligible & source_ok & ~texture_ok).sum())
                    crossed = support & ~((warped>=bounds[:,0]) & (warped<bounds[:,1])).all(dim=-1)
                    counts['cross_core'] += int((eligible & (pool(crossed.float())>0)).sum())
            if motion['smooth_weight']:
                eligible = core_features & feature_mask(valid[b,t])
                for axis, step in zip((-3,-2,-1),FEATURE_STRIDE):
                    n = eligible.shape[axis]-1
                    nearby = eligible.narrow(axis,0,n) & eligible.narrow(axis,1,n)
                    smooth = smooth + flow[0].diff(dim=axis)[:,nearby].abs().sum()/step
                    counts['smooth_pairs'] += int(nearby.sum())*3
            if surface_reach is not None:
                radius_vox = motion['surface_truncation_vox']
                for k in range(2):
                    pair_regions = supervision['correspondence'][b,:,k]
                    surface_points = torch.nonzero(supervision['surface'][b,t,k] & pair_regions[t] & surface_reach[s,k]).float()
                    surface_points = surface_points[((surface_points>=bounds[:,0]) & (surface_points<bounds[:,1])).all(dim=-1)]
                    if not len(surface_points):
                        continue
                    warped = surface_points + flow_to_points(flow,surface_points[None])[0]
                    source_points = (warped+offset)[None]
                    values,known = pull3d(source_distance[None,s,k:k+1],source_points,source_regions[s:s+1,k])
                    _,support = pull3d(source_valid[b:b+1,s:s+1].float(),source_points,source_valid[b:b+1,s])
                    penalty = _boundary_distance(warped,support[0],origin,get_index)
                    score = torch.where(known[0],values[0,0].clamp(0,radius_vox),radius_vox)+.1*penalty
                    surface_dirs[t,s,k] = (score.mean(),len(score),int((~support).sum()))
        for (t,s,k),(a,na,oa) in surface_dirs.items():
            if t<s and (s,t,k) in surface_dirs:
                c,nc,oc = surface_dirs[s,t,k]
                surface = surface + (a+c)/2
                counts['surface_pairs'] += 1
                counts['surface_points'] += na+nc
                counts['surface_outside'] += oa+oc
                prefix = ('heart_outer','lv')[k]+'_surface_'
                for label,n in [('pairs',1),('points',na+nc),('outside',oa+oc)]:
                    counts[prefix+label] = counts.get(prefix+label,0)+n
    return {**counts,'struct':structural/max(1,counts['windows']),'smooth':smooth/max(1,counts['smooth_pairs']),
            'surface':surface/max(1,counts['surface_pairs'])}
