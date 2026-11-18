"""Output-grid window coordinates shared by patch training and tiled inference."""
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
import torch


STAGE2_ALIGNMENT = (2,8,8)


def _x2_geometry():
    """Output-grid receptive radius and step at the end of backbone conv3_3."""
    radius, step = [0,0,0], [1,1,1]
    for down in ((2,1,1),(1,2,2),(1,2,2)):
        for axis,stride in enumerate(down):
            # One 3-tap down-convolution, then two residual blocks with
            # two more 3-tap convolutions per spatial axis in each block.
            radius[axis] += step[axis]
            step[axis] *= stride
            radius[axis] += 4 * step[axis]
    return tuple(radius),tuple(step)


X2_RADIUS, X2_STRIDE = _x2_geometry()


@dataclass(frozen=True)
class Window:
    origin: tuple
    stop: tuple
    core_origin: tuple
    core_shape: tuple

    @property
    def slices(self):
        return tuple(slice(start, stop) for start, stop in zip(self.origin,self.stop))

    @property
    def core(self):
        return tuple(slice(start - origin, start - origin + size)
                     for start, origin, size in zip(self.core_origin,self.origin,self.core_shape))


def plan_window(volume_shape, core_origin, core_shape, recipe):
    halo = [ceil(d + r + margin) for d,r,margin in zip(
        recipe['max_displacement_vox'],recipe['lncc_radius_vox'],recipe['support_margin_vox'])]
    alignment = recipe.get('alignment', STAGE2_ALIGNMENT)
    low = tuple(max(0,(start-h)//a*a) for start,h,a in zip(core_origin,halo,alignment))
    high = tuple(min(n,ceil((start+k+h)/a)*a)
                 for n,start,k,h,a in zip(volume_shape,core_origin,core_shape,halo,alignment))
    return Window(low, high, tuple(core_origin), tuple(core_shape))


def required_window(volume_shape, core_origin, core_shape, recipe):
    """Propagate support on the full scan's lattices, independently of the requested margin."""
    low,high = [],[]
    for n,start,k,d,r,step,rf,align in zip(volume_shape,core_origin,core_shape,
            recipe['max_displacement_vox'],recipe['lncc_radius_vox'],X2_STRIDE,X2_RADIUS,STAGE2_ALIGNMENT):
        # Inclusive outermost sample coordinates after LNCC and allowed displacement.
        first,last = start-r-d, min(start+k,n)-1+r+d
        # Trilinear reference corners plus one-voxel gradient differences.
        ref_first,ref_last = floor(first)-1, ceil(last)+1
        # Bracketing x2 feature centers, each backed by its input receptive field.
        enc_first = floor(first/step)*step-rf
        enc_last = ceil(last/step)*step+rf
        low.append(max(0,min(ref_first,enc_first)//align*align))
        high.append(min(n,ceil((max(ref_last,enc_last)+1)/align)*align))
    return Window(tuple(low),tuple(high),tuple(core_origin),tuple(core_shape))


def check_support(window, recipe, source_fov, case_id, *, prior_fov=None, volume_shape=None):
    """Check actual W against scan-clipped requirements before any local padding."""
    if volume_shape is None:
        volume_shape = source_fov.shape[-3:]
    required = required_window(volume_shape,window.core_origin,window.core_shape,recipe)
    for axis,(lo,hi,read_lo,read_hi) in enumerate(zip(required.origin,required.stop,window.origin,window.stop)):
        for side,missing_start,missing_stop in (('left',lo,read_lo),('right',read_hi,hi)):
            if missing_stop <= missing_start:
                continue
            slab = list(required.slices)
            slab[axis] = slice(missing_start,missing_stop)
            # With no domain supplied the direct array API treats the full scan as valid.
            if (source_fov is None or np.any(source_fov[(...,*slab)])
                    or (prior_fov is not None and np.any(prior_fov[(...,*slab)]))):
                raise ValueError(f'{case_id}: W lacks source support on {"ZYX"[axis]} {side} '
                                 f'({missing_stop-missing_start} vox)')


def crop_core(array, context):
    bounds = context['core_bounds']
    if bounds.ndim == 3:
        if not torch.equal(bounds, bounds[:1].expand_as(bounds)):
            raise ValueError('Batched core bounds must match')
        bounds = bounds[0]
    return array[(..., *(slice(start,stop) for start,stop in bounds.tolist()))]
