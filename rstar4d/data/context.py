"""Output-grid window coordinates shared by patch training and tiled inference."""
from dataclasses import dataclass
from math import ceil

import numpy as np
import torch


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
    alignment = recipe.get('alignment', (2,8,8))
    low = tuple(max(0,(start-h)//a*a) for start,h,a in zip(core_origin,halo,alignment))
    high = tuple(min(n,ceil((start+k+h)/a)*a)
                 for n,start,k,h,a in zip(volume_shape,core_origin,core_shape,halo,alignment))
    return Window(low, high, tuple(core_origin), tuple(core_shape))


def check_support(window, recipe, source_fov, case_id):
    """A clipped scan edge is legal. An omitted slab of real support is an input error."""
    required = plan_window(source_fov.shape[-3:], window.core_origin, window.core_shape, recipe)
    for axis,(lo,hi,read_lo,read_hi) in enumerate(zip(required.origin,required.stop,window.origin,window.stop)):
        for missing_start,missing_stop in ((lo,read_lo),(read_hi,hi)):
            if missing_stop <= missing_start:
                continue
            slab = list(required.slices)
            slab[axis] = slice(missing_start,missing_stop)
            if np.any(source_fov[(...,*slab)]):
                raise ValueError(f'{case_id}: W lacks source support on {"ZYX"[axis]} ({missing_stop-missing_start} vox)')


def crop_core(array, context):
    bounds = context['core_bounds']
    if bounds.ndim == 3:
        if not torch.equal(bounds, bounds[:1].expand_as(bounds)):
            raise ValueError('Batched core bounds must match')
        bounds = bounds[0]
    return array[(..., *(slice(start,stop) for start,stop in bounds.tolist()))]
