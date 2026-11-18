import unittest

import numpy as np
import torch

from rstar4d.config import resolve_stage2, validate_train
from rstar4d.evaluation import report
from rstar4d.inference.sliding_window import predict_volume


class ContextIdentity(torch.nn.Module):
    stage = 2

    def forward(self, image):
        return image[:,:1]


class CardiacDeploymentTests(unittest.TestCase):
    def test_odd_size_core_tiling_uses_same_grid_as_full(self):
        rng = np.random.default_rng(31)
        source = rng.random((3,9,21,27),dtype=np.float32)
        recipe = resolve_stage2(validate_train({'stage2_crop_mode':'core_with_context'}),2)
        kwargs = {'context':{'source_fov':np.ones_like(source,dtype=bool),'phase_labels':[1,2,8]},
                  'recipe':recipe,'tile':(4,16,16),'overlap':(2,8,8)}
        model = ContextIdentity()
        norm = {'scale':1.,'offset':0.}
        full = predict_volume(model,source,source[0],norm,mode='full',**kwargs)
        tiled = predict_volume(model,source,source[0],norm,mode='tiled',**kwargs)
        np.testing.assert_allclose(full,tiled,atol=2e-7)

    def test_surfaces_in_output_voxels_and_empty_failures(self):
        self.assertTrue(hasattr(report,'surface_metrics'))
        left = np.zeros((4,4,4),dtype=bool)
        left[1,:,:] = True
        right = np.roll(left,1,axis=0)
        near = report.surface_metrics(left,right,1.)
        self.assertEqual((near['surface_dice'],near['hd95'],near['assd']),(1.,1.,1.))
        far = report.surface_metrics(left,right,.9)
        self.assertEqual(far['surface_dice'],0.)
        empty = report.surface_metrics(np.zeros_like(left),right,1.)
        self.assertEqual(empty['status'],'empty_prediction')
        self.assertIsNone(empty['hd95'])
