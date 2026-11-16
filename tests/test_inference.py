import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from rstar4d.inference.sliding_window import predict_volume
from rstar4d.evaluation.image_metrics import image_metrics
from rstar4d.evaluation.projection_metrics import projection_metrics


class Identity(torch.nn.Module):
    stage=2
    def forward(self,x): return x[:,:1]


class InferenceTests(unittest.TestCase):
    def test_tiled_identity_odd_shapes_and_temporal_axis(self):
        for t in [10,11]:
            a=np.random.default_rng(2).random((t,7,19,21),dtype=np.float32)*.02
            p=a.mean(0)
            out=predict_volume(Identity(),a,p,{'scale':.02,'offset':0},tile=(4,16,16),overlap=(2,8,8),device='cpu',amp=False)
            np.testing.assert_allclose(out,a,rtol=2e-6,atol=1e-8)
            self.assertEqual(out.shape,a.shape)

    def test_metric_units_empty_roi_and_window_mask(self):
        a=np.full((10,2,16,16),.02,np.float32); mask=np.ones_like(a,bool)
        same=image_metrics(a,a,mask)
        self.assertEqual(same['rmse'],0); self.assertAlmostEqual(same['ssim'],1,6)
        shifted=image_metrics(a+.002,a,mask)
        self.assertAlmostEqual(shifted['rmse'],.002,7)
        empty=image_metrics(a,a,mask,roi=np.zeros_like(mask))
        self.assertIsNone(empty['rmse']); self.assertIsNone(empty['ssim'])
        mask[:]=False; self.assertIsNone(image_metrics(a,a,mask)['ssim'])

    def test_projection_inapplicable(self):
        for meta in [{'intensity_domain':'calibrated_mu','full_object_fov':True},
                     {'intensity_domain':'mu','full_object_fov':False}]:
            result=projection_metrics(None,meta,None)
            self.assertEqual(result['status'],'not_applicable')

if __name__=='__main__': unittest.main()
