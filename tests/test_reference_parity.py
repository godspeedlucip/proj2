"""Optional local-source compatibility checks; package never imports these sources at runtime."""
import dataclasses
import copy
import importlib
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from rstar4d.simulation.reference import SeriesSettings, make_phase_gt, phase_grid, ProjectionTool
from rstar4d.simulation.backend import geometry, project, reconstruct, output_grid


@unittest.skipUnless(Path('D:/desktop/课题2/generate_round1_contrast_gt.py').exists(),'Original local generator unavailable')
class SourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_bytecode=sys.dont_write_bytecode; sys.dont_write_bytecode=True
        sys.path.insert(0,'D:/desktop/课题2')
        cls.original=importlib.import_module('generate_round1_contrast_gt')
        cls.original_recon=importlib.import_module('generate_round1_contrast_recon')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove('D:/desktop/课题2'); sys.dont_write_bytecode=cls.old_bytecode

    def test_odd_even_shift_crop_wide_and_zoom_masks(self):
        rng=np.random.default_rng(3)
        for shape,output,fov in [((18,40,38),16,0),((17,31,29),32,0),((17,31,29),24,18)]:
            hu=rng.uniform(-1000,1000,shape).astype(np.float32); hu[:,0:3,:]=-2048
            for up,down,left,right in [(2,0,1,0),(0,3,0,2)]:
                s=SeriesSettings(voxel_spacing_multiplier=1.5,n_detector_rows=48,n_detector_cols=48,
                    d_detector_row_mm=1,d_detector_col_mm=1,fov_output_pixels=output,fov_source_pixels=fov,
                    shift_up_voxels=up,shift_down_voxels=down,shift_left_voxels=left,shift_right_voxels=right,
                    padding_row_pixels=4,padding_col_pixels=4,fov_mask_shrink_pixels=2)
                original_settings=self.original_recon.SeriesSettings(**dataclasses.asdict(s))
                spacing=(.7,.8,.9)
                a=make_phase_gt(hu,spacing,s,np.poly1d([1,0]),0,0,return_valid_mask=True)
                b=self.original.make_phase_gt(hu,spacing,original_settings,np.poly1d([1,0]),0,0,return_valid_mask=True)
                for actual,expected in zip(a,b): np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
                geo=geometry(shape,spacing,s)
                np.testing.assert_allclose(geo.offOrigin,[0,-(down-up)*.8*1.5,-(right-left)*.9*1.5],atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(),'Original reconstruction parity requires CUDA')
    def test_original_cuda_reconstruction_with_shifted_odd_grids(self):
        from rstar4d.simulation.reference import TigreTool
        angles=TigreTool.make_angles(60,0,300)
        for shape,width,fov in [((18,40,38),24,0),((17,31,29),32,0),((17,31,29),24,18)]:
            for direction in [-1,1]:
                s=SeriesSettings(voxel_spacing_multiplier=1,n_detector_rows=64,n_detector_cols=64,
                    d_detector_row_mm=1,d_detector_col_mm=1,padding_row_pixels=4,padding_col_pixels=4,
                    fov_output_pixels=width,fov_source_pixels=fov,fov_mask_shrink_pixels=2,
                    shift_up_voxels=max(0,direction*2),shift_down_voxels=max(0,-direction*2),shift_right_voxels=1)
                geo=geometry(shape,(.8,.9,1),s)
                z,y,x=np.indices(shape); mu=(((z-shape[0]/2+1)/4)**2+((y-shape[1]/2-2)/6)**2+((x-shape[2]/2+3)/5)**2<1).astype(np.float32)*.02
                projections=project(mu,geo,angles)
                a,spacing=reconstruct(projections,geo,angles,s)
                b,reference_spacing=self.original_recon.reconstruct_prior(projections.copy(),copy.deepcopy(geo),angles.copy(),self.original_recon.TigreTool(),
                    padding_row_pixels=s.padding_row_pixels,padding_col_pixels=s.padding_col_pixels,
                    fov_output_transverse_pixels=s.fov_output_pixels,fov_source_pixels=s.fov_source_pixels,
                    fov_mask_diameter_shrink_pixels=s.fov_mask_shrink_pixels)
                np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-6)
                np.testing.assert_allclose(spacing,reference_spacing,rtol=0,atol=1e-7)
                self.assertEqual(tuple(output_grid(geo,s)['shape_zyx']),a.shape)

if __name__=='__main__': unittest.main()
