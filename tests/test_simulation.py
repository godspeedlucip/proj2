import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from rstar4d.simulation.reference import TigreTool, SeriesSettings, ProjectionTool, calibrated_volume, make_phase_gt, apply_reconstruction_calibration
from rstar4d.simulation.phase_mapping import build_projection_phase_mapping
from rstar4d.simulation.backend import reconstruct, geometry, project

class SimulationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA/TIGRE required')
    def test_raw_phantom_pipeline(self):
        from rstar4d.simulation.phantom import make_phantom
        from rstar4d.simulation.pipeline import simulate_manifest
        from rstar4d.config import load_config
        from rstar4d.data.contracts import validate_manifest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=make_phantom(root/'raw',views=200)
            cfg=load_config(paths['config'],'simulation')
            limited={**cfg,'limit':1}
            partial_manifest=simulate_manifest(paths['manifest'],limited,root/'prepared')
            self.assertEqual(len(validate_manifest(partial_manifest)),1)
            manifest=simulate_manifest(paths['manifest'],cfg,root/'prepared')
            rows=validate_manifest(manifest,paths['split'])
            self.assertEqual(len(rows),3)
            self.assertEqual({len(r['metadata']['phase_labels']) for r in rows},{10,11})
            # Matching cache must reuse validated complete cases.
            simulate_manifest(paths['manifest'],cfg,root/'prepared')

    def test_units_padding_calibration(self):
        hu=np.array([[[-2048.,-1000.,0.]]],dtype=np.float32)
        mu=calibrated_volume(hu,np.poly1d([1,0]),0,0,apply_air_threshold=False)
        np.testing.assert_allclose(mu,[[[0,0,.02]]])
        a=np.arange(12,dtype=np.float32).reshape(1,4,3)
        np.testing.assert_array_equal(ProjectionTool.pad_linatech(a,2),np.concatenate([a[:,1::-1],a,a[:,-2:]],axis=1))
        output=apply_reconstruction_calibration(mu.copy(),np.poly1d([2,1]))
        np.testing.assert_allclose(output,[[[0,0,1.04]]])

    def test_cycle_mapping(self):
        p=Path('D:/desktop/课题2/data/cycle.csv')
        if not p.exists(): self.skipTest('User source cycle not present')
        for t,counts in [(10,[32,66,67,67,67,67,67,67,66,34]),(11,[28,62,60,60,61,58,61,60,60,62,28])]:
            m=build_projection_phase_mapping(p,t,min_phase=1,max_phase=t,n_projections=600)
            self.assertEqual([r.projection_index for r in m],list(range(600)))
            np.testing.assert_array_equal(np.bincount([r.phase_label for r in m])[1:],counts)
            self.assertEqual(m[-1].angle_deg,299.5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA/TIGRE required')
    def test_cuda_reference_reconstruction_and_mask(self):
        s=SeriesSettings(n_detector_rows=64,n_detector_cols=64,d_detector_row_mm=1,d_detector_col_mm=1,
            voxel_spacing_multiplier=1,padding_row_pixels=4,padding_col_pixels=4,fov_output_pixels=32,
            fov_source_pixels=32,fov_mask_shrink_pixels=2)
        geo=geometry((32,32,32),(1,1,1),s)
        z,y,x=np.indices((32,32,32))-15.5
        hu=np.where(x*x+y*y+z*z<49,0.,-1000.).astype(np.float32)
        angles=TigreTool.make_angles(120,0,360)
        p=project(TigreTool.hu_to_mu(hu),geo,angles,32); original=p.copy()
        output,spacing=reconstruct(p,geo,angles,s)
        np.testing.assert_array_equal(p,original)
        self.assertEqual(output.shape,(32,32,32)); self.assertTrue(np.isfinite(output).all())
        self.assertLess(abs(float(output[14:18,14:18,14:18].mean())/.02-1),.05)
        _,mask_spacing,mask=make_phase_gt(hu,(1,1,1),s,np.poly1d([1,0]),0,0,return_valid_mask=True)
        np.testing.assert_allclose(mask_spacing,spacing)
        self.assertTrue(mask[16,16,16])

if __name__=='__main__': unittest.main()
