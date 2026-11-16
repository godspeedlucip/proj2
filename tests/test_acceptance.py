import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from rstar4d.models import create_model, initialize_stage2, forward_volume
from rstar4d.training.losses import compute_loss
from rstar4d.data.contracts import basic_metadata
from rstar4d.data.paired import resample_pair
from rstar4d.evaluation.projection_metrics import projection_metrics
from rstar4d.simulation.reference import SeriesSettings, TigreTool, make_phase_gt
from rstar4d.simulation.backend import geometry, output_grid, project, reconstruct


class AcceptanceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'Shifted source chain requires TIGRE CUDA')
    def test_shifted_source_fdk_output_reprojection(self):
        import copy
        s=SeriesSettings(voxel_spacing_multiplier=1,n_detector_rows=64,n_detector_cols=64,
            d_detector_row_mm=1,d_detector_col_mm=1,fov_output_pixels=32,fov_source_pixels=32,
            padding_row_pixels=4,padding_col_pixels=4,fov_mask_shrink_pixels=0,shift_up_voxels=3,shift_right_voxels=2)
        geo=geometry((32,32,32),(1,1,1),s); grid=output_grid(geo,s)
        z,y,x=np.indices((32,32,32))-15.5
        mu=(((z+1)/4)**2+((y-1)/6)**2+((x+2)/5)**2<1).astype(np.float32)*.02
        angles=TigreTool.make_angles(120,0,360); observed=project(mu,geo,angles)
        reconstructed,_=reconstruct(observed,geo,angles,s)
        outgeo=copy.deepcopy(geo); outgeo.offOrigin=np.asarray(grid['center_zyx_mm'],np.float32)
        correct=project(reconstructed,outgeo,angles); wrong=project(reconstructed,geo,angles)
        self.assertLess(np.linalg.norm(correct-observed),np.linalg.norm(wrong-observed))
        source_center=np.array([(mu*a).sum()/mu.sum() for a in [z,y,x]])+geo.offOrigin
        positive=np.maximum(reconstructed,0)
        output_center=np.array([(positive*a).sum()/positive.sum() for a in [z,y,x]])
        np.testing.assert_allclose(output_center,source_center,atol=1.)

    def test_boundary_profiles(self):
        for profile in ['official_v1','pig_ordered_v1']:
            conv=create_model(2,profile).conv1_2.conv3
            with torch.no_grad(): conv.weight.fill_(1); conv.bias.zero_()
            x=torch.zeros(1,16,5,1,1); x[:,:,4]=1
            value=conv(x)[0,0,0,0,0].item()
            self.assertEqual(value,16 if profile=='official_v1' else 0)

    def test_no_valid_ssim_window(self):
        p=torch.rand(1,1,11,1,16,16,requires_grad=True); q=torch.rand_like(p)
        mask=torch.zeros_like(p,dtype=torch.bool); mask[...,::2,::2]=True
        result=compute_loss(p,q,mask,1,{'kind':'public_code','ssim_weight':.1})
        self.assertTrue(result['skipped_ssim']); result['total'].backward()
        self.assertTrue(torch.equal(p.grad[~mask],torch.zeros_like(p.grad[~mask])))
        with self.assertRaises(ValueError): compute_loss(p,q,torch.zeros_like(mask),1,{'kind':'l1'})

    def test_shared_resampling(self):
        a=np.arange(2*4*8*8,dtype=np.float32).reshape(2,4,8,8)
        meta=basic_metadata(2,[1,1,1])
        arrays={'cbct':a,'prior':a.mean(0),'target':a+2,'valid_mask':np.ones_like(a,dtype=np.uint8)}
        result,newmeta=resample_pair(arrays,meta,{'shape':[4,4,4],'spacing':[1,2,2]})
        self.assertEqual(result['cbct'].shape,(2,4,4,4))
        np.testing.assert_allclose((result['target']-result['cbct'])[result['valid_mask']>0],2,atol=1e-5)
        self.assertEqual(newmeta['output_spacing_zyx_mm'],[1,2,2])

    @unittest.skipUnless(torch.cuda.is_available(),'Real projection self-check needs TIGRE CUDA')
    def test_self_projection_no_duplicate_source_offset(self):
        s=SeriesSettings(voxel_spacing_multiplier=1,n_detector_rows=32,n_detector_cols=32,
            d_detector_row_mm=1,d_detector_col_mm=1,fov_output_pixels=16,fov_source_pixels=16,
            padding_row_pixels=0,padding_col_pixels=0,fov_mask_shrink_pixels=0)
        geo=geometry((16,16,16),(1,1,1),s); grid=output_grid(geo,s)
        z,y,x=np.indices((16,16,16))-7.5
        a=np.stack([(z*z+y*y+(x-v)**2<9).astype(np.float32)*.02 for v in [0,1]])
        angles=TigreTool.make_angles(20,0,300); labels=np.arange(20)%2+1
        mixed=np.zeros((20,32,32),np.float32)
        for t in [0,1]: mixed[labels==t+1]=project(a[t],geo,angles[labels==t+1])
        meta=basic_metadata(2,[1,1,1],geometry=grid,full_object_fov=True,
            full_fov_verified_from_source_support=True,source_offOrigin_zyx_mm=[0,3,4])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'projections.h5'
            with h5py.File(p,'w') as f:
                f['mixed_line_integral']=mixed; f['state_label']=labels; f['angles_rad']=angles
            result=projection_metrics(a,meta,p)
            self.assertLess(result['rmse'],1e-7)

    @unittest.skipUnless(torch.cuda.is_available(),'Overfit acceptance uses GPU')
    def test_two_stages_overfit_nonconstant_fixture(self):
        torch.manual_seed(2026); torch.set_num_threads(2)
        previous=None; results=[]
        for stage in [1,2]:
            model=create_model(stage).cuda()
            if previous is not None: initialize_stage2(model,previous)
            t,z,y,x=torch.meshgrid(torch.arange(10),torch.arange(1 if stage==1 else 4),torch.arange(16),torch.arange(16),indexing='ij')
            target=(.55+.3*torch.sin(x*.3+y*.17+t*.2+z*.3)).float().cuda()[None,None]
            degraded=target+.15*torch.sin((x*.9+t*.4).float()).cuda()[None,None]
            prior=degraded.mean(2,keepdim=True).expand_as(degraded)
            inp=torch.cat([degraded,prior],1).contiguous(); mask=torch.ones_like(target,dtype=torch.bool)
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.00015)
            losses=[]
            for step in range(300):
                optimizer.zero_grad(set_to_none=True); pred=forward_volume(model,inp)
                loss=compute_loss(pred,target,mask,stage,{'kind':'public_code','ssim_weight':.1})
                self.assertTrue(torch.isfinite(loss['total'])); loss['total'].backward(); optimizer.step()
                losses.append(float(loss['l1'].detach()))
                if len(losses)>=60 and np.mean(losses[-20:])<=.5*np.mean(losses[:20]): break
            self.assertLessEqual(np.mean(losses[-20:]),.5*np.mean(losses[:20]))
            previous={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            results.append({'stage':stage,'steps':len(losses),'first20_l1':float(np.mean(losses[:20])),'last20_l1':float(np.mean(losses[-20:]))})
        print('OVERFIT_RESULT '+json.dumps(results),flush=True)

if __name__=='__main__': unittest.main()
