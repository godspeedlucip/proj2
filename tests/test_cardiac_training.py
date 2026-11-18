import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rstar4d.config import validate_train
from rstar4d.data.contracts import basic_metadata, write_case, write_manifest
from rstar4d.training import trainer
from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.models import create_model
from rstar4d.utils.io import sha256


class CardiacTrainingTests(unittest.TestCase):
    def test_cross_core_warp_matches_full_volume_at_known_shift(self):
        from rstar4d.training.motion_losses import motion_terms
        torch.manual_seed(19)
        a = torch.rand(12,24,24)
        b = a.roll((1,2,3),dims=(0,1,2))
        target = torch.stack((a,b))[None,None]
        valid = torch.ones_like(target[:,0],dtype=torch.bool)
        motion = {'struct_weight':1.,'surface_weight':0.,'smooth_weight':0.}
        recipe = {'lncc_radius_vox':[1]*3,'edge_threshold':1e-4,'variance_threshold':1e-8}
        pairs = torch.tensor([[0,1],[1,0]])
        context = {'source_fov':valid,'origin':torch.zeros(1,3,dtype=torch.long),
                   'core_bounds':torch.tensor([[[5,8],[8,14],[9,14]]])}
        truth = torch.tensor([[[1.,2.,3.],[-1.,-2.,-3.]]])[...,None,None,None]
        full = motion_terms({'pairs':pairs,'flow_vox':truth.expand(-1,-1,-1,6,6,6)},
                            {'target':target},context,motion,recipe)
        small = (slice(2,12),slice(4,24),slice(4,24))
        offset = torch.tensor([[2,4,4]])
        cropped = {'source_fov':valid[(...,*small)],'origin':offset,
                   'core_bounds':context['core_bounds']-offset[...,None]}
        warp = motion_terms({'pairs':pairs,'flow_vox':truth.expand(-1,-1,-1,5,5,5)},
                            {'target':target[(...,*small)]},cropped,motion,recipe)
        zero = motion_terms({'pairs':pairs,'flow_vox':torch.zeros(1,2,3,5,5,5)},
                            {'target':target[(...,*small)]},cropped,motion,recipe)
        self.assertEqual(warp['windows'],full['windows'])
        self.assertGreater(warp['cross_core'],0)
        self.assertEqual(warp['outside'],0)
        torch.testing.assert_close(warp['struct'],full['struct'],atol=1e-5,rtol=1e-4)
        self.assertLess(float(warp['struct']),float(zero['struct']))

    def test_structure_out_of_fov_retains_window_denominator_and_gradient(self):
        self.assertTrue(hasattr(trainer,'compute_batch'))
        from rstar4d.training.motion_losses import motion_terms
        torch.manual_seed(8)
        t = torch.rand(1,1,1,8,12,12).repeat(1,1,2,1,1,1)
        fov = torch.ones(1,2,8,12,12,dtype=torch.bool)
        context = {'source_fov':fov,'core_bounds':torch.tensor([[[2,6],[3,9],[3,9]]]),
                   'origin':torch.zeros(1,3,dtype=torch.long)}
        spec = {'struct_weight':1.,'surface_weight':0.,'smooth_weight':0.}
        recipe = {'lncc_radius_vox':[1]*3,'edge_threshold':1e-4,'variance_threshold':1e-8,'match_chunk_size':64}
        pairs = torch.tensor([[0,1],[1,0]])
        baseline = motion_terms({'pairs':pairs,'flow_vox':torch.zeros(1,2,3,4,3,3)}, {'target':t},context,spec,recipe)
        displacement = torch.tensor(14.,requires_grad=True)
        flow = torch.zeros(1,2,3,4,3,3)
        flow[:,:,2] = displacement
        shifted = motion_terms({'pairs':pairs,'flow_vox':flow},{'target':t},context,spec,recipe)
        self.assertEqual(baseline['windows'],shifted['windows'])
        self.assertGreater(shifted['windows'],0)
        self.assertLess(baseline['struct'].item(),1e-4)
        self.assertGreater(shifted['struct'].item(),2.)
        shifted['struct'].backward()
        self.assertGreater(displacement.grad.item(),0.)

    def test_stage2_normal_entry_and_strict_resume(self):
        self.assertTrue(hasattr(trainer,'compute_batch'))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            shape = (5,16,16)
            affine = np.eye(4)
            meta = basic_metadata(2,(1.,1.,1.),geometry={'shape_zyx':list(shape),'voxel_spacing_zyx_mm':[1.]*3,
                'voxel_to_scanner_zyx':affine.tolist(),'center_zyx_mm':((np.asarray(shape)-1)/2).tolist()})
            rng = np.random.default_rng(5)
            rows = []
            for case in ('train','val'):
                path = root/(case+'.h5')
                x = rng.random((2,*shape),dtype=np.float32)*.02
                write_case(path,x,x[0],x*.9,np.ones_like(x),meta)
                rows.append({'case_id':case,'animal_id':case,'path':str(path),'sha256':sha256(path)})
            manifest = root/'manifest.jsonl'
            write_manifest(manifest,rows)
            split = {'train':['train'],'val':['val'],'test':[]}
            init = root/'stage1.pt'
            torch.save(create_model(1).state_dict(),init)
            cfg = validate_train({'device':'cpu','amp':False,'threads':2,'stage2_patches':[[4,12,12]],
                'stage2_crop_mode':'core_with_context','epochs':{'2':2},'steps_per_epoch':{'2':1},'loss':{'kind':'l1'},
                'extensions':{'motion':{'mode':'learned'},'prior':{'mode':'image_only'}}})
            trainer.train(cfg,manifest,split,root/'run',2,init={'path':init,'source_stage':1},stop_after_epoch=1)
            last = root/'run/checkpoints/last.pt'
            first = load_checkpoint(last)
            self.assertEqual(first['stage2_resolved']['crop_mode'],'core_with_context')
            self.assertEqual(first['step'],1)
            trainer.train(cfg,manifest,split,root/'run',2,resume=last)
            final = load_checkpoint(last)
            self.assertEqual(final['step'],2)
            self.assertTrue(any(k.startswith('motion.') for k in final['state_dict']))
