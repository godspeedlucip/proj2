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
    def test_raw_geometry_fallback_requires_the_measured_projection_source_and_angles(self):
        import h5py
        from rstar4d.data.acquisition import angle_summary, load_geometry
        from rstar4d.utils.io import canonical_hash
        angles = np.deg2rad([5.,35.,65.,95.,125.,155.])
        labels = [2,9,2,9,2,9]
        phases = [2,9]
        selected = [0,2]
        expected = angle_summary(angles,labels,phases,selected_indices=selected,selected_phase=2)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/f'scan-{i}.h5' for i in range(2)]
            for i,path in enumerate(paths):
                with h5py.File(path,'w') as f:
                    f['angles_rad'] = angles
                    f['state_label'] = labels
                    # Different scans can have identical angular summaries.
                    f['mixed_line_integral'] = np.full((len(angles),1,1),i,np.float32)
            layout = {'source_projections_sha256':sha256(paths[0]),'phase':2,'actual_indices':selected,
                      'correct_angles_sha256':canonical_hash(expected.tolist())}
            record = {'case_id':'phantom','projections':paths[0],'projections_sha256':sha256(paths[0]),
                      'cardiac_layout':layout}
            np.testing.assert_array_equal(load_geometry(record,phases),expected)
            with self.assertRaisesRegex(ValueError,'projection source'):
                load_geometry(dict(record,projections=paths[1],projections_sha256=sha256(paths[1])),phases)
            with self.assertRaisesRegex(ValueError,'selected projection'):
                load_geometry(dict(record,cardiac_layout=dict(layout,actual_indices=[2,4])),phases)
            # Older paired layouts predate the optional digest of the selected angles.
            older = {key:value for key,value in layout.items() if key!='correct_angles_sha256'}
            np.testing.assert_array_equal(load_geometry(dict(record,cardiac_layout=older),phases),expected)

    def test_all_off_dataset_and_tiler_check_actual_read_and_allow_scan_padding(self):
        from rstar4d.config import resolve_stage2
        from rstar4d.data.sampling import PatchDataset
        from rstar4d.inference.sliding_window import predict_volume
        class Identity(torch.nn.Module):
            stage = 2
            def forward(self,x):
                return x[:,:1]
        tile = (6,16,16)
        cfg = validate_train({'stage2_crop_mode':'core_with_context','small_volume':'pad',
                              'stage2_patches':[list(tile)],'stage2_context':{'support_margin_vox':[0]*3}})
        with tempfile.TemporaryDirectory() as folder:
            for key,shape in [('interior',(6,16,96)),('edge',(5,13,15))]:
                path = Path(folder)/(key+'.h5')
                image = np.zeros((2,*shape),np.float32)
                valid = np.ones_like(image,bool)
                meta = basic_metadata(2,[1.]*3,geometry={
                    'shape_zyx':list(shape),'voxel_spacing_zyx_mm':[1.]*3,
                    'voxel_to_scanner_zyx':np.eye(4).tolist(),'center_zyx_mm':((np.asarray(shape)-1)/2).tolist()})
                write_case(path,image,image[0],image,valid,meta)
                sample = {'case_id':key,'origin':[0,0,0],'shape':tile}
                dataset = PatchDataset([{'case_id':key,'animal_id':key,'path':str(path)}],cfg,2,0,1,[sample])
                def infer():
                    return predict_volume(Identity(),image,image[0],cfg['normalization'],tile=tile,overlap=(0,0,0),
                                          recipe=resolve_stage2(cfg,2),context={'source_fov':valid,'phase_labels':[1,2],
                                                                               'case_id':key})
                if key=='interior':
                    for read in (lambda:dataset[0],infer):
                        with self.assertRaisesRegex(ValueError,'interior.*X right.*vox'):
                            read()
                else:
                    batch = dataset[0]
                    self.assertEqual(batch['context']['source_fov'].shape[1:],tile)
                    self.assertEqual(int(batch['valid'].sum()),valid.size)
                    np.testing.assert_array_equal(infer(),image)

    def test_4096_known_shift_windows_full_and_checked_crop(self):
        from rstar4d.config import load_config, resolve_stage2
        from rstar4d.data.context import plan_window, check_support, Window
        from rstar4d.training.motion_losses import motion_terms
        torch.set_num_threads(2)
        cfg = load_config(Path(__file__).resolve().parents[1]/'configs/train/pig_cardiac_ablation_16gb.yaml')
        cfg['extensions']['motion']['mode'] = 'learned'
        options = resolve_stage2(cfg,2)
        recipe = options['recipe']
        zero_margin = {**recipe,'support_margin_vox':[0]*3}
        def score(window,target,axis):
            slices = window.slices
            target_w = target[(...,*slices)][None,None]
            ctx = {'origin':torch.tensor([window.origin]),
                   'source_fov':torch.ones_like(target_w[:,0],dtype=torch.bool),
                   'core_bounds':torch.tensor([[(s.start,s.stop) for s in window.core]])}
            grid = tuple(n//s for n,s in zip(target_w.shape[-3:],(2,4,4)))
            flow = torch.zeros(1,2,3,*grid)
            flow[0,0,axis] = 8.
            flow[0,1,axis] = -8.
            return motion_terms({'pairs':torch.tensor([[0,1],[1,0]]),'flow_vox':flow},
                                {'target':target_w},ctx,options['motion'],recipe)
        core = (8,16,16)
        for shape,origin,axis,seed in [((64,48,48),(24,16,16),0,43),((32,32,152),(8,8,62),2,47)]:
            with self.subTest(shape=shape):
                valid = np.ones((2,*shape),bool)
                texture = torch.rand(shape,generator=torch.Generator().manual_seed(seed))
                target = torch.stack((texture,texture.roll(8,axis)))
                short = plan_window(shape,origin,core,zero_margin)
                with self.assertRaisesRegex(ValueError,'translated.*vox'):
                    check_support(short,zero_margin,valid,'translated')
                read = plan_window(shape,origin,core,recipe)
                check_support(read,recipe,valid,'translated')
                full = score(Window((0,0,0),shape,origin,core),target,axis)
                crop = score(read,target,axis)
                # Directly score the invalid W to reproduce the old derivative at its edge.
                unchecked = score(short,target,axis)
                self.assertEqual([row['windows'] for row in (full,crop,unchecked)],[4096]*3)
                self.assertGreater(float(unchecked['struct']-full['struct']),1e-4)
                self.assertEqual((full['outside'],crop['outside']),(0,0))
                torch.testing.assert_close(crop['struct'],full['struct'],atol=1e-5,rtol=1e-4)
                if axis==0:
                    self.assertEqual((short.origin[0],short.stop[0]),(14,42))
                    self.assertAlmostEqual(float(unchecked['struct']),.0294358,places=6)
                else:
                    self.assertGreater(read.origin[axis],0)
                    self.assertLess(read.stop[axis],shape[axis])

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
