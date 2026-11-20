import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.cardiac_ablation import write_matrix
from rstar4d.config import load_config, resolve_stage2, validate_train
from rstar4d.data.contracts import read_manifest
from rstar4d.data.sampling import InvalidCore, PatchDataset, validation_samples
from rstar4d.training.motion_losses import motion_terms
from rstar4d.training.cardiac_benchmark import benchmark_manifest
from validation.cardiac_smoke import fixture


CONFIGS = Path(__file__).resolve().parents[1] / 'configs/train'
PAPER = [[16,256,256],[32,144,144],[64,128,128]]


class CardiacSamplingTests(unittest.TestCase):
    def test_real_paper_sizes_in_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = read_manifest(fixture(Path(tmp)))[:1]
            path = rows[0]['path']
            volume = (66,260,264)
            with h5py.File(path,'r+') as f:
                meta = json.loads(f.attrs['metadata_json'])
                meta['geometry']['shape_zyx'] = list(volume)
                meta['geometry']['center_zyx_mm'] = ((np.array(volume)-1)/2).tolist()
                f.attrs['metadata_json'] = json.dumps(meta)
                for name in ('cbct','target','valid_mask','prior'):
                    del f[name]
                    shape = volume if name=='prior' else (2,*volume)
                    f.create_dataset(name,shape,dtype='u1' if name=='valid_mask' else 'f4',
                                     chunks=True,fillvalue=1 if name=='valid_mask' else .01)
                del f['roi_labels']
            cfg = load_config(CONFIGS/'pig_cardiac_ablation.yaml')
            for shape in PAPER:
                sample = dict(case_id=rows[0]['case_id'],origin=[1,2,3],shape=shape)
                item = PatchDataset(rows,cfg,2,1,1,[sample])[0]
                self.assertEqual(tuple(item['input'].shape),(2,2,*shape))

    def test_full_motion_training_step_in_manifest_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = fixture(Path(tmp))
            cfg = load_config(CONFIGS/'pig_cardiac_ablation.yaml')
            cfg.update(stage2_patches=[[6,16,16]],device='cpu',threads=1,amp=False)
            cfg['extensions']['motion'].update(mode='learned',surface_weight=.1)
            cfg['extensions']['prior']['mode'] = 'geometry'
            cfg['extensions']['boundary']['enabled'] = True
            result = benchmark_manifest(cfg,manifest,None,1)['results'][0]
            self.assertEqual(result['window_shape'],[6,16,16])
            self.assertEqual(result['measured_updates'],1)
            self.assertGreater(result['sample_diagnostics']['windows'],0)

    def test_actual_input_and_same_training_coordinates_across_eight_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = read_manifest(fixture(root))[:1]
            for preset in ('pig_cardiac_ablation.yaml','pig_cardiac_ablation_16gb.yaml'):
                configs = write_matrix(CONFIGS/preset, root/'matrix')
                self.assertEqual(len(configs),8)
                replay = None
                for arm, path in configs.items():
                    cfg = load_config(path)
                    self.assertEqual(cfg['stage2_patches'],PAPER)
                    # Same crop implementation, scaled down so all eight arms can read a real phantom.
                    cfg['stage2_patches'] = [[4,8,8],[6,16,16],[8,16,8]]
                    ds = PatchDataset(records,cfg,2,1,9)
                    observed = []
                    for i in range(len(ds)):
                        item = ds[i]
                        self.assertEqual(list(item['input'].shape[-3:]),item['patch_shape'].tolist(),arm)
                        observed.append((item['sampling_mode'],item['origin'].tolist(),item['patch_shape'].tolist()))
                    if replay is None:
                        replay = observed
                    self.assertEqual(observed,replay,arm)

    def test_training_mixture_and_unbiased_fixed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = read_manifest(fixture(Path(tmp)))[:1]
            cfg = load_config(CONFIGS/'pig_cardiac_ablation.yaml')
            cfg.update(stage2_patches=[[4,8,8]],stage1_patch=[1,8,8],validation_samples_per_case=5)
            ds = PatchDataset(rows,cfg,2,3,4000)
            counts = Counter()
            z = set()
            for i in range(len(ds)):
                rng = np.random.default_rng([cfg['seed'],3,i,2])
                mode = ds.sampling_mode(i)
                _,_,(depth,y,x) = ds.proposal(i,rng,mode=mode)
                counts[mode] += 1
                z.add(depth)
                if mode=='center':
                    self.assertTrue(4<=x<=12 and 4<=y<=12)
                if mode=='x_end':
                    self.assertEqual(x,16)
            for key,p in [('center',.70),('uniform',.25),('x_end',.05)]:
                self.assertAlmostEqual(counts[key]/len(ds),p,delta=.025)
            self.assertEqual(z,set(range(5)))
            uniform = {**cfg,'stage2_sampling':{'center':0.,'uniform':1.,'x_end':0.}}
            self.assertEqual(validation_samples(rows,cfg,2),validation_samples(rows,uniform,2))
            self.assertEqual(PatchDataset(rows,cfg,1,2,1).sampling_mode(0),'uniform')

    def test_motion_reference_can_cross_interior_cut_and_preserves_physical_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = read_manifest(fixture(Path(tmp)))[:1]
            cfg = load_config(CONFIGS/'pig_cardiac_ablation.yaml')
            cfg['stage2_patches'] = [[8,16,16]]
            cfg['extensions']['motion']['mode'] = 'learned'
            opts = resolve_stage2(cfg,2)
            for x0 in (0,8):
                sample = dict(case_id=rows[0]['case_id'],origin=[0,4,x0],shape=[8,16,16])
                item = PatchDataset(rows,cfg,2,1,1,[sample])[0]
                ctx = {k:v[None] for k,v in item['context'].items()}
                ref = {k:v[None] for k,v in item['motion_reference'].items()}
                flow = torch.zeros(1,1,3,4,4,4)
                flow[:,:,2] = 4.
                terms = motion_terms({'pairs':torch.tensor([[0,1]]),'flow_vox':flow},
                    {'target':item['target'][None],'source_path':[item['source_path']],'motion_reference':ref},
                    ctx,opts['motion'],opts['recipe'])
                self.assertGreater(terms['windows'],0)
                if x0==0:
                    self.assertEqual(terms['outside'],0)
                    self.assertGreater(terms['cross_core'],0)
                else:
                    self.assertGreater(terms['outside'],0)

    def test_invalid_probabilities_rejected(self):
        for weights in ((.8,.25,.05),(-.1,1.,.1),(float('nan'),1.,0.)):
            with self.assertRaisesRegex(ValueError,'stage2_sampling'):
                validate_train({'stage2_sampling':dict(zip(('center','uniform','x_end'),weights))})

    def test_failed_edge_proposals_retry_only_at_the_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = read_manifest(fixture(Path(tmp)))[:1]
            cfg = load_config(CONFIGS/'pig_cardiac_ablation.yaml')
            cfg.update(stage2_patches=[[4,8,8]],max_attempts=16,
                       stage2_sampling={'center':0.,'uniform':0.,'x_end':1.})
            with h5py.File(records[0]['path'],'r+') as f:
                f['valid_mask'][...,16:] = 0
            with self.assertRaises(InvalidCore):
                PatchDataset(records,cfg,2,3,1)[0]
