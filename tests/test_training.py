import copy
import tempfile
import unittest
from pathlib import Path
import torch
from tests.test_data import fixture
from rstar4d.data.contracts import write_manifest
from rstar4d.config import train_defaults
from rstar4d.training.trainer import train
from rstar4d.training.pipeline import pipeline
from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.utils.io import atomic_json


def inputs(root):
    rows=[fixture(root,10,'a','a'),fixture(root,11,'b','b'),fixture(root,10,'c','c')]
    write_manifest(root/'manifest.jsonl',rows)
    atomic_json(root/'split.json',{'train':['a'],'val':['b'],'test':['c']})
    cfg=train_defaults(); cfg.update(device='cpu',amp=False,deterministic=True,threads=1,
        epochs={'1':2,'2':2},steps_per_epoch={'1':2,'2':2},stage1_patch=[1,16,16],stage2_patches=[[4,16,16]],
        loss={'kind':'l1','ssim_weight':0.,'window':[0,.05]})
    return cfg,root/'manifest.jsonl',root/'split.json'


class TrainingTests(unittest.TestCase):
    def test_nonfinite_output_records_failure_without_checkpoint(self):
        from unittest.mock import patch
        from rstar4d.utils.io import read_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            with patch('rstar4d.training.trainer.forward_volume',side_effect=lambda model,x:x[:,:1]*float('nan')):
                with self.assertRaises(FloatingPointError): train(cfg,manifest,split,root/'failed',1)
            self.assertIn('Nonfinite loss',read_json(root/'failed/error.json')['error'])
            self.assertFalse((root/'failed/checkpoints/best.pt').exists())
            self.assertFalse((root/'failed/checkpoints/last.pt').exists())

    def test_accumulation_final_short_group_has_full_weight(self):
        from unittest.mock import patch
        from rstar4d.data.contracts import read_manifest
        from rstar4d.data.sampling import PatchDataset
        class LinearProbe(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(()))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            cfg.update(accumulation=2,epochs={'1':1,'2':1},steps_per_epoch={'1':3,'2':3})
            rows=[r for r in read_manifest(manifest) if r['animal_id']=='a']
            dataset=PatchDataset(rows,cfg,1,1,3)
            gradients=[float(dataset[i]['input'][:1].mean()) for i in range(3)]
            def optimizer(params,lr,**kwargs): return torch.optim.SGD(params,lr=lr)
            with patch('rstar4d.training.trainer.create_model',side_effect=lambda *args:LinearProbe()), \
                 patch('rstar4d.training.trainer.forward_volume',side_effect=lambda model,x:x[:,:1]*model.weight), \
                 patch('rstar4d.training.trainer.torch.optim.Adam',side_effect=optimizer):
                train(cfg,manifest,split,root/'accumulation',1)
            actual=load_checkpoint(root/'accumulation/checkpoints/last.pt')
            expected=cfg['learning_rate']*((gradients[0]+gradients[1])/2+gradients[2])
            self.assertAlmostEqual(actual['state_dict']['weight'].item(),expected,places=9)
            self.assertEqual(actual['step'],2)

    def test_exact_epoch_resume_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            from rstar4d.data.contracts import read_manifest
            rows=read_manifest(manifest); rows.append(fixture(root,10,'a','a_second'))
            write_manifest(manifest,rows)
            train(cfg,manifest,split,root/'full',1)
            train(cfg,manifest,split,root/'resumed',1,stop_after_epoch=1)
            write_manifest(manifest,rows[::-1])
            ck=root/'resumed/checkpoints/last.pt'
            train(cfg,manifest,split,root/'resumed',1,resume=ck)
            a=load_checkpoint(root/'full/checkpoints/last.pt'); b=load_checkpoint(ck)
            for k,v in a['state_dict'].items(): self.assertTrue(torch.equal(v,b['state_dict'][k]),k)
            self.assertEqual(a['scheduler'],b['scheduler'])
            for i,state in a['optimizer']['state'].items():
                for k,v in state.items():
                    if torch.is_tensor(v): self.assertTrue(torch.equal(v,b['optimizer']['state'][i][k]))
            changed=copy.deepcopy(cfg); changed['learning_rate']*=2
            with self.assertRaises(ValueError): train(changed,manifest,split,root/'resumed',1,resume=ck)

    def test_two_stage_pipeline_reentry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            result=pipeline(cfg,manifest,split,root/'run')
            last=load_checkpoint(root/'run/stage2/checkpoints/last.pt')
            self.assertEqual(last['stage'],2); self.assertTrue(last['stage_complete'])
            self.assertEqual(last['initialization']['source_stage'],1)
            self.assertTrue(Path(result['checkpoint']).is_file())
            second=pipeline(cfg,manifest,split,root/'run')
            self.assertEqual(result['checkpoint'],second['checkpoint'])

if __name__=='__main__': unittest.main()
