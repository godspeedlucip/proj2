import json
import unittest
from pathlib import Path
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ssim

from rstar4d.models import create_model, forward_volume, initialize_stage2
from rstar4d.training.losses import compute_loss
from rstar4d.config import train_defaults, validate_train

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

class ModelTests(unittest.TestCase):
    def test_weights_and_exact_notebook_forward(self):
        for stage in [1,2]:
            cells=json.loads((ROOT/f'RSTAR/githubcode/Training_Tetris{stage}.ipynb').read_text(encoding='utf-8'))['cells']
            ns={'torch':torch,'nn':nn,'F':F}
            for i in [8,9,10,11]: exec(''.join(cells[i]['source']),ns)
            reference=ns['myUNet']().eval()
            state=torch.load(ROOT/f'RSTAR/githubcode/checkpoints/checkpoints_Tetris{stage}.pth',map_location='cpu',weights_only=True)
            reference.load_state_dict(state)
            actual=create_model(stage).eval(); actual.load_state_dict(state,strict=True)
            self.assertEqual(sum(p.numel() for p in actual.parameters()),2118693)
            self.assertEqual(len(actual.state_dict()),232)
            for t in [10,11]:
                x=torch.rand(1,2,t,1 if stage==1 else 4,16,16)
                with torch.no_grad(): a=actual(x); b=reference(x)
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_stage_transition_preserves_other_weights(self):
        source=create_model(1).state_dict(); model=create_model(2)
        initialize_stage2(model,source)
        for k,v in model.state_dict().items():
            if '.conv2.' not in k and '.conv2_2.' not in k:
                self.assertTrue(torch.equal(v,source[k]),k)
        x=torch.rand(1,2,10,4,16,16)
        model(x).sum().backward()
        self.assertIsNotNone(model.conv1.conv2.weight.grad)
        stage1=create_model(1); stage1(x[:,:,:,:1]).sum().backward()
        self.assertIsNone(stage1.conv1.conv2.weight.grad)

    def test_public_loss_exact_and_mask_gradient(self):
        for stage in [1,2]:
            for t in [10,11]:
                pred=torch.rand(1,1,t,1 if stage==1 else 12,16,16,requires_grad=True)
                target=torch.rand_like(pred); valid=torch.ones_like(pred,dtype=torch.bool)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    a=compute_loss(pred,target,valid,stage,{'kind':'public_code','ssim_weight':.1})['total']
                    p=pred if stage==1 else pred.squeeze(1)
                    q=target if stage==1 else target.squeeze(1)
                    b=(pred-target).abs().mean()+.1*(1-ssim(p*1000,q*1000))
                torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
                valid[...,0,0]=False
                loss=compute_loss(pred,target,valid,stage,{'kind':'public_code','ssim_weight':.1})['total']
                loss.backward()
                self.assertTrue(torch.equal(pred.grad[...,0,0],torch.zeros_like(pred.grad[...,0,0])))

    def test_odd_shape_and_config(self):
        model=create_model(2).eval()
        with torch.no_grad(): out=forward_volume(model,torch.rand(1,2,11,5,17,19))
        self.assertEqual(tuple(out.shape),(1,1,11,5,17,19))
        cfg=train_defaults(); self.assertEqual(cfg['epochs'],{'1':60,'2':30})
        cfg['unknown_typo']=True
        with self.assertRaises(ValueError): validate_train(cfg)

if __name__=='__main__': unittest.main()
