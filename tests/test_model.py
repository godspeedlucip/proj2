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

    def test_stam_config_and_original_stage2_equivalence(self):
        import hashlib
        from rstar4d.config import load_config
        from rstar4d.models import attach_stam

        original_fixture = json.loads((ROOT/'validation/stam_baseline_fp32.json').read_text())
        state = torch.load(ROOT/original_fixture['checkpoint'], map_location='cpu', weights_only=True)
        cfgs = [load_config(ROOT/f'configs/train/pig_stam_{policy}.yaml') for policy in ('frozen', 'joint')]
        for cfg in cfgs:
            self.assertEqual(cfg['stam']['warmup_epochs'], 0)
        self.assertEqual(cfgs[0]['stam'], dict(cfgs[1]['stam'], train_policy='frozen'))
        with self.assertRaisesRegex(ValueError, 'Stage II'):
            validate_train(cfgs[0], 1)
        with self.assertRaisesRegex(ValueError, 'backbone_lr_ratio'):
            validate_train(dict(cfgs[1], stam=dict(cfgs[1]['stam'], backbone_lr_ratio=float('nan'))), 2)

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            models = {}
            for mode in ('off', 'joint', 'temporal_only'):
                torch.manual_seed(91)
                model = create_model(2)
                model.load_state_dict(state, strict=True)
                attach_stam(model, {'mode': mode})
                model.eval()
                models[mode] = model
                self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
                if model.stam is not None:
                    expected = 494912 if mode == 'joint' else 395776
                    self.assertEqual(sum(p.numel() for p in model.stam.parameters() if p.requires_grad), expected)
            for key, value in models['temporal_only'].stam.state_dict().items():
                self.assertTrue(torch.equal(value, models['joint'].stam.state_dict()[key]), key)
            for phases in (10, 11):
                x = torch.linspace(-.3, .9, 2 * phases * 3 * 17 * 19).reshape(1, 2, phases, 3, 17, 19)
                with torch.no_grad():
                    baseline = forward_volume(models['off'], x)
                    for mode in ('joint', 'temporal_only'):
                        self.assertTrue(torch.equal(forward_volume(models[mode], x), baseline))
                if torch.__version__ == original_fixture['torch']:
                    self.assertEqual(hashlib.sha256(baseline.numpy().tobytes()).hexdigest(),
                                     original_fixture['outputs'][str(phases)]['sha256'])
        finally:
            torch.set_num_threads(previous_threads)

    def test_stam_nonzero_axis_reference(self):
        from rstar4d.models.stam import STAMJointBlock

        def reference(block, features):
            expected = torch.empty_like(features)
            for b in range(features.shape[0]):
                for z in range(features.shape[3]):
                    f = features[b, :, :, z].permute(1, 2, 3, 0)
                    u = block.input_projection(f)
                    spatial = u.clone()
                    if block.spatial_adapter is not None:
                        for t in range(u.shape[0]):
                            spatial[t] += block.spatial_adapter(block.sam.attn(block.sam.norm1(u[t:t+1])))[0]
                    temporal = torch.empty_like(u)
                    for y in range(u.shape[1]):
                        for x in range(u.shape[2]):
                            sequence = u[:, y, x][None, :, None]
                            residual = block.temporal_adapter(block.sam.attn(block.sam.norm1(sequence)))
                            temporal[:, y, x] = (sequence + residual)[0, :, 0]
                    fused = spatial + temporal
                    norm = block.sam.norm2(fused)
                    addition = block.output_projection(fused + block.sam.mlp(norm) + block.fusion_adapter(norm))
                    expected[b, :, :, z] = (f + addition).permute(3, 0, 1, 2)
            return expected

        torch.manual_seed(94)
        for mode in ('joint', 'temporal_only'):
            block = STAMJointBlock(mode, z_chunk_size=0).eval()
            with torch.no_grad():
                block.output_projection.weight.normal_(std=.002)
                block.sam.attn.rel_pos_h.normal_(std=.03)
                block.sam.attn.rel_pos_w.normal_(std=.03)
                for phases in (10, 11):
                    features = torch.randn(2, 128, phases, 2, 2, 3)
                    actual = block(features)
                    torch.testing.assert_close(actual, reference(block, features), rtol=1e-5, atol=1e-6)
                    block.z_chunk_size = 1
                    torch.testing.assert_close(block(features), actual, rtol=1e-5, atol=1e-6)
                    updated = features.clone()
                    updated[0, :, :, 0].neg_()
                    changed = block(updated)
                    torch.testing.assert_close(changed[1], actual[1], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(changed[0, :, :, 1], actual[0, :, :, 1], rtol=1e-5, atol=1e-6)
                    block.z_chunk_size = 0

    def test_stam_checks_sam_block_keys_and_shapes(self):
        from rstar4d.models.stam import SAMBlock
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as d:
            path = Path(d) / 'sam.pth'
            sam = SAMBlock()
            keys = {'image_encoder.blocks.5.' + key: value for key, value in sam.state_dict().items()}
            torch.save(dict(keys, unrelated=torch.zeros(1)), path)
            self.assertEqual(len(sam.load_pretrained(path)['sha256']), 64)
            wrong = dict(keys)
            wrong['image_encoder.blocks.5.attn.rel_pos_h'] = torch.zeros(27, 64)
            wrong['image_encoder.blocks.5.extra'] = torch.zeros(1)
            del wrong['image_encoder.blocks.5.mlp.lin1.weight']
            torch.save(wrong, path)
            with self.assertRaisesRegex(ValueError, 'missing=.*mlp.lin1.weight.*unexpected=.*extra.*invalid=.*rel_pos_h'):
                sam.load_pretrained(path)

if __name__=='__main__': unittest.main()
