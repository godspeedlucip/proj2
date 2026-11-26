import unittest

import numpy as np
import torch
from rstar4d.models.model import create_model, is_z_parameter

from rstar_decnn.backbone import restore_backbone
from rstar_decnn.decnn import DeCNN3D, WIDTHS
from rstar_decnn.model import Combined
from rstar_decnn.predict import decnn_volume
from rstar_decnn.train import _fresh, optimizer_for, read_config, transition, update
from .engineering_smoke import FIXTURES, ROOT


def prepare_torch():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


@unittest.skipUnless(torch.cuda.is_available(), "full-width tests require CUDA")
class ModelContracts(unittest.TestCase):
    def setUp(self):
        prepare_torch()
        self.device = torch.device("cuda")

    def test_independent_encoders_four_skip_paths_and_odd_grid(self):
        model = DeCNN3D().to(self.device)
        for g, p in zip(model.gated.parameters(), model.prior.parameters()):
            self.assertNotEqual(g.data_ptr(), p.data_ptr())
        observed = []
        hooks = [level.register_forward_pre_hook(
            lambda module, args: observed.append(tuple(x.shape[1] for x in args))
        ) for level in model.decoder]
        x = torch.randn(1, 1, 16, 32, 16, device=self.device)
        prior = torch.randn_like(x)
        y = model(x, prior)
        (y - 0.6).square().mean().backward()
        expected = [(2 * WIDTHS[-1], 256, 256), (256, 128, 128), (128, 64, 64), (64, 32, 32)]
        self.assertEqual(observed, expected)
        for branch in (model.gated, model.prior, model.decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters()))
        for hook in hooks:
            hook.remove()
        model.eval()
        rng = np.random.default_rng(2026)
        scan = rng.uniform(0, 0.05, (3, 17, 17, 19)).astype(np.float32)
        prior = scan.mean(0)
        cfg = dict(mode="full", tile=(16, 16, 16), overlap=(0, 0, 0), device=self.device, amp=False)
        transform = {"scale": 0.02, "offset": 0}
        original = decnn_volume(model, scan, prior, transform, **cfg)
        scan[[0, 2]] += 0.02
        result = decnn_volume(model, scan, prior, transform, **cfg)
        self.assertEqual(original.shape, scan.shape)
        np.testing.assert_array_equal(original[1], result[1])

    def test_stage1_to_stage2_copies_shared_and_initializes_z(self):
        config = read_config(ROOT / "configs" / "tetris.yaml")
        backbone = create_model(1)
        with torch.no_grad():
            for param in backbone.parameters():
                param.fill_(0.12)
        movie = Combined(backbone, 0.0, config["upsampling"], decnn=False)
        before = {k: p.clone() for k, p in backbone.state_dict().items()}
        self.assertIsNone(movie.decnn)
        self.assertIsNone(getattr(backbone, "stam", None))
        optimizer = optimizer_for(movie, config)
        self.assertEqual([g["name"] for g in optimizer.param_groups], ["backbone"])
        spec = {"stage": 1, "model_profile": "official_v1"}
        upgraded, spec, opt, steps = transition(movie, spec, optimizer, config, config["stage1_steps"], 4, self.device)
        self.assertEqual(steps, 0)
        self.assertEqual(spec["stage"], 2)
        self.assertEqual([g["name"] for g in opt.param_groups], ["decnn", "backbone", "stam", "sam", "alpha"])
        for key, original in before.items():
            actual = upgraded.backbone.state_dict()[key].cpu()
            if not is_z_parameter(key):
                self.assertTrue(torch.equal(original, actual), key)
            elif key in [f"conv{i}.conv2.weight" for i in range(1, 5)]:
                expected = torch.empty_like(actual)
                torch.nn.init.dirac_(expected)
                self.assertTrue(torch.equal(expected, actual), key)
            else:
                self.assertEqual(float(actual.abs().sum()), 0.0, key)

    def test_random_stam_receives_gradient_after_output_map_update(self):
        config = read_config(ROOT / "configs" / "scratch_stage2.yaml")
        config.update(phase_batch=1, loss={"kind": "l1"}, decnn_recompute=True)
        meta = {"normalization": {"scale": 0.02, "offset": 0}, "intensity_domain": "mu",
                "calibration_application": "none"}
        model, spec, _ = _fresh(config, [{"metadata": meta}], {})
        model.to(self.device)
        model.decnn.head[0].bias.data.fill_(0.5)
        opt = optimizer_for(model, config)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        x = torch.rand(1, 1, 2, 16, 16, 32, device=self.device) * 0.05
        batch = {"x": x, "a_s": x.mean(2), "a_d": x.mean(2) * 0.95, "y": x * 0.8 + 0.01,
                 "valid": torch.ones_like(x, dtype=torch.bool)}
        sam = model.backbone.stam.sam.attn.qkv.weight
        old_sam = sam.detach().clone()
        self.assertTrue(torch.count_nonzero(model.backbone.stam.output_projection.weight) == 0)
        self.assertTrue(update(model, batch, spec, opt, scaler, config)[2])
        self.assertTrue(torch.equal(sam, old_sam))
        self.assertFalse(torch.count_nonzero(model.backbone.stam.output_projection.weight) == 0)
        self.assertIsNotNone(model.alpha.grad)
        self.assertTrue(update(model, batch, spec, opt, scaler, config)[2])
        self.assertGreater(float(sam.grad.norm()), 0)
        self.assertFalse(torch.equal(sam, old_sam))
        self.assertGreater(float(model.backbone.stam.temporal_adapter.down.weight.grad.norm()), 0)
        self.assertEqual(int(model.decnn.gated.blocks[0][0][1].num_batches_tracked), 4)

    @unittest.skipUnless(FIXTURES.is_dir(), "upstream STAM smoke fixtures are unavailable")
    def test_unfreeze_preserves_prediction_momentum_and_alpha(self):
        path = FIXTURES / "joint" / "checkpoints" / "best.pt"
        backbone, spec, _ = restore_backbone(path, joint=True, prior_mode="all_views_fdk", engineering=True)
        config = read_config(ROOT / "configs" / "scratch_stage2.yaml")
        config.update(phase_batch=1, warmup_steps=1, milestones=[1], alpha=0.35)
        model = Combined(backbone, config["alpha"], config["upsampling"]).to(self.device)
        model.set_policy("preheat")
        opt = optimizer_for(model, config)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        x = torch.rand(1, 1, 2, 16, 16, 32, device=self.device) * 0.05
        batch = {"x": x, "a_s": x.mean(2), "a_d": x.mean(2), "y": x * 0.8 + 0.01,
                 "valid": torch.ones_like(x, dtype=torch.bool)}
        update(model, batch, spec, opt, scaler, config)
        parameter = next(model.decnn.parameters())
        moment = opt.state[parameter]["exp_avg"]
        before = model.eval()
        with torch.no_grad():
            s = before.normalized_backbone(batch, spec["normalization"])
            d = before.normalized_decnn(x[:, :, 0], batch["a_d"], config["transform"])
        updated, _, same_opt, count = transition(model, spec, opt, config, 1, 1, self.device)
        self.assertIs(same_opt, opt)
        self.assertIs(opt.state[parameter]["exp_avg"], moment)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(updated.alpha), 0.35)
        self.assertEqual([g["name"] for g in opt.param_groups], ["decnn", "backbone", "stam", "alpha"])
        self.assertEqual(len({id(p) for g in opt.param_groups for p in g["params"]}),
                         sum(len(g["params"]) for g in opt.param_groups))
        with torch.no_grad():
            torch.testing.assert_close(s, updated.normalized_backbone(batch, spec["normalization"]), rtol=1e-5, atol=5e-6)
            torch.testing.assert_close(d, updated.normalized_decnn(x[:, :, 0], batch["a_d"], config["transform"]), rtol=1e-5, atol=5e-6)
        update(updated, batch, spec, opt, scaler, config)
        self.assertTrue(all(p.grad is None for p in updated.backbone.stam.sam.parameters()))
        for p in (updated.backbone.conv1.conv1.weight, updated.backbone.stam.output_projection.weight, parameter):
            self.assertGreater(float(p.grad.norm()), 0)


if __name__ == "__main__":
    unittest.main()
