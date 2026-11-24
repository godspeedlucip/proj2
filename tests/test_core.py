import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from dpi_moco.config import read_config
from dpi_moco.data import phase_bins
from dpi_moco.inference import add_volume_noise, field_statistics, fuse_target, noise_seed
from dpi_moco.losses import contrastive_loss, critic_losses, field_smoothness, generator_losses, local_ncc, warp_pull
from dpi_moco.models import Critic2D, RegisNet3D, RestoreNet2D
from dpi_moco.physics import pigae
from dpi_moco.training import step_2d


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(14)

    def test_bins(self):
        cycles, phases = np.indices((4, 4)).reshape(2, -1)
        ids = np.arange(16)
        bins = phase_bins(ids, phases, cycles, 4)
        np.testing.assert_array_equal(bins["pseudo"][0], [0, 5, 10, 15])
        missing = ids != 5
        bins = phase_bins(ids[missing], phases[missing], cycles[missing], 4)
        for groups in bins.values():
            np.testing.assert_array_equal(np.sort(np.concatenate(groups)), ids[missing])
        with self.assertRaisesRegex(ValueError, "Empty real"):
            phase_bins(ids[phases != 0], phases[phases != 0], cycles[phases != 0], 4)

    def test_physical_modes(self):
        sparse, prior = np.array([90], np.float32), np.array([100], np.float32)
        for mode, expected in (("printed", 80), ("static_consistent", 100), ("off", 90)):
            self.assertEqual(pigae(sparse, prior, sparse, mode).item(), expected)
        self.assertEqual(read_config("configs/engineering.yaml")["fusion_mode"], "mean")

    def test_networks_and_critic_gradients(self):
        generator, denoise, critic, register = RestoreNet2D(), RestoreNet2D(), Critic2D(), RegisNet3D()
        counts = [sum(p.numel() for p in model.parameters() if p.requires_grad)
                  for model in (generator, denoise, critic, register)]
        self.assertEqual(counts, [7319297, 7319297, 1172225, 300547])
        odd = torch.randn(1, 1, 35, 49)
        with torch.no_grad():
            for model in (generator, denoise):
                model.head.weight.zero_()
                model.head.bias.zero_()
                torch.testing.assert_close(model.eval()(odd), odd, rtol=0, atol=0)
            self.assertNotEqual(generator.encoder[0].first[1].running_mean.data_ptr(),
                                denoise.encoder[0].first[1].running_mean.data_ptr())
            _, features = critic(critic.adapt(torch.zeros(1, 1, 128, 128)), True)
            self.assertEqual([tuple(x.shape[1:]) for x in features],
                             [(32, 64, 64), (64, 32, 32), (128, 16, 16), (256, 8, 8)])
        target, source = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
        prediction = (0.2 * source).requires_grad_()
        fake, real, degraded = map(critic.adapt, (prediction, target, source))
        discriminator = critic_losses(critic, fake, real, training=True)
        discriminator["critic"].backward()
        self.assertIsNone(prediction.grad)
        self.assertTrue(torch.isfinite(discriminator["gp"]))
        critic.zero_grad(set_to_none=True)
        critic.requires_grad_(False)
        metrics = generator_losses(critic, fake, real, degraded, "full")
        for name in ("adversarial", "contrastive"):
            gradient, = torch.autograd.grad(metrics[name], prediction, retain_graph=True)
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        zeros, ones = torch.zeros(2, 1, 1, 1), torch.ones(2, 1, 1, 1)
        self.assertEqual(contrastive_loss([zeros] * 4, [zeros] * 4, [ones] * 4)[0], 0)
        self.assertGreater(contrastive_loss([zeros] * 4, [ones] * 4, [zeros] * 4)[0], 0)

    def test_fixed_patch_overfit(self):
        model = RestoreNet2D()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        target = torch.rand(2, 1, 32, 32)
        source = target * 0.85 + 0.05
        losses = [float(step_2d({"model": model}, {"model": optimizer}, source, target,
                                "mse", train=True)[0]["mse"].detach()) for _ in range(14)]
        self.assertLess(losses[-1], losses[0] / 4)

    def test_warp_and_boundary_ncc(self):
        source = torch.zeros(1, 1, 17, 24, 53)
        source[0, 0, 6, 8, 40] = 1
        field = torch.zeros(1, 3, *source.shape[-3:])
        torch.testing.assert_close(warp_pull(source, field), source, atol=1e-5, rtol=0)
        for axis, target in enumerate(((6, 8, 44), (6, 12, 40), (10, 8, 40))):
            field.zero_()
            field[:, axis] = -4
            self.assertAlmostEqual(warp_pull(source, field)[0, 0][target].item(), 1, places=5)
            self.assertEqual(field_smoothness(field).item(), 0)
        moving, fixed = (torch.rand(1, 1, 3, 4, 5, dtype=torch.float64) for _ in range(2))
        padded = [F.pad(value[0, 0], (4,) * 6) for value in (moving, fixed)]
        explicit = []
        for z, y, x in np.ndindex((3, 4, 5)):
            windows = [value[z:z + 9, y:y + 9, x:x + 9].flatten() for value in padded]
            a, b = [window - window.mean() for window in windows]
            explicit.append((a * b).sum().square() / (a.square().sum() * b.square().sum() + 1e-5))
        torch.testing.assert_close(local_ncc(moving, fixed), -torch.stack(explicit).mean(), atol=1e-12, rtol=0)
        self.assertLess(local_ncc(moving, moving), local_ncc(moving, fixed))

    def test_noise(self):
        gamma = 0.0007
        volume = np.zeros((100, 32, 32), np.float32)
        seed = noise_seed("case", 1, 42)
        result = add_volume_noise(volume, gamma, seed)
        self.assertLess(abs(result.std() / gamma - 1), 0.02)
        np.testing.assert_array_equal(result, add_volume_noise(volume, gamma, seed))
        self.assertFalse(np.array_equal(result, add_volume_noise(volume, gamma, noise_seed("case", 2, 42))))

    def test_fusion_uses_corrected_pixels(self):
        with TemporaryDirectory() as directory:
            c_paths, b_paths = [], []
            for phase in range(10):
                c, b = (Path(directory) / f"{prefix}{phase}.npy" for prefix in ("c", "b"))
                np.save(c, np.full((5, 4, 7), 0.012, dtype=np.float32))
                np.save(b, np.full((5, 4, 7), 0.04 + 0.02 * phase, dtype=np.float32))
                c_paths.append(c)
                b_paths.append(b)
            counts = []

            def register(moving, fixed):
                self.assertGreater(float(moving.min()), 2)
                self.assertAlmostEqual(float(fixed.min()), 2)
                counts.append(1)
                return moving.new_zeros((1, 3, *moving.shape[-3:]))

            for config, expected in ((read_config(None), 0.120), (read_config("configs/engineering.yaml"), 0.012)):
                fused, extras = fuse_target(register, c_paths, b_paths, 0, config, torch.device("cpu"),
                                            alternatives={"d_from_b": b_paths})
                np.testing.assert_allclose(fused, expected, atol=1e-7)
                np.testing.assert_allclose(extras["d_zero_motion"], expected, atol=1e-7)
                self.assertGreater(extras["d_from_b"].min(), expected)
            self.assertEqual(len(counts), 18)
            fused, _ = fuse_target(register, c_paths[:1], b_paths[:1], 0, read_config(None), torch.device("cpu"))
            np.testing.assert_allclose(fused, 0.012)
        field = np.zeros((3, 11, 3, 5), dtype=np.float32)
        field[0] = 2
        qc = field_statistics(field, {"spacing_xyz_mm": [0.5, 1.0, 2.0]})
        self.assertEqual(qc["displacement_mm_percentiles"]["p95"], 1)
        self.assertEqual(qc["fold_fraction"], 0)
        field[0] = -2 * np.arange(5)
        self.assertEqual(field_statistics(field, {"spacing_xyz_mm": [1, 1, 1]})["fold_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
