"""SSIM values and gradients must not depend on the model's TF32 setting."""
import unittest
import warnings

import torch
from pytorch_msssim import ssim

from rstar4d.training.losses import compute_loss


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA precision regression')
class LossPrecisionTests(unittest.TestCase):
    def test_public_ssim_matches_double_reference_with_tf32_enabled(self):
        generator = torch.Generator().manual_seed(73)
        prediction = 0.8 + 0.004 * torch.randn(1, 1, 10, 1, 256, 256, generator=generator)
        target = 1.0 + 0.004 * torch.randn(prediction.shape, generator=generator)
        reference = prediction.double().requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            expected_ssim = ssim(reference*1000, target.double()*1000, data_range=255)
        expected = (reference-target.double()).abs().mean() + 0.1*(1-expected_ssim)
        expected_gradient = torch.autograd.grad(expected, reference)[0]
        for stage in (1, 2):
            with self.subTest(stage=stage), torch.backends.cudnn.flags(enabled=True, allow_tf32=True, benchmark=True):
                actual_input = prediction.cuda().requires_grad_(True)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    actual = compute_loss(actual_input, target.cuda(), torch.ones_like(target, dtype=torch.bool).cuda(),
                                          stage, {'kind':'public_code', 'ssim_weight':0.1})
                actual_gradient = torch.autograd.grad(actual['total'], actual_input)[0]
                torch.testing.assert_close(actual['ssim'].cpu().double(), expected_ssim.detach(), atol=0.003, rtol=0.01)
                relative_gradient_error = (actual_gradient.cpu().double()-expected_gradient).norm()/expected_gradient.norm()
                self.assertLess(float(relative_gradient_error), 0.03)
                self.assertTrue(torch.backends.cudnn.allow_tf32, 'Loss must restore the caller precision setting')

    def test_masked_ssim_agrees_with_same_valid_crop(self):
        generator = torch.Generator().manual_seed(74)
        prediction = 0.8 + 0.004 * torch.randn(1, 1, 10, 1, 256, 256, generator=generator)
        target = 1.0 + 0.004 * torch.randn(prediction.shape, generator=generator)
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[..., :3, :] = False
        reference = prediction.double().requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            expected_ssim = ssim(reference[..., 3:, :]*1000, target.double()[..., 3:, :]*1000, data_range=255)
        expected = (reference-target.double()).abs()[valid].mean() + 0.1*(1-expected_ssim)
        expected_gradient = torch.autograd.grad(expected, reference)[0]
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=True, benchmark=True):
            actual_input = prediction.cuda().requires_grad_(True)
            actual = compute_loss(actual_input, target.cuda(), valid.cuda(), 1, {'kind':'public_code','ssim_weight':0.1})
            actual_gradient = torch.autograd.grad(actual['total'], actual_input)[0]
            torch.testing.assert_close(actual['ssim'].cpu().double(), expected_ssim.detach(), atol=0.003, rtol=0.01)
            relative_gradient_error = (actual_gradient.cpu().double()-expected_gradient).norm()/expected_gradient.norm()
            self.assertLess(float(relative_gradient_error), 0.03)
            self.assertEqual(int(torch.count_nonzero(actual_gradient.cpu()[~valid])), 0)
            self.assertTrue(torch.backends.cudnn.allow_tf32)


if __name__ == '__main__': unittest.main()
