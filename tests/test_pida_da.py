import unittest

import torch

from rstar4d.config import train_defaults, validate_train
from rstar4d.models import attach_pida, create_model, forward_volume
from rstar4d.models.pida_da import PIDADeformableAttention, is_pida_parameter, match_planes, modulated_sample
from rstar4d.training.trainer import make_optimizer


class PIDAOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)

    def test_cosine_center_and_global_raw_softmax(self):
        query = torch.tensor([[[[1., 0.]], [[0., 0.]]]])
        key = torch.tensor([[[[1., 2.]], [[0., 2.]]]])
        index, weight = match_planes(query, key, query_chunk_size=1)
        self.assertEqual(index.tolist(), [[0, 0]])
        torch.testing.assert_close(weight, torch.tensor([[0.26894142, 0.5]]))

        keys = torch.zeros(1, 2, 2, 4)
        keys[:, 0] = 1
        keys[0, :, -1, -1] = torch.tensor([0., 1.])
        queries = keys.flip((2, 3))
        index, weight = match_planes(queries, keys, query_chunk_size=3)
        self.assertEqual(index[0, :2].tolist(), [7, 0])
        self.assertAlmostEqual(float(weight[0, 0]), 1 / (1 + 7 / torch.e), places=6)
        zeros = torch.zeros_like(keys)
        index, weight = match_planes(queries, zeros)
        self.assertEqual(index.count_nonzero(), 0)
        torch.testing.assert_close(weight, torch.full_like(weight, 1/8))

    def test_deformable_sample_coordinates_and_zero_extension(self):
        y, x = torch.meshgrid(torch.arange(3), torch.arange(4), indexing='ij')
        value = (10*y + x).float().reshape(1, 1, 3, 4)
        centers = torch.tensor([[6, 6, 6, 6, 7, 7] + [0]*6])
        offsets = torch.zeros(1, 18, 3, 4)
        offsets[:, 8, 0] = torch.tensor([0., 1., 0., .5])
        offsets[:, 9, 0] = torch.tensor([0., 0., 1., -.5])
        offsets[:, 9, 1, :2] = torch.tensor([1., .5])
        masks = torch.full((1, 9, 3, 4), .8)
        kernel = torch.zeros(1, 1, 3, 3)
        kernel[0, 0, 1, 1] = 1
        actual = modulated_sample(value, centers, offsets, masks, kernel).flatten()
        torch.testing.assert_close(actual[:6], .8 * torch.tensor([12., 22., 13., 16.5, 0., 6.5]))

        kernel.zero_()
        kernel[0, 0, 0, 2] = 1
        actual = modulated_sample(value, centers, offsets, masks, kernel).flatten()
        torch.testing.assert_close(actual[0], torch.tensor(2.4))

    def test_chunking_preserves_values_gradients_and_independent_planes(self):
        da = PIDADeformableAttention(3, query_chunk_size=0, plane_batch_size=0)
        inputs = [(torch.randn(2, 3, 2, 2, 3, 4) * .25).requires_grad_() for _ in range(4)]
        parameters = (*inputs, *da.parameters())
        dense = da.forward_volume(*inputs)
        grads = torch.autograd.grad(dense.square().sum(), parameters)
        da.query_chunk_size = 5
        da.plane_batch_size = 3
        blocked = da.forward_volume(*inputs)
        grads_blocked = torch.autograd.grad(blocked.square().sum(), parameters)
        torch.testing.assert_close(blocked, dense, atol=1e-6, rtol=1e-5)
        for a, b in zip(grads, grads_blocked):
            self.assertTrue(torch.isfinite(a).all() and a.count_nonzero() > 0)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        da.eval()
        changed = [array.detach().clone() for array in inputs]
        changed[2][0, :, 1, 0] += .7
        with torch.no_grad():
            updated = da.forward_volume(*changed)
        affected = (updated - dense).abs().amax(dim=(1, 4, 5)) > 1e-7
        self.assertEqual(affected.nonzero().tolist(), [[0, 1, 0]])

    def test_extreme_scores_and_autocast(self):
        q = torch.full((1, 2, 3, 5), 1e15, requires_grad=True)
        k = torch.cat((q[:, :, :, :2], -q[:, :, :, 2:]), dim=3)
        da = PIDADeformableAttention(2, query_chunk_size=4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            index, weight = match_planes(q, k, query_chunk_size=4)
            result = da(q, k, torch.randn_like(q), torch.randn_like(q))
        self.assertEqual(index.count_nonzero(), 0)
        torch.testing.assert_close(weight, torch.full_like(weight, 1/6))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.dtype, torch.bfloat16)


class PIDAHostTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)

    def test_config_rejects_wrong_host(self):
        da = {'enabled': True}
        for stage, mode in ((1, 'joint'), (2, 'off'), (2, 'temporal_only')):
            with self.subTest(stage=stage, mode=mode):
                with self.assertRaisesRegex(ValueError, 'Stage II'):
                    create_model(stage, stam={'mode': mode}, pida_da=da)
        with self.assertRaisesRegex(ValueError, 'plane_batch_size'):
            validate_train({'pida_da': {'plane_batch_size': -1}}, 2)

    def test_prior_is_independent_of_phase_input_and_uses_shared_encoder(self):
        model = create_model(2, 'pig_ordered_v1', {'mode': 'joint'}, {'enabled': True})
        x = torch.randn(1, 2, 10, 4, 16, 16)
        changed = x.clone()
        changed[:, 0] += 1
        with torch.no_grad():
            a = model._prior_features(x)
            b = model._prior_features(changed)
            changed[:, 1] += 1
            c = model._prior_features(changed)
        for ka, kb, kc in zip(a, b, c):
            self.assertTrue(torch.equal(ka, kb))
            self.assertFalse(torch.equal(ka, kc))
        seen = []
        hook = model.conv3.register_forward_hook(lambda module, *_: seen.append(id(module)))
        try:
            forward_volume(model.eval(), x)
        finally:
            hook.remove()
        self.assertEqual(seen, [id(model.conv3)] * 2)
        self.assertEqual(sum(isinstance(module, PIDADeformableAttention) for module in model.modules()), 1)

    def test_zero_step_and_two_updates_in_both_frozen_modes(self):
        cfg = train_defaults()
        cfg.update(weight_decay=.01, learning_rate=.001)
        cfg['stam']['mode'] = 'joint'
        x = torch.rand(1, 2, 10, 3, 9, 11)
        a = create_model(2, 'pig_ordered_v1', cfg['stam']).eval()
        with torch.no_grad():
            a.stam.output_projection.weight.normal_(std=.01)
            baseline = forward_volume(a, x)
        for policy in ('stam', 'pida_only'):
            with self.subTest(policy=policy):
                b = create_model(2, 'pig_ordered_v1', cfg['stam']).eval()
                b.load_state_dict(a.state_dict())
                attach_pida(b, {'enabled': True, 'train_policy': policy})
                with torch.no_grad():
                    torch.testing.assert_close(forward_volume(b, x), baseline, atol=1e-6, rtol=1e-5)
                optimizer = make_optimizer(b.train(), cfg, backbone_frozen=True)
                active = [p for group in optimizer.param_groups for p in group['params']]
                self.assertEqual(len(active), len({id(p) for p in active}))
                self.assertEqual({id(p) for p in active}, {id(p) for p in b.parameters() if p.requires_grad})
                if policy == 'pida_only':
                    self.assertEqual({id(p) for p in active}, {id(p) for p in b.pida_da.parameters()})
                    self.assertTrue(b.pida_da.training)
                    self.assertTrue(all(not m.training for m in b.children() if m is not b.pida_da))
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    forward_volume(b, x).square().mean().backward()
                    if step == 0 and policy == 'stam':
                        for name, parameter in b.named_parameters():
                            if name.endswith('.pida_weight'):
                                self.assertGreater(parameter.grad.abs().max(), 0.)
                    if step == 1 or policy == 'pida_only':
                        self.assertGreater(b.pida_da.kernel.grad.abs().max(), 0.)
                    if step == 1:
                        for gradient in (b.pida_da.offset.weight.grad, b.pida_da.mask.weight.grad):
                            self.assertTrue(torch.isfinite(gradient).all() and gradient.count_nonzero() > 0)
                    optimizer.step()
                for name, old in a.state_dict().items():
                    if not is_pida_parameter(name) and (policy == 'pida_only'
                            or not name.startswith('stam.') or name.startswith('stam.sam.')):
                        self.assertTrue(torch.equal(b.state_dict()[name], old), name)
