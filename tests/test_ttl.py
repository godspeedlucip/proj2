import unittest

import torch
from torch.nn import functional as F

from rstar4d.models.ttl import TemporalTransformerSkip, temporal_visibility


def slice_reference(ttl, x):
    """Reference a single [1,64,T,H,W] slice using explicit block coordinates."""
    embedded = ttl.embed(x)
    _, channels, phases, h, w = embedded.shape
    grid = torch.empty_like(embedded)
    for y in range(h):
        for col in range(w):
            tokens = embedded[0, :, :, y, col].T
            for block in ttl.blocks:
                q, k, v = block.qkv(block.norm1(tokens)).chunk(3, dim=-1)
                heads = []
                for head in range(16):
                    section = slice(16 * head, 16 * (head + 1))
                    scores = q[:, section] @ k[:, section].T / 4 + block.head_bias[head]
                    heads.append(scores.softmax(dim=-1) @ v[:, section])
                tokens = tokens + block.proj(torch.cat(heads, dim=-1))
                tokens = tokens + block.mlp(block.norm2(tokens))
            grid[0, :, :, y, col] = tokens.T
    restored = ttl.unembed(F.interpolate(grid, scale_factor=(1, 2, 2), mode='nearest'))
    return x + F.leaky_relu(ttl.temporal(restored), 0.2)


class TTLStage2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_stage2_skip_routing(self):
        from rstar4d.models import create_model, forward_volume

        model = create_model(2, stam={'mode': 'temporal_only'}, ttl={'mode': 'full'}).eval()
        torch.nn.init.normal_(model.ttl.temporal.weight, std=0.01)
        features = {}

        def capture(name, pre=False):
            def hook(_module, args, output=None):
                features[name] = (args[0] if pre else output).detach().clone()
            return hook

        model.conv3_3.register_forward_hook(capture('encoder_residual'))
        model.conv3.register_forward_hook(capture('encoder_input'))
        model.conv4.register_forward_pre_hook(capture('deep_input', pre=True))
        model.conv5.register_forward_hook(capture('upsampled'))
        model.conv5_2.register_forward_pre_hook(capture('fusion', pre=True))
        with torch.no_grad():
            output = forward_volume(model, torch.randn(1, 2, 11, 5, 17, 25))
            self.assertEqual(list(output.shape), [1, 1, 11, 5, 17, 25])
            encoder = features['encoder_input'] + features['encoder_residual']
            torch.testing.assert_close(features['deep_input'], encoder)
            skip = model.ttl(encoder)
            self.assertFalse(torch.equal(skip, encoder))
            torch.testing.assert_close(features['fusion'][:, :64], features['upsampled'])
            torch.testing.assert_close(features['fusion'][:, 64:], skip)
            self.assertEqual([n for n, m in model.named_modules() if isinstance(m, TemporalTransformerSkip)], ['ttl'])

    def test_off_legacy_state_dict(self):
        from rstar4d.config import validate_train
        from rstar4d.models import create_model, forward_volume, model_from_checkpoint
        from rstar4d.models.stam import architecture

        structure = {'mode': 'temporal_only'}
        original = create_model(2, stam=structure).eval()
        torch.nn.init.normal_(original.stam.output_projection.weight, std=0.01)
        value = {'stage': 2, 'state_dict': original.state_dict(),
                 'stam': {'architecture': architecture(structure['mode'])}}
        a = model_from_checkpoint(value).eval()
        self.assertEqual(set(a.state_dict()), set(create_model(2, stam=structure, ttl={'mode': 'off'}).state_dict()))
        self.assertFalse(any(name.startswith('ttl.') for name in a.state_dict()))
        from rstar4d.models import attach_ttl
        b = model_from_checkpoint(value)
        attach_ttl(b, {'mode': 'full'})
        b.eval()
        with torch.no_grad():
            x = torch.randn(1, 2, 11, 5, 17, 25)
            self.assertTrue(torch.equal(forward_volume(original, x), forward_volume(a, x)))
            self.assertTrue(torch.equal(forward_volume(a, x), forward_volume(b, x)))
        self.assertEqual(validate_train({})['ttl']['mode'], 'off')
        self.assertEqual(validate_train({'ttl': {'mode': 'full'}}, 2)['stam']['mode'], 'from_parent')
        with self.assertRaisesRegex(ValueError, 'Stage II'):
            validate_train({'ttl': {'mode': 'full'}}, 1)
        with self.assertRaisesRegex(ValueError, 'STAM'):
            create_model(2, ttl={'mode': 'full'})


class TTLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(16)

    def test_nonzero_slice_reference(self):
        ttl = TemporalTransformerSkip('full', z_chunk_size=2).eval()
        self.assertEqual(sum(p.numel() for p in ttl.parameters()), 1_887_136)
        for phases in (10, 11):
            with self.subTest(phases=phases), torch.no_grad():
                x = torch.randn(2, 64, phases, 3, 4, 6)
                torch.nn.init.zeros_(ttl.temporal.weight)
                torch.nn.init.zeros_(ttl.temporal.bias)
                self.assertTrue(torch.equal(ttl(x), x))
                torch.nn.init.normal_(ttl.temporal.weight, std=0.01)
                torch.nn.init.normal_(ttl.temporal.bias, std=0.01)
                expected = torch.empty_like(x)
                for case in range(x.shape[0]):
                    for z in range(x.shape[3]):
                        expected[case:case+1, :, :, z] = slice_reference(ttl, x[case:case+1, :, :, z])
                prediction = ttl(x)
                torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-5)
                perturbed = x.clone()
                perturbed[1, :, :, 1] += 2
                altered = (ttl(perturbed) != prediction).any(dim=(1, 2, 4, 5))
                self.assertEqual(altered.nonzero().tolist(), [[1, 1]])

    def test_temporal_visibility(self):
        for mode, topology, expected in (
            ('full', None, [0, 1, 2, 3, 4]),
            ('local3', 'ordered_states', [0, 1]),
            ('local3', 'cyclic', [0, 1, 4]),
        ):
            with self.subTest(mode=mode, topology=topology):
                visible = temporal_visibility(5, mode, topology)
                self.assertEqual(visible[0].nonzero().flatten().tolist(), expected)
        for topology in ('ordered_states', 'cyclic'):
            self.assertTrue(temporal_visibility(2, 'local3', topology).all())
        ttl = TemporalTransformerSkip('local3')
        x = torch.randn(2, 64, 5, 2, 4, 4)
        for invalid in (None, 'unknown', ['ordered_states']):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'topolog'):
                ttl(x, invalid)
        with torch.no_grad():
            ttl.eval()
            torch.nn.init.normal_(ttl.temporal.weight, std=0.01)
            topologies = ['ordered_states', 'cyclic']
            result = ttl(x, topologies)
            for case, topology in enumerate(topologies):
                torch.testing.assert_close(result[case:case+1], ttl(x[case:case+1], topology))

    def test_droppath_uses_axial_samples(self):
        ttl = TemporalTransformerSkip('full', z_chunk_size=2).train()
        x = torch.randn(2, 64, 5, 3, 4, 4)
        masks = [
            torch.tensor([0, 1, 1, 0, 1, 1.]).reshape(6, 1, 1, 1) / 0.9,
            torch.tensor([1, 1, 0, 1, 0, 1.]).reshape(6, 1, 1, 1) / 0.9,
        ]
        block = ttl.blocks[0]
        tokens = torch.randn(6, 4, 5, 256)
        with torch.no_grad():
            first = tokens + block.attention(block.norm1(tokens)) * masks[0]
            expected = first + block.mlp(block.norm2(first)) * masks[1]
            torch.testing.assert_close(block(tokens, None, masks), expected)
            torch.nn.init.normal_(ttl.temporal.weight, std=0.01)
            torch.manual_seed(29)
            chunked = ttl(x)
            ttl.z_chunk_size = 0
            torch.manual_seed(29)
            torch.testing.assert_close(ttl(x), chunked, atol=1e-6, rtol=1e-5)
