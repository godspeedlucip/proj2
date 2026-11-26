import unittest

import torch
from torch.nn import functional as F

from rstar4d.config import validate_train
from rstar4d.models import create_model, forward_volume, initialize_stage2
from rstar4d.models.cycn import CycN3DBlock, CycNBranch, phase_windows
from rstar4d.training.optimization import build_optimizer, learning_rates


def reference_index(position, t, topology):
    if topology == 'cyclic':
        return position % t
    if position < 0:
        return -position
    if position >= t:
        return 2 * (t - 1) - position
    return position


def reference_delta(branch, fdk, topologies):
    state = branch.block.state_dict()
    output = torch.zeros(fdk.shape[0], branch.fuse.out_channels, *fdk.shape[1:])
    for batch, topology in enumerate(topologies):
        t = fdk.shape[1]
        for z in range(fdk.shape[2]):
            for phase in range(t):
                encoded = []
                centers = [phase] if branch.mode == 'center_only' else [
                    reference_index(phase + shift, t, topology) for shift in (-1, 0, 1)]
                for center in centers:
                    indices = [center] * 3 if branch.mode == 'center_only' else [
                        reference_index(center + shift, t, topology) for shift in (-1, 0, 1)]
                    triplet = torch.stack([fdk[batch, index, z] for index in indices])[None, None]
                    feature = F.conv3d(triplet, state['conv_encode1.0.weight'], state['conv_encode1.0.bias'])
                    feature = F.prelu(feature, state['conv_encode1.1.weight']).squeeze(2)
                    feature = F.conv2d(feature, state['conv_encode1_2.0.weight'], state['conv_encode1_2.0.bias'])
                    encoded.append(F.prelu(feature, state['conv_encode1_2.1.weight']))
                if len(encoded) == 1:
                    encoded *= 3
                mixed = F.conv2d(torch.cat(encoded, 1), branch.fuse.weight)
                output[batch, :, phase, z, 4:-4, 4:-4] = mixed[0]
    return output


class CycNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)

    def test_original_block_and_config(self):
        block = CycN3DBlock().eval()
        self.assertEqual(sum(p.numel() for p in block.parameters()), 710)
        with torch.no_grad():
            self.assertEqual(block(torch.rand(1, 1, 3, 512, 512)).shape, (1, 4, 504, 504))
        for cfg in [{'mode':'invalid'}, {'z_chunk':0}, {'z_chunk':1.2},
                    {'backbone_lr_ratio':1.1}, {'backbone_lr_ratio':0.}, {'train_policy':'frozen'}]:
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, 'cycn'):
                validate_train({'cycn': cfg})

    def test_windows_against_per_sample_oracle(self):
        topologies = ['cyclic', 'ordered_states']
        for t in (10, 11):
            cyclic, reflected = (phase_windows(t, mode) for mode in topologies)
            self.assertEqual(cyclic[cyclic[4]].tolist(), [[2, 3, 4], [3, 4, 5], [4, 5, 6]])
            self.assertEqual(cyclic[cyclic[0]].tolist(), [[t-2, t-1, 0], [t-1, 0, 1], [0, 1, 2]])
            self.assertEqual(reflected[reflected[0]].tolist(), [[0, 1, 2], [1, 0, 1], [0, 1, 2]])
            self.assertEqual(reflected[reflected[-1]].tolist(), [[t-3, t-2, t-1], [t-2, t-1, t-2], [t-3, t-2, t-1]])
            x = torch.rand(2, t, 3, 13, 17)
            variants = {}
            for mode in ('five_phase', 'center_only'):
                torch.manual_seed(55)
                cfg = validate_train({'cycn': {'mode': mode, 'z_chunk': 2}})['cycn']
                branch = CycNBranch(cfg, 16).eval()
                torch.nn.init.uniform_(branch.fuse.weight, -.1, .1)
                variants[mode] = branch.state_dict()
                with torch.no_grad():
                    actual = branch.correction(x, topologies)
                    torch.testing.assert_close(actual, reference_delta(branch, x, topologies), rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual[0:1], branch.correction(x[:1], 'cyclic'), rtol=1e-5, atol=1e-6)
                    rolled = branch.correction(x[:1].roll(1, 1), 'cyclic')
                    torch.testing.assert_close(rolled, actual[:1].roll(1, 2), rtol=1e-5, atol=1e-6)
                    changed = x.clone()
                    changed[:, -1] += 2
                    delta = branch.correction(changed, topologies)
                    self.assertTrue(torch.equal(actual[1, :, 0], delta[1, :, 0]))
                    self.assertEqual(torch.equal(actual[0, :, 0], delta[0, :, 0]), mode == 'center_only')
                    self.assertEqual(actual[..., :4, :].count_nonzero(), 0)
                    self.assertEqual(actual[..., -4:].count_nonzero(), 0)
                    branch.z_chunk = 9
                    torch.testing.assert_close(actual, branch.correction(x, topologies), rtol=1e-5, atol=1e-6)
            for key in variants['five_phase']:
                self.assertTrue(torch.equal(variants['five_phase'][key], variants['center_only'][key]))

    def test_stems_zero_init_alignment_and_transfer(self):
        for stage in (1, 2):
            vanilla = create_model(stage).eval()
            model = create_model(stage, cycn={'mode':'five_phase', 'z_chunk':1}).eval()
            model.load_state_dict({**model.state_dict(), **vanilla.state_dict()})
            self.assertEqual(sum(p.numel() for p in model.cycn_branch.parameters()), 902)
            x = torch.rand(1, 2, 10, 1 if stage == 1 else 3, 17, 19)
            with torch.no_grad():
                output = forward_volume(model, x, 'ordered_states')
                self.assertEqual(output.shape, (1, 1, 10, x.shape[3], 17, 19))
                torch.testing.assert_close(output, forward_volume(vanilla, x), rtol=1e-5, atol=1e-6)
                with self.assertRaisesRegex(ValueError, 'case-90.*temporal_topology'):
                    forward_volume(model, x, None, 'case-90')
                torch.nn.init.uniform_(model.cycn_branch.fuse.weight, -.1, .1)
                fdk = x[:, 0].clone()
                x[:, 1] += 2
                self.assertTrue(torch.equal(model.cycn_branch.correction(fdk, 'cyclic'),
                                            model.cycn_branch.correction(x[:, 0], 'cyclic')))
                corrected = forward_volume(model, x, 'ordered_states')
                self.assertGreater(float((corrected - forward_volume(vanilla, x)).abs().max()), 0)
                stem_input = x if stage == 1 else torch.cat((x, x[:, :, :, -1:]), dim=3)
                stem = model.conv1
                xy = stem.conv1(stem_input.flatten(2,3)).unflatten(2, (10, stem_input.shape[3]))
                explicit = xy + reference_delta(model.cycn_branch, stem_input[:, 0], ['ordered_states'])
                if stage == 2:
                    explicit = stem.conv2(explicit.flatten(4)).unflatten(-1, (17,19))
                torch.testing.assert_close(model.cycn_branch.stem(stem, stem_input, 'ordered_states'),
                                           F.leaky_relu(explicit, .2), rtol=1e-5, atol=1e-6)
        stage1 = create_model(1, cycn={'mode':'five_phase'})
        with torch.no_grad():
            stage1.cycn_branch.fuse.weight.fill_(.05)
        initialize_stage2(model, stage1.state_dict())
        for key, value in stage1.cycn_branch.state_dict().items():
            self.assertTrue(torch.equal(model.cycn_branch.state_dict()[key], value))
        image = create_model(2, cycn={'mode':'five_phase','fusion':'image_residual'})
        self.assertEqual(sum(p.numel() for p in image.cycn_branch.parameters()), 722)
        image.load_state_dict({**image.state_dict(), **model.state_dict(),
                               'cycn_branch.fuse.weight':image.cycn_branch.fuse.weight})
        with torch.no_grad():
            image.cycn_branch.fuse.weight.fill_(.03)
            delta = reference_delta(image.cycn_branch, stem_input[:, 0], ['cyclic'])
            augmented = torch.cat((stem_input[:, :1]+delta, stem_input[:, 1:]), dim=1)
            torch.testing.assert_close(image.cycn_branch.stem(image.conv1, stem_input, 'cyclic'),
                                       image.conv1(augmented), rtol=1e-5, atol=1e-6)

    def test_two_updates_and_freezing(self):
        for policy in ('joint', 'frozen'):
            cfg = validate_train({'cycn':{'mode':'five_phase','train_policy':policy}})
            model = create_model(2, cycn=cfg['cycn'])
            optimizer = build_optimizer(model, cfg)
            self.assertEqual(list(learning_rates(optimizer)), ['cycn'] if policy=='frozen' else ['backbone','cycn'])
            groups = [id(p) for group in optimizer.param_groups for p in group['params']]
            self.assertEqual(len(groups), len(set(groups)))
            self.assertEqual(set(groups), {id(p) for p in model.parameters() if p.requires_grad})
            initial = {name:p.detach().clone() for name,p in model.named_parameters()}
            x = torch.rand(1, 2, 10, 2, 16, 16)
            target = torch.rand_like(x[:, :1])
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = (forward_volume(model, x, 'cyclic') - target).abs().mean()
                loss.backward()
                grad = model.cycn_branch.block.conv_encode1[0].weight.grad
                self.assertEqual(bool(grad.abs().max() > 0), update == 1)
                optimizer.step()
                self.assertFalse(torch.equal(initial['cycn_branch.fuse.weight'], model.cycn_branch.fuse.weight))
            for name, parameter in model.named_parameters():
                if policy=='frozen' and not name.startswith('cycn_branch.'):
                    self.assertTrue(torch.equal(initial[name], parameter), name)
            self.assertEqual(torch.equal(initial['conv1.conv1.weight'], model.conv1.conv1.weight), policy=='frozen')


if __name__ == '__main__':
    unittest.main()
