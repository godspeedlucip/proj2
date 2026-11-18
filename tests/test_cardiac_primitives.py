import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import rstar4d.config as config
from rstar4d.training import losses


class CardiacPrimitivesTests(unittest.TestCase):
    def test_config_resolves_independent_stages_and_factors(self):
        self.assertIn('extensions', config.train_defaults())
        old = config.resolve_stage2(config.validate_train({}), 2)
        self.assertEqual(old['crop_mode'], 'legacy')
        for motion, prior, boundary in itertools.product(('off', 'learned'), ('off', 'geometry'), (False, True)):
            cfg = config.validate_train({
                'stage2_crop_mode': 'core_with_context',
                'extensions': {'motion': {'mode': motion}, 'prior': {'mode': prior}, 'boundary': {'enabled': boundary}},
            })
            stage2 = config.resolve_stage2(cfg, 2)
            self.assertEqual(stage2['recipe']['max_displacement_vox'], [8, 8, 8])
            self.assertEqual(stage2['motion']['struct_weight'], .05 if motion == 'learned' else 0.)
            self.assertEqual(stage2['boundary']['weight'], .5 if boundary else 0.)
            self.assertEqual(config.resolve_stage2(cfg, 1)['prior']['mode'], 'off')

    def test_invalid_parameters(self):
        for ext in ({'motion': {'mode': False}}, {'motion': {'smooth_weight': float('nan')}},
                    {'motion': {'surface_weight': .1}}, {'boundary': {'half_width_vox': 0}},
                    {'boundary': {'enabled': True, 'weight': 0}}):
            with self.subTest(ext=ext), self.assertRaises(ValueError):
                config.validate_train({'extensions': ext})

    def test_angle_layout_and_wrapped_trajectory(self):
        self.assertTrue(hasattr(__import__('rstar4d.data', fromlist=['acquisition']), 'acquisition'))
        from rstar4d.data.acquisition import angle_summary
        angles = np.deg2rad([350, 0, 10, 30, 65, 100])
        labels = [8, 8, 8, 3, 3, 3]
        summary = angle_summary(angles, labels, [3, 8])
        self.assertEqual(summary.shape, (2, 14))
        np.testing.assert_allclose(summary[:, 12:], [[.5, 35/360], [.5, 10/360]])
        self.assertAlmostEqual(float(summary[1, 11]), 1/3)
        a = angle_summary(angles, labels, [3, 8], selected_indices=[0, 2], selected_phase=8)
        b = angle_summary(angles, labels, [3, 8], selected_indices=[0, 1], selected_phase=8)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertAlmostEqual(float(a[1, 12]), 2/6)
        self.assertNotEqual(float(a[1, 13]), float(b[1, 13]))

    def test_geometry_summary_rejected_on_wrong_variant_and_swapped_training(self):
        from rstar4d.data.acquisition import load_geometry
        with tempfile.TemporaryDirectory() as directory:
            summary=Path(directory)/'summary.json'
            content={'encoding':'angles_v1','phase_labels':[2,9],'values':[[0.]*14]*2,
                     'projections_sha256':'shared-projections','paired_sha256':'layout-a','diagnostic_swap':True}
            summary.write_text(json.dumps(content),encoding='utf-8')
            record={'case_id':'x','angle_summary':summary,'projections_sha256':'shared-projections','sha256':'layout-b'}
            with self.assertRaisesRegex(ValueError,'different paired volume'):
                load_geometry(record,[2,9])
            record['sha256']='layout-a'
            with self.assertRaisesRegex(ValueError,'inference diagnostics'):
                load_geometry(record,[2,9],allow_diagnostic_swap=False)
            self.assertEqual(load_geometry(record,[2,9]).shape,(2,14))

    def test_aligned_windows_and_interpolation_support(self):
        self.assertTrue(hasattr(__import__('rstar4d.data', fromlist=['context']), 'context'))
        from rstar4d.data.context import plan_window
        from rstar4d.models.sampling_grid import pull3d
        recipe = {'max_displacement_vox': [1]*3, 'lncc_radius_vox': [1]*3, 'support_margin_vox': [1]*3}
        window = plan_window((12, 25, 31), (4, 11, 13), (3, 5, 7), recipe)
        self.assertEqual(window.origin, (0, 8, 8))
        self.assertEqual(window.stop, (10, 24, 24))
        source = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
        valid = torch.ones_like(source, dtype=torch.bool)
        valid[..., 0, 0, 0] = False
        coords = torch.tensor([[[1., 1., 1.], [.5, .5, .5], [1., 0., 0.]]])
        values, support = pull3d(source, coords, valid)
        self.assertEqual(support.tolist(), [[True, False, True]])
        torch.testing.assert_close(values[0, 0], torch.tensor([7., 0., 4.]))

    def test_missing_real_context_is_distinguished_from_outside_fov(self):
        from dataclasses import replace
        from rstar4d.data.context import plan_window, check_support
        recipe = {'max_displacement_vox':[2]*3,'lncc_radius_vox':[1]*3,'support_margin_vox':[1]*3}
        domain = np.ones((2,12,16,16),dtype=bool)
        correctly_read = plan_window(domain.shape[-3:],(1,5,5),(4,6,6),recipe)
        shortened = replace(correctly_read,origin=(1,*correctly_read.origin[1:]))
        with self.assertRaisesRegex(ValueError,'pig-x.*Z.*1 vox'):
            check_support(shortened,recipe,domain,'pig-x')
        domain[:,0] = False
        check_support(shortened,recipe,domain,'pig-x')

    def test_boundary_averages_structures_and_ramps(self):
        self.assertTrue(hasattr(losses, 'boundary_terms'))
        pred = torch.tensor([[[[[[1., 1., 1., 3.]]]]]], requires_grad=True)
        bands = torch.tensor([[[[[[1, 1, 1, 0]]]], [[[[0, 0, 0, 1]]]]]], dtype=torch.bool)
        result = losses.boundary_terms(pred, torch.zeros_like(pred), bands, {'heart_outer': 1, 'lv': 1}, 1.)
        self.assertEqual(result['total'].item(), 2.)
        self.assertEqual(result['structures']['lv']['pairs'], 0)
        self.assertEqual(result['structures']['heart_outer']['phase_voxels'].tolist(), [[3]])
        self.assertEqual([losses.boundary_scale(ep, 5, .5) for ep in range(1, 7)], [0., .125, .25, .375, .5, .5])

    def test_surface_ignores_internal_heart_and_truncated_interfaces(self):
        self.assertTrue(hasattr(__import__('rstar4d.data', fromlist=['cardiac_annotations']), 'cardiac_annotations'))
        from rstar4d.data.cardiac_annotations import trusted_surface
        labels = np.ones((9, 9, 9), dtype=np.uint8)
        labels[..., 5:] = 0
        labels[3, 3, 2] = 2
        trusted = np.ones_like(labels, dtype=bool)
        trusted[:, :, 0] = False
        heart = trusted_surface(labels > 0, trusted)
        lv = trusted_surface(labels == 2, trusted)
        self.assertFalse(heart[..., :4].any())
        self.assertTrue(heart[4, 4, 4])
        self.assertTrue(lv[3, 3, 2])
