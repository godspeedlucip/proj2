from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from rstar4d.cli import dispatch, parser
from rstar4d.config import load_config
from rstar4d.data.contracts import basic_metadata, write_case, write_manifest
from rstar4d.moco.data import case_rows
from rstar4d.moco.evaluate import edge_measure
from rstar4d.moco.fusion import Accumulator, contribution, field_quality, ordered_pairs, reinject
from rstar4d.moco.prepare import calibrate, pigae
from rstar4d.moco.registration import RegisNet3D, estimate_field
from rstar4d.moco.settings import dependencies
from rstar4d.moco.train import load_registration
from rstar4d.training.checkpoint import atomic_save
from rstar4d.utils.io import sha256


class MoCoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_xyz_pull_on_asymmetric_odd_volume(self):
        image = torch.zeros((7, 9, 13))
        image[4, 3, 7] = 16
        field = torch.zeros((1, 3, *image.shape))
        field[:, 0] = 2
        field[:, 1] = -1
        field[:, 2] = 1
        moved, weights = contribution(image, torch.ones_like(image, dtype=torch.bool), field)
        self.assertEqual(tuple(torch.nonzero(moved > 15)[0]), (3, 4, 5))
        self.assertEqual(weights[0, 0, 0].item(), 0)
        qa = field_quality(field[0].numpy(), np.ones(image.shape, bool), [2., 1.5, .5])
        self.assertAlmostEqual(qa["mean_displacement_mm"], np.sqrt(7.25), places=5)
        self.assertEqual(qa["folded_fraction"], 0)

    def test_fractional_geometry_support_self_and_oob(self):
        target = torch.full((3, 3, 4), 10.)
        source = torch.full_like(target, 16.)
        all_valid = torch.ones_like(target, dtype=torch.bool)
        support = all_valid.clone()
        support[:, :, 2:] = False
        field = torch.zeros((1, 3, *target.shape))
        field[:, 0] = .5
        acc = Accumulator(target, all_valid)
        acc.add(source, support, field, adjacent=True)
        out = acc.result()
        voxel = (1, 1, 1)
        self.assertAlmostEqual(out["d"][voxel].item(), 12., places=5)
        self.assertEqual(out["weight"][voxel].item(), 1.5)
        self.assertEqual(out["contribution_count"][voxel].item(), 2)
        self.assertEqual(out["raw_sum"][voxel].item(), 18.)
        self.assertTrue((out["d"][:, :, -1] == 10).all())
        values, weight = contribution(source, all_valid, field)
        self.assertTrue((values[:, :, -1] == 0).all() and (weight[:, :, -1] == 0).all())
        np.testing.assert_allclose(values[weight > 0], 16 * weight[weight > 0], atol=1e-6)
        empty = Accumulator(target, ~all_valid)
        empty.add(source, support, field, adjacent=True)
        self.assertFalse(empty.result()["valid_mask"].any())

    def test_pairs_and_gap(self):
        for count in (10, 11):
            expected = list(range(101, 101 + 2 * count, 2))
            for mode, amount in [("adjacent", 2 * (count - 1)), ("all", count * (count - 1))]:
                pairs = ordered_pairs(expected, expected, mode)
                self.assertEqual(sum(len(values) for values in pairs.values()), amount)
                self.assertTrue(all(k not in v for k, v in pairs.items()))
        self.assertEqual(ordered_pairs([40, 3, 17, 1], [40, 3, 2, 17, 1], "adjacent"),
                         {40: [3], 3: [40], 17: [1], 1: [17]})

    def test_reinjection_zero_and_distinct_nonzero(self):
        baseline = np.full((7, 9, 11), .02, dtype=np.float32)
        single = baseline.copy()
        single[3, 4, 5] = .035
        multi = baseline.copy()
        multi[3, 5, 6] = .035
        support = np.zeros(baseline.shape, bool)
        support[1:-1, 2:-2, 1:-1] = True
        params = (1., [.7, 1., 1.5])
        for array in (single, multi):
            np.testing.assert_array_equal(reinject(baseline, array, support, 0., *params), baseline)
        a, b = [reinject(baseline, image, support, .5, *params) for image in (single, multi)]
        self.assertFalse(np.array_equal(a, b))
        np.testing.assert_array_equal(a[~support], baseline[~support])
        np.testing.assert_array_equal(reinject(baseline, baseline + .01, support, 1., *params), baseline)

    def test_static_pigae_preserves_calibrated_prior(self):
        prior = np.arange(27, dtype=np.float32).reshape(3, 3, 3) * .001
        sparse = prior * .75
        fit = [3., .9, .003]
        consistent = pigae(sparse, prior, sparse, "static_consistent", fit)
        np.testing.assert_array_equal(consistent, calibrate(prior, fit))
        self.assertEqual(consistent[0, 0, 0], 0.)
        self.assertFalse(np.array_equal(pigae(sparse, prior, sparse, "printed", fit), consistent))

    def test_context_aligned_model_blocks(self):
        torch.manual_seed(24)
        model = RegisNet3D().eval()
        moving, fixed = torch.rand((2, 1, 1, 145, 17, 19))
        with torch.inference_mode():
            full = estimate_field(model, moving, fixed, 4.)
            tiled = estimate_field(model, moving, fixed, 4., (61, 17, 19), 64)
        torch.testing.assert_close(tiled, full, atol=1e-7, rtol=1e-5)
        self.assertGreater(full.abs().max().item(), 0.)

    def test_wall_profile_thresholds(self):
        x = np.arange(-5., 5.25, .25)
        target = np.clip((x + 1.) / 2, 0, 1) * .02
        prediction = np.clip((x - .5 + 2) / 4, 0, 1) * .025
        loc, width = edge_measure(target, x, 1., .001)
        self.assertAlmostEqual(loc, 0.)
        self.assertAlmostEqual(width, 1.6)
        loc, width = edge_measure(prediction, x, 1., .001)
        self.assertAlmostEqual(loc, .5)
        self.assertAlmostEqual(width, 3.2)
        with self.assertRaisesRegex(ValueError, "low contrast"):
            edge_measure(np.ones_like(x), x, 1., .001)

    def test_contrast_only_phantom_uses_fixed_geometry(self):
        grid = np.indices((11, 13, 15))
        sphere = sum((axis - center) ** 2 for axis, center in zip(grid, (5, 6, 7))) < 16
        source = np.where(sphere, .032, .012).astype(np.float32)
        target = np.where(sphere, .023, .012).astype(np.float32)
        support = np.ones_like(sphere)
        acc = Accumulator(torch.from_numpy(target), torch.from_numpy(support))
        acc.add(torch.from_numpy(source), torch.from_numpy(support), None, adjacent=True)
        combined = acc.result()["d"].numpy()
        self.assertAlmostEqual(float((combined[sphere] - target[sphere]).mean()), .0045)
        field = np.zeros((3, *sphere.shape), dtype=np.float32)
        self.assertEqual(field_quality(field, support, [1., 1., 1.])["mean_displacement_mm"], 0.)
        x = np.arange(-5., 5.25, .25)
        contour = np.clip((x + .5) / 1, 0, 1)
        self.assertEqual(edge_measure(.012 + .011 * contour, x, 1., .001),
                         edge_measure(.012 + .02 * contour, x, 1., .001))

    def test_config_contracts(self):
        config = load_config(kind="moco")
        self.assertEqual(dependencies(config), ({"b"}, False, False))
        config["fusion"]["source"] = "c"
        config["outputs"] = ["d"]
        with self.assertRaisesRegex(ValueError, "pigae.mode=off"):
            dependencies(config)
        config["pigae"]["mode"] = "static_consistent"
        config["registration"]["mode"] = "zero"
        self.assertEqual(dependencies(config), ({"c"}, True, False))

    def test_cli_zero_field_missing_state_without_projections_or_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shape = (4, 3, 4, 5)
            image = np.stack([np.full(shape[1:], k, dtype=np.float32) for k in (1, 2, 7, 8)])
            affine = np.eye(4)
            meta = basic_metadata(4, [1., 1., 1.], phase_labels=[40, 5, 11, 90],
                                  geometry={"voxel_to_scanner_zyx": affine.tolist(),
                                            "voxel_spacing_zyx_mm": [1., 1., 1.],
                                            "shape_zyx": list(shape[1:]), "center_zyx_mm": [1, 1.5, 2]})
            paired = root / "paired.h5"
            write_case(paired, image, image[0], None, np.ones(shape, bool), meta)
            manifest = root / "manifest.jsonl"
            record = {"case_id": "case", "animal_id": "animal", "path": str(paired),
                      "sha256": sha256(paired), "projections": root.joinpath("not_required.h5").as_posix(),
                      "expected_phase_labels": [40, 5, 2, 11, 90]}
            write_manifest(manifest, [record])
            args = ["infer-moco", "--config", "configs/moco/a_zero.yaml", "--manifest", str(manifest),
                    "--out", str(root / "out"), "--register-checkpoint", str(root / "not_required.pt")]
            with patch.object(RegisNet3D, "forward", side_effect=AssertionError("unexpected RegisNet call")):
                result = dispatch(parser().parse_args(args))
                resumed = dispatch(parser().parse_args(args))
            self.assertEqual((result["forward_calls"], result["fields_total"]), (0, 4))
            self.assertEqual(resumed["resumed_targets"], 4)
            with h5py.File(root / "out/d/case/prediction.h5", "r") as out:
                np.testing.assert_array_equal(out["prediction"][:, 1, 1, 1], [1.5, 1.5, 7.5, 7.5])
            record["expected_phase_labels"] = [5, 40, 2, 11, 90]
            write_manifest(manifest, [record])
            with self.assertRaisesRegex(ValueError, "expected_phase_labels conflicts"):
                case_rows(manifest)

    def test_mismatched_checkpoint_inputs_and_pair_range(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrong.pt"
            config = load_config(kind="moco")
            row = {"metadata": basic_metadata(2, [1., 1., 1.])}
            from rstar4d.moco.train import contract
            payload = contract(config, row, None)
            payload.update(input="x", train_pairs="all")
            atomic_save(path, {"schema_version": 1, "checkpoint_kind": "registration_weights",
                               "stage": "registration", "state_dict": {}, "provider_contract": payload})
            with self.assertRaisesRegex(ValueError, "input, train_pairs"):
                load_registration(path, config, [row], None)


if __name__ == "__main__":
    unittest.main()
