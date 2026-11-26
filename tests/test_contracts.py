from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import h5py
import numpy as np
import torch
from rstar4d.data.contracts import basic_metadata, write_case, write_manifest
from rstar4d.utils.io import atomic_jsonl, sha256

from rstar_decnn.backbone import normalize, physical
from rstar_decnn.data import load_rows, prepare, read_case, sidecar_metadata, validate_transform
from rstar_decnn.evaluate import _candidate_table, pair_group, settings
from rstar_decnn.predict import fuse

from .engineering_smoke import grid


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = basic_metadata(3, [1, 1, 1], geometry=grid((4, 5, 6)),
                                   phase_labels=[11, 2, 9], prior_mode="phase_weighted_fdk")
        self.x = (np.arange(360, dtype=np.float32).reshape(3, 4, 5, 6) - 90) / 10000
        self.prior = self.x.mean(0)
        self.mask = np.ones(self.x.shape, np.uint8)
        self.mask[1, 2, 2, 2] = 0
        self.path = self.root / "case.h5"
        self.base = self.root / "source.jsonl"
        self.prepared = self.root / "prepared.jsonl"

    def write(self, desired=None, grid_override=None):
        write_case(self.path, self.x, self.prior, desired, self.mask[[2, 0, 1]], self.meta)
        with h5py.File(self.path, "r+") as f:
            f.attrs["valid_mask_metadata_json"] = json.dumps({"phase_labels": [9, 11, 2]})
            if desired is not None:
                f.attrs["target_metadata_json"] = json.dumps(
                    {"phase_labels": [9, 11, 2], **({"geometry": grid_override} if grid_override else {})}
                )
        item = {"case_id": "exam", "animal_id": "animal-2", "path": str(self.path), "sha256": sha256(self.path)}
        write_manifest(self.base, [item])
        return item

    def test_label_alignment_crop_and_affine_flip(self):
        self.write(self.x[[2, 0, 1]])
        prepare(self.base, self.prepared, prior_mode="phase_weighted_fdk", shared=True)
        row = load_rows(self.prepared)[0][0]
        sample = read_case(row)
        np.testing.assert_array_equal(sample["x"], sample["y"])
        np.testing.assert_array_equal(sample["valid"], self.mask)
        self.assertEqual(sample["phase_labels"], [11, 2, 9])
        crop = read_case(row, (slice(1, 4), slice(1, 5), slice(2, 5)), flips=(1,))
        coords = np.array([1, 2, 0, 1])
        mapped = np.asarray(crop["geometry"]["voxel_to_scanner_zyx"]) @ coords
        original = np.asarray(self.meta["geometry"]["voxel_to_scanner_zyx"]) @ [2, 2, 2, 1]
        np.testing.assert_allclose(mapped, original)
        for key in ("x", "y", "valid"):
            self.assertEqual(crop[key][1, 1, 2, 0], sample[key][1, 2, 2, 2])
        for key in ("a_s", "a_d"):
            self.assertEqual(crop[key][1, 2, 0], sample[key][2, 2, 2])

    def test_two_reversible_transforms_with_negative_values(self):
        transform = {"scale": 0.1, "offset": 0.03, "unit": "mm^-1", "source": "fixture",
                     "physical_range": [-0.02, 0.1]}
        for contract in (self.meta["normalization"], validate_transform(transform)):
            np.testing.assert_allclose(physical(normalize(self.x, contract), contract), self.x, atol=1e-8)
        self.assertLess(normalize(self.x, self.meta["normalization"]).min(), 0)
        self.assertGreaterEqual(normalize(self.x, transform).min(), 0)
        with self.assertRaisesRegex(ValueError, "ReLU"):
            validate_transform(dict(transform, offset=0))

    def test_reject_shifted_target_grid(self):
        shifted = deepcopy(self.meta["geometry"])
        shifted["voxel_to_scanner_zyx"][0][3] += 0.25
        shifted["center_zyx_mm"][0] += 0.25
        self.write(self.x[[2, 0, 1]], shifted)
        with self.assertRaisesRegex(ValueError, r"exam: physical grid mismatch"):
            prepare(self.base, self.prepared, prior_mode="phase_weighted_fdk", shared=True)

    def test_reject_missing_projection_identity(self):
        self.write()
        with self.assertRaisesRegex(ValueError, r"exam: missing same-scan"):
            prepare(self.base, self.prepared, prior_mode="phase_weighted_fdk")

    def test_no_reference_sidecar_reuse_and_mismatch(self):
        item = self.write()
        prior = self.root / "all.h5"
        item["metadata"] = self.meta
        meta = sidecar_metadata(item, {"mixed_projections_sha256": "bytestream-sha"})
        index = self.root / "prior-index.jsonl"
        for wrong in (False, True):
            with h5py.File(prior, "w") as f:
                f.attrs["metadata_json"] = json.dumps(dict(meta, case_id="wrong") if wrong else meta)
                f["prior"] = self.prior + 1
            atomic_jsonl(index, [{"case_id": "exam", "path": prior.name, "sha256": sha256(prior)}])
            if wrong:
                with self.assertRaisesRegex(ValueError, r"exam: DeCNN prior identity"):
                    prepare(self.base, self.prepared, prior_mode="phase_weighted_fdk", prior_index=index)
            else:
                prepare(self.base, self.prepared, prior_mode="phase_weighted_fdk", prior_index=index)
                prepared = load_rows(self.prepared)[0][0]
                sample = read_case(prepared, supervision=False)
                self.assertIsNone(sample["y"])
                np.testing.assert_array_equal(sample["a_d"], self.prior + 1)


class CalibrationContracts(unittest.TestCase):
    def test_additional_extractor_failure_cannot_improve_calibration(self):
        def case(key, boundary, motion):
            return {"case_id": key, "animal_id": key, "boundary_mae": boundary, "motion_error_ml": motion,
                    "lv": {"phases": [], "boundary_mae": boundary, "motion": {
                        "status": "ok" if motion is not None else "prediction_unavailable",
                        "absolute_error_ml": motion}}}

        reference = {"2026": {"A": case("A", 0.5, 0.2), "B": case("B", 0.6, 0.7)}}
        candidate = {"2026": {"A": case("A", 0.2, 0.1), "B": case("B", 0.2, None)}}
        table = _candidate_table({0: reference, 1: candidate}, [0, 1], settings())
        self.assertTrue(table[0]["eligible"])
        self.assertFalse(table[1]["eligible"])
        self.assertEqual(table[1]["extra_missing_motion"][0]["case_id"], "B")
        paired, missing = pair_group(candidate, reference, "motion_error_ml")
        self.assertEqual(len(paired), 1)
        self.assertEqual(len(missing), 1)

    def test_alpha_endpoint_preserves_training_gradient(self):
        a = torch.tensor(0.0, requires_grad=True)
        s, d = torch.tensor([2.]), torch.tensor([5.], requires_grad=True)
        combined = fuse(s, d, a)
        (combined.square() + (d - 1).square()).sum().backward()
        self.assertEqual(a.grad.item(), 12)
        self.assertEqual(d.grad.item(), 8)
        self.assertIs(fuse(s, None, 0, inference=True), s)
        self.assertIs(fuse(None, d, 1, inference=True), d)


if __name__ == "__main__":
    unittest.main()
