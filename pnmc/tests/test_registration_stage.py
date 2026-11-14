"""The physical registration stage must work from the package alone."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pnmc.config import RegistrationConfig
from pnmc.data.schema import ArrayRef, CaseManifest, GridSpec
from pnmc.data.volume_io import save_volume
from pnmc.motion.registration import register
from pnmc.motion.warp import warp_pull
from pnmc.pipeline import _registration_stage


def _case_and_enhanced(tmp_path):
    grid = GridSpec((8, 8, 8), (1., 1., 1.), (-3.5, -3.5, -3.5),
                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "scanner")
    measurement = tmp_path / "measurement.npy"
    np.save(measurement, np.zeros((2, 1, 1), np.float32))
    projection = ArrayRef(measurement, None, "view,v,u", "line_integral",
                          hashlib.sha256(measurement.read_bytes()).hexdigest())
    geometry = {"nDetector_vu": [1, 1]}
    case = CaseManifest("1", "subject", "case", "acq", ["a", "b"],
                        projection, [0., 1.], ["a", "b"], [0, 1], geometry,
                        grid, grid, {}, None, {}, manifest_sha256="case-sha")
    volumes = []
    for phase in case.phase_ids:
        ref = save_volume(tmp_path / (phase + ".npy"),
                          torch.zeros(grid.shape_zyx), grid)
        volumes.append({"subject_id": case.subject_id, "case_id": case.case_id,
                        "acquisition_id": case.acquisition_id,
                        "phase_id": phase, "grid_sha256": grid.fingerprint(),
                        "prediction_mu": ref.to_dict()})
    enhanced = tmp_path / "enhanced.json"
    enhanced.write_text(json.dumps({"status": "passed", "volumes": volumes}),
                        encoding="utf-8")
    return case, enhanced


def test_registration_stage_writes_all_ordered_pairs(tmp_path, monkeypatch):
    from pnmc.motion import stage

    case, enhanced = _case_and_enhanced(tmp_path)
    calls = []

    def fake_register(moving, fixed, grid, config):
        calls.append((moving.shape, fixed.shape))
        return torch.zeros((3, *grid.shape_zyx)), {
            "direction": "target_fixed_to_source_moving"}

    class Projector:
        def backproject(self, values, indices):
            return torch.ones(case.reconstruction_grid.shape_zyx)

    monkeypatch.setattr(stage, "register", fake_register)
    config = RegistrationConfig(shrink_factors=(1,), iterations=(1,),
                                image_sigma_mm=(0.,), taper_mm=2.)
    report = stage.register_case(case, enhanced, config, tmp_path / "fields",
                                 projector=Projector())
    assert len(calls) == 2
    assert len(report["pairs"]) == 4
    assert {(pair["target"], pair["source"]) for pair in report["pairs"]} == {
        ("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")}
    assert all(pair["unit"] == "mm" and pair["components"] == "xyz"
               and pair["qc"]["status"] == "passed" for pair in report["pairs"])
    assert report["direction"] == "target_fixed_to_source_moving"
    assert report["status"] == "passed"
    assert report["support"]["m0"]["sha256"]
    assert report["support"]["core"]["sha256"]
    assert np.count_nonzero(np.load(report["pairs"][0]["path"])) == 0


def test_pipeline_uses_package_registration_stage(tmp_path, monkeypatch):
    from pnmc.motion import stage

    def fake_register_case(case, enhanced, config, output, *, projector):
        (output / "registration.json").write_text('{"status":"passed"}')

    monkeypatch.setattr(stage, "register_case", fake_register_case)
    result = _registration_stage(None, Path("enhanced.json"),
                                 RegistrationConfig(), tmp_path, object())
    assert result == tmp_path / "registration.json"


def test_actual_demons_improves_textured_translation():
    grid = GridSpec((32, 32, 32), (1., 1., 1.), (-15.5, -15.5, -15.5),
                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "scanner")
    z, y, x = np.mgrid[:32, :32, :32]
    fixed = (np.exp(-((x-13)**2+(y-17)**2+(z-15)**2)/28.) +
             .6*np.exp(-((x-22)**2+(y-10)**2+(z-21)**2)/17.)).astype(np.float32)
    moving = np.zeros_like(fixed)
    moving[:, :, 1:] = fixed[:, :, :-1]
    fixed_t = torch.from_numpy(fixed)
    moving_t = torch.from_numpy(moving)
    config = RegistrationConfig(shrink_factors=(2, 1), iterations=(25, 35),
                                image_sigma_mm=(1., 0.), field_sigma_mm=1.,
                                support_threshold=.01, taper_mm=2.)
    field, details = register(moving_t, fixed_t, grid, config)
    assert details["direction"] == "target_fixed_to_source_moving"
    region = fixed_t > .2
    before = torch.mean((moving_t[region]-fixed_t[region])**2)
    after = torch.mean((warp_pull(moving_t, field, grid, grid)[region]-fixed_t[region])**2)
    assert after < before
    assert field[0, region].median().item() > 0
