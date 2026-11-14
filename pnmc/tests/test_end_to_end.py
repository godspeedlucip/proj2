"""Stage signatures, incomplete writes, and inference input isolation."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

import pytest

from pnmc.pipeline import StageStore, _load_fields, digest_json, fields_for_target
from pnmc.config import NetworkConfig, ReconstructionConfig
from pnmc.data.schema import ArrayRef, CaseManifest, GridSpec
from pnmc.data.volume_io import save_volume
from pnmc.pipeline import run_case


def test_published_reconstruction_config_is_valid():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "reconstruction.json"
    config = ReconstructionConfig.from_json(config_path)
    assert config.sweeps == 20
    assert config.fdk_padding == 100


def test_failed_registration_qc_blocks_mc(tmp_path):
    grid = GridSpec((4, 4, 4), (1., 1., 1.), (0., 0., 0.),
                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "scanner")
    case = type("Case", (), {"phase_ids": ["a"],
                             "reconstruction_grid": grid})()

    def artifact(name, values):
        path = tmp_path / (name + ".npy")
        np.save(path, values)
        return {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    support = {name: artifact(name, np.ones(grid.shape_zyx, np.bool_))
               for name in ("m0", "core")}
    field = artifact("field", np.zeros((3, *grid.shape_zyx), np.float32))
    report = {"status": "passed",
              "direction": "target_fixed_to_source_moving",
              "grid_sha256": grid.fingerprint(), "support": support,
              "pairs": [{"target": "a", "source": "a", **field,
                         "grid_sha256": grid.fingerprint(), "unit": "mm",
                         "components": "xyz", "qc": {"status": "failed"}}]}
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="QC invalid"):
        _load_fields(path, case)


def test_stage_store_reuses_only_matching_complete_artifacts(tmp_path):
    store = StageStore(tmp_path / "run", resume=True)
    calls = []

    def build(folder):
        calls.append(1)
        path = folder / "result.txt"
        path.write_text("one", encoding="utf-8")
        return {"result": path}

    first = store.run("enhanced", "input-a", build)
    assert first["reused"] is False
    second = store.run("enhanced", "input-a", build)
    assert second["reused"] is True and len(calls) == 1
    Path(first["artifacts"]["result"]["path"]).write_text("damaged", encoding="utf-8")
    third = store.run("enhanced", "input-a", build)
    assert third["reused"] is False and len(calls) == 2
    changed = store.run("enhanced", "input-b", build)
    assert changed["reused"] is False and len(calls) == 3
    assert changed["folder"] != first["folder"]


def test_incomplete_stage_does_not_resume_as_success(tmp_path):
    store = StageStore(tmp_path / "run", resume=True)

    def fail(folder):
        (folder / "half.txt").write_text("partial")
        raise ValueError("stopped")

    with pytest.raises(ValueError, match="stopped"):
        store.run("registered", "same", fail)
    rebuilt = store.run("registered", "same", lambda folder: {
        "result": _write(folder / "result.txt", "complete")})
    assert rebuilt["reused"] is False
    assert not (Path(rebuilt["folder"]) / "half.txt").exists()


def _write(path, value):
    path.write_text(value, encoding="utf-8")
    return path


def test_digest_json_changes_with_each_dependency():
    base = {"projection": "a", "geometry": "b", "normalization": "c",
            "checkpoint": "d", "field": "e", "support": "f"}
    original = digest_json(base)
    for key in base:
        modified = dict(base, **{key: "changed"})
        assert digest_json(modified) != original


def test_mc_target_uses_fixed_observation_to_moving_target_field():
    pairs = {("a", "a"): "aa", ("a", "b"): "a_to_b",
             ("b", "a"): "b_to_a", ("b", "b"): "bb"}
    # Reconstructing a uses the field sampled on observation b and pointing
    # to source a; it must not use the reverse registration.
    assert fields_for_target(pairs, "a", ["a", "b"]) == {
        "a": "aa", "b": "b_to_a"}


def test_cpu_case_flows_from_physical_manifest_to_mc_without_reference(tmp_path):
    grid = GridSpec((24, 24, 24), (1., 1., 1.), (-11.5, -11.5, -11.5),
                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "scanner")
    geometry = {"DSO_mm": 500., "DSD_mm": 900.,
                "nDetector_vu": [1, 1], "dDetector_vu_mm": [1., 1.],
                "offDetector_vu_mm": [0., 0.],
                "rotDetector_xyz_rad": [0., 0., 0.], "COR_mm": 0.,
                "offOrigin_zyx_mm": [0., 0., 0.]}
    projection = tmp_path / "measured.npy"
    np.save(projection, np.zeros((2, 1, 1), np.float32))
    ref = ArrayRef(projection, None, "view,v,u", "line_integral",
                   hashlib.sha256(projection.read_bytes()).hexdigest())
    case = CaseManifest("1", "animal", "case", "acq", ["a", "b"], ref,
                        [0., 1.], ["a", "b"], [0, 1], geometry, grid, grid,
                        {"reference_by_phase": {"a": "must-not-open"}}, None,
                        {"normalization_status": "pending_train_fit"})
    volumes = []
    for phase in case.phase_ids:
        volume_ref = save_volume(tmp_path / (phase + ".npy"),
                                 torch.zeros(grid.shape_zyx), grid)
        volumes.append({"subject_id": "animal", "case_id": "case",
                        "acquisition_id": "acq", "phase_id": phase,
                        "grid_sha256": grid.fingerprint(),
                        "prediction_mu": volume_ref.to_dict()})
    enhanced = tmp_path / "enhanced.json"
    enhanced.write_text(json.dumps({"status": "passed", "volumes": volumes}))
    support = {}
    for label, value in (("m0", np.ones(grid.shape_zyx, np.bool_)),
                         ("core", np.pad(np.ones((4, 4, 4), np.bool_), 10))):
        path = tmp_path / (label + ".npy")
        np.save(path, value)
        support[label] = {"path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    pairs = []
    for target in case.phase_ids:
        for source in case.phase_ids:
            path = tmp_path / (target + "_" + source + ".npy")
            np.save(path, np.zeros((3, *grid.shape_zyx), np.float32))
            pairs.append({"target": target, "source": source,
                          "path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "grid_sha256": grid.fingerprint(), "unit": "mm",
                          "components": "xyz", "qc": {"status": "passed"}})
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"status": "passed", "pairs": pairs,
                                        "support": support,
                                        "direction": "target_fixed_to_source_moving",
                                        "grid_sha256": grid.fingerprint()}))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture checkpoint identity")

    class ToyProjector:
        device = torch.device("cpu")

        def project(self, volume, indices):
            return volume.sum().reshape(1, 1, 1).repeat(len(indices), 1, 1)

        def backproject(self, values, indices):
            return torch.ones(grid.shape_zyx) * values.sum()

    result = run_case(case, checkpoint=checkpoint,
                      network_config=NetworkConfig(device="cpu"),
                      reconstruction_config=ReconstructionConfig(sweeps=1),
                      output=tmp_path / "run", projector=ToyProjector(),
                      enhanced_fixture=enhanced, registration_fixture=registration)
    assert result["status"] == "passed"
    assert result["reference_evaluation"] == "not_run"
    assert set(result["targets"]) == {"a", "b"}
    resumed = run_case(case, checkpoint=checkpoint,
                       network_config=NetworkConfig(device="cpu"),
                       reconstruction_config=ReconstructionConfig(sweeps=1),
                       output=tmp_path / "run", resume=True,
                       projector=ToyProjector(), enhanced_fixture=enhanced,
                       registration_fixture=registration)
    assert resumed["stages"]["reconstructed"]["reused"] is True
