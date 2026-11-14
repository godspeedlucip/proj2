"""Reject unusable checkpoints and cases before physical reconstruction."""

from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from pnmc.config import NetworkConfig, ReconstructionConfig
from pnmc.data.schema import ArrayRef, GridSpec
from pnmc.models.pnnet import PNNet2D


def make_inputs(tmp_path):
    grid = GridSpec((16, 16, 16), (1., 1., 1.), (-7.5, -7.5, -7.5),
                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "scanner")
    projection = tmp_path / "projection.npy"
    np.save(projection, np.zeros((2, 1, 1), np.float32))
    ref = ArrayRef(projection, None, "view,v,u", "line_integral",
                   hashlib.sha256(projection.read_bytes()).hexdigest())
    case = {
        "schema_version": "1", "subject_id": "subject", "case_id": "case",
        "acquisition_id": "acq", "phase_ids": ["a", "b"],
        "projections": ref.to_dict(), "angles_rad": [0., 1.],
        "phase_id_per_view": ["a", "b"], "view_id": [0, 1],
        "geometry": {"DSO_mm": 500., "DSD_mm": 900.,
                     "nDetector_vu": [1, 1], "dDetector_vu_mm": [1., 1.],
                     "offDetector_vu_mm": [0., 0.],
                     "rotDetector_xyz_rad": [0., 0., 0.], "COR_mm": 0.,
                     "offOrigin_zyx_mm": [0., 0., 0.]},
        "projection_volume_grid": grid.to_dict(),
        "reconstruction_grid": grid.to_dict(),
        "normalization": None,
        "provenance": {"normalization_status": "pending_train_fit"},
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    network = NetworkConfig(channels=(2, 4, 8, 8, 16), query_chunk=8,
                            device="cpu")
    network_path = tmp_path / "network.json"
    network_path.write_text(json.dumps(asdict(network)), encoding="utf-8")
    reconstruction_path = tmp_path / "reconstruction.json"
    reconstruction_path.write_text(json.dumps(asdict(ReconstructionConfig())),
                                   encoding="utf-8")
    architecture = {"model": "pnnet", "channels": list(network.channels),
                    "query_chunk": network.query_chunk,
                    "checkpoint_chunks": network.checkpoint_chunks,
                    "prior_fusion": network.prior_fusion,
                    "use_non_local": network.use_non_local}
    model = PNNet2D(network.channels, network.query_chunk,
                    network.checkpoint_chunks, network.prior_fusion,
                    network.use_non_local)
    checkpoint = tmp_path / "best.pt"
    payload = {"format_version": 1, "model_state_dict": model.state_dict(),
               "provenance": {"architecture": architecture,
                              "normalization": {
                                  "mode": "affine_clipped", "a": 0., "b": 1.,
                                  "fit_manifest_sha256": "fit"},
                              "train_grid_hashes": [grid.fingerprint()]}}
    torch.save(payload, checkpoint)
    return case_path, checkpoint, network_path, reconstruction_path


def _check(paths, *, device="cpu"):
    from pnmc import preflight

    return preflight.check_reconstruction_inputs(*paths, device=device)


def test_valid_checkpoint_and_physical_case_pass_without_outputs(tmp_path,
                                                                 monkeypatch):
    from pnmc import preflight

    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight, "_check_dependencies", lambda: {
        "tigre": "available", "SimpleITK": "available"})
    report = _check(paths)
    assert report["status"] == "passed"
    assert report["architecture"]["model"] == "pnnet"
    assert report["train_grid_sha256"] == report["case_grid_sha256"]
    assert report["checkpoint_sha256"] == hashlib.sha256(
        paths[1].read_bytes()).hexdigest()
    assert not (tmp_path / "out").exists()


def test_rejects_wrong_training_grid_before_fdk(tmp_path, monkeypatch):
    from pnmc import preflight

    paths = make_inputs(tmp_path)
    payload = torch.load(paths[1], map_location="cpu", weights_only=False)
    payload["provenance"]["train_grid_hashes"] = ["0" * 64]
    torch.save(payload, paths[1])
    monkeypatch.setattr(preflight, "_check_dependencies", lambda: {})
    with pytest.raises(ValueError, match="grid"):
        _check(paths)
    assert not (tmp_path / "out").exists()


def test_rejects_non_pnnet_checkpoint_and_config_mismatch(tmp_path,
                                                          monkeypatch):
    from pnmc import preflight

    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight, "_check_dependencies", lambda: {})
    payload = torch.load(paths[1], map_location="cpu", weights_only=False)
    payload["provenance"]["architecture"]["model"] = "priornet_matched"
    torch.save(payload, paths[1])
    with pytest.raises(ValueError, match="PN-Net|pnnet"):
        _check(paths)
    payload["provenance"]["architecture"]["model"] = "pnnet"
    torch.save(payload, paths[1])
    network = json.loads(paths[2].read_text())
    network["channels"] = [4, 4, 8, 8, 16]
    paths[2].write_text(json.dumps(network), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture"):
        _check(paths)


def test_rejects_missing_projection_and_unavailable_cuda(tmp_path,
                                                          monkeypatch):
    from pnmc import preflight

    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight, "_check_dependencies", lambda: {})
    (tmp_path / "projection.npy").unlink()
    with pytest.raises(ValueError, match="projection|does not exist"):
        _check(paths)
    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        _check(paths, device="cuda:0")


def test_corrupt_checkpoint_and_invalid_cuda_index_fail_clearly(tmp_path,
                                                                monkeypatch):
    from pnmc import preflight

    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight, "_check_dependencies", lambda: {})
    paths[1].write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        _check(paths)
    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA device"):
        _check(paths, device="cuda:5")


def test_cli_check_only_prints_report_without_output_directory(tmp_path,
                                                                monkeypatch,
                                                                capsys):
    from pnmc import cli, preflight

    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight, "_check_dependencies", lambda: {})
    output = tmp_path / "out"
    assert cli.main(["reconstruct", "--case", str(paths[0]),
                     "--checkpoint", str(paths[1]),
                     "--network-config", str(paths[2]),
                     "--config", str(paths[3]),
                     "--output", str(output), "--device", "cpu",
                     "--check-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert not output.exists()


@pytest.mark.parametrize("override", [
    {"row_relative_epsilon": -1.},
    {"column_relative_epsilon": 1.},
    {"zero_row_measurement_tolerance": -1.},
    {"fdk_padding": -1},
    {"registration": {"support_threshold": 1.2}},
    {"registration": {"image_sigma_mm": [-1., 0., 0.]}},
])
def test_cli_preflight_rejects_runtime_invalid_parameters(tmp_path, monkeypatch,
                                                           capsys, override):
    from pnmc import cli, preflight

    paths = make_inputs(tmp_path)
    monkeypatch.setattr(preflight, "_check_dependencies", lambda: {})
    config = json.loads(paths[3].read_text())
    config.update(override)
    paths[3].write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        cli.main(["reconstruct", "--case", str(paths[0]),
                  "--checkpoint", str(paths[1]),
                  "--network-config", str(paths[2]),
                  "--config", str(paths[3]),
                  "--output", str(output), "--device", "cpu",
                  "--check-only"])
    assert exc.value.code == 2
    assert "failed" in capsys.readouterr().err
    assert not output.exists()
