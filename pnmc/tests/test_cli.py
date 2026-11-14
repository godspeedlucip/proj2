"""Published commands expose PN-Net inference and physical reconstruction."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pnmc.config import NetworkConfig, ReconstructionConfig


CHECKOUT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("command", ["infer", "register", "reconstruct"])
def test_subcommand_help_is_available_without_root_scripts(command):
    result = subprocess.run([sys.executable, "-B", "-m", "pnmc", command,
                             "--help"], cwd=CHECKOUT, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout
    if command == "reconstruct":
        assert "--resume" in result.stdout
        assert "--backend" not in result.stdout


def test_reconstruct_passes_case_checkpoint_config_and_resume(tmp_path, monkeypatch,
                                                              capsys):
    from pnmc import cli

    network_path = tmp_path / "network.json"
    network_path.write_text(json.dumps(asdict(NetworkConfig())), encoding="utf-8")
    reconstruction_path = tmp_path / "reconstruction.json"
    reconstruction_path.write_text(json.dumps(asdict(ReconstructionConfig())),
                                   encoding="utf-8")
    case_path = tmp_path / "case.json"
    case_path.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    case = type("Case", (), {"case_id": "case-1"})()
    monkeypatch.setattr(cli, "load_case_manifest", lambda path, **kw: case)
    called = {}

    def fake_run_case(actual_case, **kwargs):
        called.update(case=actual_case, **kwargs)
        return {"status": "passed", "case_id": actual_case.case_id}

    monkeypatch.setattr(cli, "run_case", fake_run_case)
    output = tmp_path / "output"
    assert cli.main(["reconstruct", "--case", str(case_path),
                     "--checkpoint", str(checkpoint),
                     "--network-config", str(network_path),
                     "--config", str(reconstruction_path),
                     "--output", str(output), "--device", "cpu",
                     "--resume"]) == 0
    assert called["case"] is case
    assert called["checkpoint"] == checkpoint
    assert called["network_config"].model == "pnnet"
    assert called["network_config"].device == "cpu"
    assert called["reconstruction_config"].sweeps == 20
    assert called["output"] == output
    assert called["resume"] is True
    assert json.loads(capsys.readouterr().out)["case_id"] == "case-1"


def test_invalid_config_exits_nonzero_without_traceback(tmp_path, capsys):
    from pnmc import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["reconstruct", "--case", str(tmp_path / "case.json"),
                  "--checkpoint", str(tmp_path / "best.pt"),
                  "--network-config", str(tmp_path / "missing.json"),
                  "--config", str(tmp_path / "reconstruction.json"),
                  "--output", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert "failed" in capsys.readouterr().err


def test_candidate_fallback_audits_failure_and_selects_next(tmp_path):
    from pnmc.cli import run_candidates

    class Candidate:
        def __init__(self, name):
            self.case_id = name

    attempted = []

    def runner(case, output):
        attempted.append(case.case_id)
        if case.case_id == "a":
            raise ValueError("QC failed")
        return {"status": "passed", "case_id": case.case_id}

    result = run_candidates([Candidate("b"), Candidate("a")], tmp_path,
                            runner=runner)
    audit = json.loads((tmp_path / "selection_audit.json").read_text())
    assert attempted == ["a", "b"]
    assert audit["candidates"][0]["reason"] == "QC failed"
    assert audit["selected_case_id"] == result["case_id"] == "b"
