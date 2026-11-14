"""The published branch contains only the pnmc directory."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


CHECKOUT = Path(__file__).resolve().parents[2]


def test_package_imports_without_repo_root_scripts():
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import pnmc.inference; import pnmc.training.engine; import pnmc.pipeline"],
        cwd=CHECKOUT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_packaged_legacy_model_matches_original_when_available():
    from pnmc.models.legacy_priornet import Dense2D

    source = os.environ.get("PNMC_LEGACY_SOURCE")
    if not source or not Path(source).is_file():
        pytest.skip("original legacy model source not available on this host")
    spec = importlib.util.spec_from_file_location("legacy_source", source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    old = legacy.Dense2D().eval()
    packaged = Dense2D().eval()
    assert list(packaged.state_dict()) == list(old.state_dict())
    packaged.load_state_dict(old.state_dict(), strict=True)
    torch.manual_seed(41)
    sparse = torch.rand(1, 1, 16, 16)
    prior = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        old_features, old_prediction = old(sparse, prior)
        new_features, new_prediction = packaged(sparse, prior)
    torch.testing.assert_close(new_features, old_features, rtol=0, atol=0)
    torch.testing.assert_close(new_prediction, old_prediction, rtol=0, atol=0)
