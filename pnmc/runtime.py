"""Atomic run creation and runtime provenance."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _version(package: str):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report() -> dict:
    import torch

    available = bool(torch.cuda.is_available())
    probe = {"status": "not_run", "reason": "CUDA unavailable"}
    if available:
        try:
            vector = torch.ones(4, device="cuda:0", dtype=torch.float32)
            result = (vector * 2).sum()
            torch.cuda.synchronize()
            if result.item() != 8.0:
                raise RuntimeError("unexpected CUDA arithmetic result")
            probe = {"status": "passed", "device": torch.cuda.get_device_name(0)}
        except Exception as exc:
            probe = {"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)}
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "numpy": _version("numpy"), "SimpleITK": _version("SimpleITK"),
            "tifffile": _version("tifffile"), "nibabel": _version("nibabel"),
            "tigre": _version("tigre"), "pytest": _version("pytest"),
            "cuda_available": available, "cuda_kernel_probe": probe}


def _code_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def create_run(root: Path, run_id: str, provenance: dict) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be a safe directory name")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = root / run_id
    try:
        run.mkdir()
    except FileExistsError:
        report_path = run / "run.json"
        if not report_path.is_file():
            raise FileExistsError("run exists without complete run.json: " + str(run))
        with report_path.open("r", encoding="utf-8") as stream:
            prior = json.load(stream)
        if prior.get("provenance") != provenance:
            raise FileExistsError("run_id belongs to different provenance: " + str(run))
        return run
    report = {"status": "not_run", "config_sha256": provenance.get("config_sha256"),
              "input_sha256": provenance.get("input_sha256"),
              "code_commit": _code_commit(), "environment": environment_report(),
              "artifacts": {}, "provenance": provenance}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=run, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, run / "run.json")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return run
