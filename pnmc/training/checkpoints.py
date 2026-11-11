"""Strict, atomic epoch-boundary checkpointing with RNG restoration."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from pnmc.runtime import _code_commit


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path: Path, *, model, optimizer, scheduler, completed_epoch: int,
                    provenance: dict, training_state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.startswith("epoch_") and path.exists():
        raise FileExistsError("epoch checkpoint already exists: " + str(path))
    payload = {"format_version": 1, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": scheduler.state_dict(),
               "completed_epoch": int(completed_epoch), "provenance": provenance,
               "training_state": training_state, "rng_state": _rng_state(),
               "code_commit": _code_commit()}
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".",
                                                   suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        if path.name.startswith("epoch_") and path.exists():
            raise FileExistsError("epoch checkpoint already exists: " + str(path))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_checkpoint(path: Path, *, model, optimizer, scheduler,
                       expected_provenance: dict) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    observed = payload.get("provenance", {})
    changed = [key for key in sorted(set(observed) | set(expected_provenance))
               if observed.get(key) != expected_provenance.get(key)]
    if changed:
        raise ValueError("checkpoint provenance differs: " + ", ".join(changed))
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "rng_state", "completed_epoch", "training_state"}
    if required.difference(payload):
        raise ValueError("checkpoint lacks complete epoch-boundary state")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    state = payload["rng_state"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])
    return payload
