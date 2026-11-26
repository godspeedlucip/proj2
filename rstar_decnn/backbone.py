"""Restore the existing RSTAR(-STAM) inference contract."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import rstar4d
import torch
from rstar4d.inference.sliding_window import predict_volume
from rstar4d.models.model import forward_volume, model_from_checkpoint
from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.utils.io import sha256

UPSTREAM_REVISION = "57d61544a0929d56307eda0e5d0edde8ce245105"


def upstream_identity():
    package = Path(rstar4d.__file__).resolve().parent
    commit = dirty = None
    git = shutil.which("git")
    if git is not None:
        revision = subprocess.run(
            [git, "-C", str(package), "rev-parse", "HEAD"], text=True, capture_output=True,
        )
        if revision.returncode == 0:
            commit = revision.stdout.strip()
            status = subprocess.run(
                [git, "-C", str(package.parent), "status", "--porcelain", "--untracked-files=no", "--", "rstar4d"],
                text=True, capture_output=True, check=True,
            )
            dirty = bool(status.stdout.strip())
    return {
        "reference_revision": UPSTREAM_REVISION,
        "revision": commit,
        "tracked_code_modified": dirty,
        "files": {p.relative_to(package).as_posix(): sha256(p) for p in sorted(package.rglob("*.py"))},
    }


def backbone_spec(value):
    keys = ("stage", "model_profile", "normalization", "intensity_domain", "calibration_application")
    result = {key: value[key] for key in keys}
    if value.get("stam"):
        result["stam"] = value["stam"]
        result["config"] = {"stam": value.get("config", {}).get("stam", {})}
    return result


def restore_backbone(path, *, joint, prior_mode, engineering=False):
    value = load_checkpoint(path, require_training=False)
    mode = (value.get("stam") or {}).get("architecture", {}).get("mode", "off")
    if value.get("stage") != 2 or mode != ("joint" if joint else "off"):
        raise ValueError(f"{path}: incompatible Stage II architecture ({mode})")
    spec = backbone_spec(value)
    norm = spec["normalization"]
    if set(norm) != {"scale", "offset"} or norm["scale"] <= 0 or not np.isfinite(list(norm.values())).all():
        raise ValueError(f"{path}: invalid checkpoint normalization")
    snapshot = value.get("data_snapshot", {})
    observed = set()
    for case in snapshot.get("cases", []):
        mode_from_case = case.get("prior_mode") or case.get("metadata", {}).get("prior_mode")
        if mode_from_case:
            observed.add(mode_from_case)
    settings = value.get("config", {})
    if settings.get("prior_mode"):
        observed.add(settings["prior_mode"])
    if observed and observed != {prior_mode}:
        raise ValueError(f"{path}: recorded prior mode {sorted(observed)} differs from {prior_mode}")
    model = model_from_checkpoint({**spec, "state_dict": value["state_dict"]})
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values() if tensor.is_floating_point()):
        raise ValueError(f"{path}: nonfinite backbone state")
    source = {
        "path": str(Path(path).resolve()), "sha256": sha256(path), "prior_mode": prior_mode,
        "prior_mode_evidence": "checkpoint" if observed else "operator-attested",
        "engineering": engineering, "data_snapshot": snapshot, "split": value.get("split"),
        "config": settings, "completed_epoch": value.get("completed_epoch"),
    }
    return model, spec, source


def check_case(spec, row):
    meta = row["metadata"]
    for field in ("normalization", "intensity_domain", "calibration_application"):
        if spec[field] != meta[field]:
            raise ValueError(f"{row['case_id']}: backbone {field} differs from input")
    if spec["model_profile"] == "pig_ordered_v1" and meta["temporal_topology"] != "ordered_states":
        raise ValueError(f"{row['case_id']}: ordered backbone requires ordered states")


def normalize(x, contract):
    return (x + contract["offset"]) / contract["scale"]


def physical(x, contract):
    return x * contract["scale"] - contract["offset"]


def backbone_forward(model, x, prior, normalization):
    expanded = prior.unsqueeze(2).expand_as(x)
    return forward_volume(model, normalize(torch.cat((x, expanded), dim=1), normalization))


def predict_backbone(model, x, prior, spec, *, mode, tile, overlap, device, amp):
    return predict_volume(model, x, prior, spec["normalization"], mode, tile, overlap, device, amp)
