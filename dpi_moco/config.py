from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml


def read_config(path):
    user = (yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}) if path else {}
    if not isinstance(user, dict):
        raise ValueError("Expected a YAML mapping for the configuration")
    profile = user.get("profile", "paper_equations")
    profiles = {"paper_equations": ("printed", "sum"),
                "engineering": ("static_consistent", "mean")}
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}")
    pigae, fusion = profiles[profile]
    options = {
        "profile": profile, "pigae_mode": pigae, "fusion_mode": fusion,
        "critic_profile": "local_highpass8", "regis_variant": "vm2_cvpr2018",
        "intensity": {"scale": 0.02, "offset": 0.0}, "seed": 0,
        "gamma": 0.0007,
        "fdk": {"filter": "ram_lak", "padding_vu": [0, 0], "dowang": False,
                "parker": False, "angle_scale": "2*pi/n_views"},
    }
    for key, value in user.items():
        if key not in options:
            raise ValueError(f"Unknown configuration option: {key}")
        if isinstance(options[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping for {key}")
            if set(value) - options[key].keys():
                raise ValueError(f"Unknown {key} options: {set(value) - options[key].keys()}")
            options[key].update(value)
        else:
            options[key] = value
    if options["pigae_mode"] not in {"printed", "static_consistent", "off"}:
        raise ValueError("Invalid pigae_mode")
    if options["fusion_mode"] not in {"sum", "mean"}:
        raise ValueError("Invalid fusion_mode")
    if (options["critic_profile"] != "local_highpass8" or
            options["regis_variant"] != "vm2_cvpr2018"):
        raise ValueError("Unsupported network profile")
    scale, offset = options["intensity"]["scale"], options["intensity"]["offset"]
    if not all(math.isfinite(x) for x in (scale, offset, options["gamma"])) or scale <= 0 or options["gamma"] < 0:
        raise ValueError("Invalid intensity transform or noise deviation")
    fdk = options["fdk"]
    if fdk["parker"] or fdk["angle_scale"] != "2*pi/n_views" or fdk["filter"] != "ram_lak":
        raise ValueError("This backend supports standard TIGRE Ram–Lak FDK")
    return options


def network_units(value, config):
    return (value - config["intensity"]["offset"]) / config["intensity"]["scale"]


def physical_units(value, config):
    return value * config["intensity"]["scale"] + config["intensity"]["offset"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software(device=None):
    import torch

    try:
        tigre = importlib.metadata.version("pytigre")
    except importlib.metadata.PackageNotFoundError:
        tigre = None
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "tigre": tigre,
            "device": str(device) if device is not None else None,
            "gpu": torch.cuda.get_device_name(device) if device and device.type == "cuda" else None,
            "precision": "float32"}
