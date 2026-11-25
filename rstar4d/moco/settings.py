from __future__ import annotations

import math
from pathlib import Path
import re

from ..utils.io import resolve


def defaults():
    return {
        "seed": 2026,
        "threads": 4,
        "device": "cuda",
        "baseline": {
            "mode": "tiled", "tile": [16, 128, 128], "overlap": [8, 64, 64],
            "amp": True, "checkpoint_sha256": None,
        },
        "registration": {
            "mode": "learned", "input": "b", "train_pairs": "adjacent",
            "normalization": {"scale": .02, "offset": 0.},
            "epochs": 50, "learning_rate": 1e-4, "min_lr": 1e-5,
            "steps_per_epoch": 0, "patch": [16, 128, 128],
            "context": [9, 9, 9], "max_displacement_vox": 4.,
            "ncc_window": 9, "smooth_weight": 4.,
            "inference_tile": None, "inference_halo": 64,
        },
        "fusion": {"pairs": "adjacent", "source": "x", "save_raw_sum": False},
        "pigae": {
            "mode": "off", "rtol": None, "atol": None, "tolerance_evidence": None,
            "view_batch": 64,
        },
        "reinjection": {"sigma_mm": 1., "alpha": 0.},
        "outputs": ["b"],
    }


def dependencies(config):
    outputs = set(config["outputs"])
    reinject = bool(outputs & {"y_single", "y_multi"})
    fuse = "d" in outputs or reinject
    registration = config["registration"]
    sources = outputs & {"x", "b"}
    if fuse:
        sources.add(config["fusion"]["source"])
        if registration["mode"] == "learned":
            sources.add(registration["input"])
    if reinject:
        if config["fusion"]["source"] != "c":
            raise ValueError("y_single/y_multi require fusion.source=c")
        sources.add("b")
    if "c" in sources and config["pigae"]["mode"] == "off":
        raise ValueError("Requested output needs c but pigae.mode=off")
    return sources, fuse, reinject


def validate(config, base):
    for key, values in {
        "baseline.mode": ("tiled", "full"),
        "registration.mode": ("learned", "zero"),
        "registration.input": ("x", "b"),
        "registration.train_pairs": ("adjacent", "all"),
        "fusion.pairs": ("adjacent", "all"),
        "fusion.source": ("x", "b", "c"),
        "pigae.mode": ("off", "static_consistent", "printed"),
        "device": ("cpu", "cuda"),
    }.items():
        value = config
        for field in key.split("."):
            value = value[field]
        if value not in values:
            raise ValueError(f"{key} must be one of {values}")
    outputs = config["outputs"]
    if (not isinstance(outputs, list) or not outputs or any(not isinstance(name, str) for name in outputs)
            or len(set(outputs)) != len(outputs)):
        raise ValueError("outputs must be a nonempty list of unique outputs")
    if set(outputs) - {"x", "b", "d", "y_single", "y_multi"}:
        raise ValueError("outputs contains an unknown output")
    reg, baseline, pigae = (config[key] for key in ("registration", "baseline", "pigae"))
    for value, field in [(baseline["amp"], "baseline.amp"),
                         (config["fusion"]["save_raw_sum"], "fusion.save_raw_sum")]:
        if type(value) is not bool:
            raise ValueError(f"{field} must be a Boolean")
    expected_sha = baseline["checkpoint_sha256"]
    if expected_sha is not None and (not isinstance(expected_sha, str)
                                     or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)):
        raise ValueError("baseline.checkpoint_sha256 must be a lowercase SHA-256 hex digest")
    for value, low, field in [
        (reg["patch"], 1, "registration.patch"),
        (reg["context"], 0, "registration.context"),
        (reg["inference_tile"], 1, "registration.inference_tile"),
        (baseline["tile"], 1, "baseline.tile"),
        (baseline["overlap"], 0, "baseline.overlap"),
    ]:
        if value is None and field in ("registration.patch", "registration.inference_tile"):
            continue
        if not isinstance(value, list) or len(value) != 3 or any(type(n) is not int or n < low for n in value):
            raise ValueError(f"{field} must be a ZYX list of three integers >= {low}")
    if any(o >= n for n, o in zip(baseline["tile"], baseline["overlap"])):
        raise ValueError("baseline.overlap must be smaller than baseline.tile")
    for value, minimum, field in [
        (reg["epochs"], 1, "registration.epochs"),
        (reg["ncc_window"], 1, "registration.ncc_window"),
        (reg["steps_per_epoch"], 0, "registration.steps_per_epoch"),
        (reg["inference_halo"], 64, "registration.inference_halo"),
        (config["threads"], 1, "threads"),
        (config["seed"], 0, "seed"),
        (pigae["view_batch"], 1, "pigae.view_batch"),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if reg["ncc_window"] % 2 == 0:
        raise ValueError("registration.ncc_window must be odd")
    for value, low, field in [
        (reg["normalization"]["scale"], 0., "registration.normalization.scale"),
        (reg["normalization"]["offset"], -math.inf, "registration.normalization.offset"),
        (reg["learning_rate"], 0., "registration.learning_rate"),
        (reg["min_lr"], 0., "registration.min_lr"),
        (reg["smooth_weight"], -1., "registration.smooth_weight"),
        (reg["max_displacement_vox"], 0., "registration.max_displacement_vox"),
        (config["reinjection"]["sigma_mm"], 0., "reinjection.sigma_mm"),
    ]:
        if type(value) not in (float, int) or not math.isfinite(value) or value <= low:
            raise ValueError(f"Invalid {field}")
    if reg["smooth_weight"] < 0 or reg["min_lr"] > reg["learning_rate"]:
        raise ValueError("Invalid registration smoothness weight or learning rate schedule")
    if reg["patch"] is not None and min(reg["context"]) < (
        math.ceil(reg["max_displacement_vox"]) + reg["ncc_window"] // 2 + 1
    ):
        raise ValueError("registration.context must cover maximum displacement, interpolation and NCC")
    alpha = config["reinjection"]["alpha"]
    if type(alpha) not in (float, int) or alpha not in (0., .25, .5, 1.):
        raise ValueError("reinjection.alpha must be 0, .25, .5 or 1")
    for field in ("rtol", "atol"):
        tolerance = pigae[field]
        if tolerance is not None and (type(tolerance) not in (float, int) or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError(f"pigae.{field} must be finite and nonnegative")
    if pigae["tolerance_evidence"]:
        if not isinstance(pigae["tolerance_evidence"], str):
            raise ValueError("pigae.tolerance_evidence must be a file path")
        pigae["tolerance_evidence"] = str(resolve(pigae["tolerance_evidence"], Path(base)))
    return config
