"""Self-contained inference artifacts and transactional training state."""
import json
import random

import numpy as np
import torch
from rstar4d.models.model import model_from_checkpoint
from rstar4d.training.checkpoint import atomic_save
from rstar4d.utils.io import canonical_hash

from .backbone import upstream_identity
from .decnn import WIDTHS
from .model import Combined


VERSION = 1


def architecture(upsampling):
    return {"version": 1, "widths": list(WIDTHS), "upsampling": upsampling, "dropout": 0.5,
            "batchnorm_eps": 1e-5, "batchnorm_momentum": 0.1}


def rng_state(sampler, device):
    return {
        "python": random.getstate(), "numpy": json.dumps(sampler.bit_generator.state),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, sampler, device):
    random.setstate(state["python"])
    sampler.bit_generator.state = json.loads(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def payload(model, backbone, config, data, upstream, source, policy):
    value = {
        "format": "rstar-decnn", "version": VERSION, "recipe": config["recipe"], "policy": policy,
        "decnn": architecture(config["upsampling"]) if model.decnn is not None else None,
        "backbone": backbone, "source": source, "upstream": upstream,
        "config": config, "data": data, "alpha": float(model.alpha.detach()),
        "weights": model.state_dict(),
    }
    value["contract_hash"] = canonical_hash({k: v for k, v in value.items() if k != "weights"})
    return value


def publish(path, value):
    atomic_save(path, value)


def load(path, *, training=False):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != "rstar-decnn" or value.get("version") != VERSION:
        raise ValueError("unsupported combination checkpoint")
    metadata = {k: v for k, v in value.items() if k not in {"weights", "training", "contract_hash"}}
    if value["contract_hash"] != canonical_hash(metadata):
        raise ValueError("checkpoint contract checksum differs")
    upstream = upstream_identity()
    if value["upstream"]["files"] != upstream["files"]:
        raise ValueError("upstream source files differ from the checkpoint")
    if not 0 <= value["alpha"] <= 1 or float(value["weights"]["alpha"]) != value["alpha"]:
        raise ValueError("checkpoint alpha differs")
    config = value["config"]
    decnn = value["decnn"]
    policy = value["policy"]
    if value["recipe"] != config["recipe"] or (policy == "stage1") != (decnn is None):
        raise ValueError("checkpoint recipe or stage differs")
    if decnn is not None and decnn != architecture(config["upsampling"]):
        raise ValueError("checkpoint DeCNN architecture differs")
    if training and "training" not in value:
        raise ValueError("inference checkpoint cannot resume training")
    return value


def model_from(value, device):
    spec = value["backbone"]
    state = {k.removeprefix("backbone."): t for k, t in value["weights"].items() if k.startswith("backbone.")}
    backbone = model_from_checkpoint({**spec, "state_dict": state})
    model = Combined(backbone, value["alpha"], value["config"]["upsampling"], decnn=value["decnn"] is not None)
    model.load_state_dict(value["weights"], strict=True)
    model.set_policy(value["policy"])
    return model.to(device)
