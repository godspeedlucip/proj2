from __future__ import annotations

import os
import random
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import atomic_path


@contextmanager
def training_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            def acquire():
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f"A training process already owns {output}") from error
        try:
            yield
        finally:
            release()


def save_torch(path, state):
    with atomic_path(path) as temporary, temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())


def latest_checkpoint(output):
    paths = [path for path in (output / "checkpoints").glob("epoch_*.pt")
             if re.fullmatch(r"epoch_\d+\.pt", path.name)]
    return max(paths, key=lambda path: int(path.stem.split("_")[1]), default=None)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def random_state(sampler, device):
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (name, keys.tolist(), position, has_gauss, cached),
            "torch": torch.get_rng_state(), "sampler": sampler.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_random_state(state, sampler, device):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    sampler.set_state(state["sampler"])
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved random state")
        torch.cuda.set_rng_state_all(state["cuda"])
