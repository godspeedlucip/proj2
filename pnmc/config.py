"""Strict configuration classes; unknown JSON fields are errors."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple


class _StrictConfig:
    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict):
            raise ValueError("{} configuration must be an object".format(cls.__name__))
        allowed = {field.name for field in fields(cls)}
        extra = set(value).difference(allowed)
        if extra:
            raise ValueError("unknown {} fields: {}".format(cls.__name__, sorted(extra)))
        return cls(**value)

    @classmethod
    def from_json(cls, path: Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class NetworkConfig(_StrictConfig):
    model: str = "pnnet"
    channels: Tuple[int, int, int, int, int] = (24, 48, 96, 96, 192)
    query_chunk: int = 256
    checkpoint_chunks: bool = True
    batch_size: int = 2
    epochs: int = 50
    learning_rate: float = 1e-4
    milestones: Tuple[int, int] = (20, 40)
    seed: int = 42
    device: str = "cuda:0"
    num_workers: int = 0
    loss_name: str = "ssim"
    prior_fusion: str = "add"
    use_non_local: bool = True

    def __post_init__(self) -> None:
        if self.model not in {"pnnet", "priornet_matched"}:
            raise ValueError("unknown model: " + self.model)
        if len(self.channels) != 5 or any(type(x) is not int or x <= 0 for x in self.channels):
            raise ValueError("channels must have five positive integers")
        if self.query_chunk <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("query_chunk, batch_size and epochs must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.loss_name not in {"ssim", "mse"}:
            raise ValueError("unknown loss_name: " + self.loss_name)
        if self.prior_fusion not in {"add", "concat"}:
            raise ValueError("unknown prior_fusion: " + self.prior_fusion)


@dataclass(frozen=True)
class RegistrationConfig(_StrictConfig):
    shrink_factors: Tuple[int, int, int] = (4, 2, 1)
    iterations: Tuple[int, int, int] = (80, 40, 20)
    image_sigma_mm: Tuple[float, float, float] = (2.0, 1.0, 0.0)
    field_sigma_mm: float = 1.0
    support_threshold: float = 0.01
    taper_mm: float = 10.0

    def __post_init__(self) -> None:
        if not (len(self.shrink_factors) == len(self.iterations) == len(self.image_sigma_mm)):
            raise ValueError("registration pyramid lengths disagree")
        if any(x <= 0 for x in self.shrink_factors + self.iterations):
            raise ValueError("registration shrink factors and iterations must be positive")
        if self.field_sigma_mm < 0 or self.taper_mm <= 0:
            raise ValueError("invalid registration sigma or taper")


@dataclass(frozen=True)
class ReconstructionConfig(_StrictConfig):
    sweeps: int = 20
    lambda_initial: float = 1.0
    lambda_decrement: float = 0.01
    lambda_minimum: float = 0.1
    row_relative_epsilon: float = 1e-6
    column_relative_epsilon: float = 1e-6
    zero_row_measurement_tolerance: float = 1e-6
    fdk_mode: str = "tigre_standard_approx"
    fdk_filter: str = "ramp"
    fdk_padding: int = 0
    registration: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.sweeps <= 0 or self.lambda_initial <= 0 or self.lambda_minimum <= 0:
            raise ValueError("reconstruction sweeps and lambda must be positive")
        if self.fdk_mode != "tigre_standard_approx" or self.fdk_filter != "ramp":
            raise ValueError("unsupported FDK configuration")
