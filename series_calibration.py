"""Load either a per-round, per-series calibration table or a legacy compare.csv."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from generate_pig_first_phase_prior import load_attenuation_calibration


REQUIRED_COLUMNS = (
    "series", "poly_a2", "poly_a1", "poly_a0", "truth_air",
    "prior_air", "truth_tissue", "prior_tissue", "truth_contrast",
    "prior_contrast", "fit_design_degree", "contrast_residual_mu", "warnings",
)


def validate_calibration_round(csv_path: Path | None, round_root: Path) -> None:
    """Reject an explicitly named round CSV for a different source round."""
    if csv_path is None:
        return
    source_match = re.search(r"round[-_]?([0-9]+)", round_root.name, re.IGNORECASE)
    csv_match = re.search(r"round[-_]?([0-9]+)", csv_path.name, re.IGNORECASE)
    if source_match and csv_match and source_match.group(1) != csv_match.group(1):
        raise ValueError(
            f"Calibration {csv_path.name} is for round {csv_match.group(1)}, "
            f"but source {round_root.name} is round {source_match.group(1)}"
        )


def load_series_calibration(
    csv_path: Path, series: str,
) -> tuple[np.poly1d, float, float, dict[str, Any]]:
    """Select exactly one series row; retain old global compare.csv support."""
    csv_path = csv_path.resolve(strict=True)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if "series" not in columns:
            polynomial, air_threshold, air_value, details = load_attenuation_calibration(csv_path)
            details["calibration_scope"] = "global_legacy"
            details["calibration_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            return polynomial, air_threshold, air_value, details
        missing_columns = set(REQUIRED_COLUMNS) - set(columns)
        if missing_columns:
            raise ValueError(f"Missing calibration columns in {csv_path}: {sorted(missing_columns)}")
        matches = [row for row in reader if row["series"].strip() == series]
    if len(matches) != 1:
        raise ValueError(f"Expected one calibration row for {series!r} in {csv_path}; found {len(matches)}")
    row = matches[0]
    numeric = {}
    for name in REQUIRED_COLUMNS[1:-1]:
        numeric[name] = float(row[name])
        if not np.isfinite(numeric[name]):
            raise ValueError(f"Nonfinite {name} for {series!r} in {csv_path}")
    degree = int(numeric["fit_design_degree"])
    if degree not in (1, 2):
        raise ValueError(f"Invalid fit_design_degree={degree} for {series!r}")
    coefficients = [numeric[name] for name in ("poly_a2", "poly_a1", "poly_a0")]
    polynomial = np.poly1d(coefficients)
    air_value = numeric["truth_air"]
    air_threshold = (air_value + numeric["truth_tissue"]) / 2.0
    # A row-specific digest avoids invalidating unrelated series when another
    # row in the same round CSV changes.
    row_digest = hashlib.sha256(json.dumps(
        {name: row[name] for name in REQUIRED_COLUMNS},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    details = {
        "source_csv": str(csv_path),
        "series": series,
        "calibration_scope": "per_series",
        "calibration_sha256": row_digest,
        "input_means_public_dataset": [numeric[f"prior_{name}"] for name in ("contrast", "tissue", "air")],
        "target_means_measured": [numeric[f"truth_{name}"] for name in ("contrast", "tissue", "air")],
        "polynomial_degree": 2,
        "fit_design_degree": degree,
        "polynomial_coefficients": coefficients,
        "air_threshold": air_threshold,
        "air_value": air_value,
        "contrast_residual_mu": numeric["contrast_residual_mu"],
        "warnings": row["warnings"],
    }
    return polynomial, air_threshold, air_value, details


def apply_reconstruction_calibration(image: np.ndarray, polynomial: np.poly1d) -> np.ndarray:
    """Apply the reviewed fit after FDK, preserving exact zero background."""
    result = np.asarray(image, dtype=np.float32)
    if result.ndim != 3:
        raise ValueError(f"Expected 3-D reconstructed image; got {result.shape}")
    if not result.flags.writeable:
        result = result.copy()
    for z0 in range(0, result.shape[0], 16):
        z1 = min(z0 + 16, result.shape[0])
        raw = result[z0:z1].copy()
        mapped = np.polyval(polynomial, raw).astype(np.float32)
        mapped[raw == 0] = 0.0
        result[z0:z1] = mapped
    return result
