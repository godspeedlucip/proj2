"""Build a fixed, uniformly spaced acquisition schedule for phase simulation."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

if __package__:
    from .utils.phase_mapping import build_projection_phase_mapping, write_projection_phase_mapping
else:
    from utils.phase_mapping import build_projection_phase_mapping, write_projection_phase_mapping


def prepare_schedule(
    output_dir: Path,
    *,
    n_projections: int = 600,
    n_cycles: int = 20,
    n_phases: int = 10,
    min_phase: int = 1,
    max_phase: int = 10,
    angle_span_deg: float = 300.0,
) -> tuple[Path, Path]:
    """Write cycle markers and one phase row for each acquired projection."""
    if n_cycles < 1 or n_projections < 1 or n_projections % n_cycles:
        raise ValueError("n_projections must be positive and divisible by n_cycles")
    width = n_projections // n_cycles
    if width % 2:
        raise ValueError("projections per cycle must be even")
    if n_phases < 2 or not (1 <= min_phase <= n_phases) or not (1 <= max_phase <= n_phases):
        raise ValueError("phase endpoints must be within 1..n_phases")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle_csv = output_dir / "cycle.csv"
    mapping_csv = output_dir / "projection_phase_mapping.csv"
    with cycle_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("cycle", "max", "min"))
        for cycle in range(1, n_cycles + 1):
            writer.writerow((cycle, (cycle - 1) * width + width // 2, cycle * width))
    mapping = build_projection_phase_mapping(
        cycle_csv, n_phases, min_phase=min_phase, max_phase=max_phase,
        n_projections=n_projections, angle_span_deg=angle_span_deg,
    )
    # The shared minimum at 30, 60, ... starts the next half-open cycle.
    write_projection_phase_mapping(
        (replace(row, cycle=row.projection_index // width + 1) for row in mapping),
        mapping_csv,
    )
    return cycle_csv, mapping_csv
