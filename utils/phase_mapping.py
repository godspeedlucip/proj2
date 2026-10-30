"""Map projection indices to simulated phase images.

The cycle markers describe one minimum-to-maximum-to-minimum cycle:

    cycle_start(minimum) -> max -> min

The first cycle starts at ``start_index`` (0 by default), and every later
cycle starts at the previous cycle's minimum index.  A phase sequence that is
ordered from minimum to maximum is used on the first half and reversed on the
second half.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CycleMarker:
    """The markers and inferred start index of one cycle."""

    cycle: int
    start_index: int
    max_index: int
    min_index: int


@dataclass(frozen=True)
class ProjectionPhase:
    """The simulated phase assigned to one projection index."""

    projection_index: int
    cycle: int
    branch: str
    phase_position: int
    phase_label: int
    phase_fraction: float
    angle_deg: float | None


def load_cycle_markers(
    cycle_csv: str | Path,
    *,
    start_index: int = 0,
) -> tuple[CycleMarker, ...]:
    """Read cycle markers and infer each cycle's starting minimum index.

    The row with ``cycle=0`` is treated as an optional metadata row.  For the
    first real cycle, ``start_index`` is used.  For every subsequent cycle,
    the previous cycle's ``min`` is used as its start.
    """

    csv_path = Path(cycle_csv)
    rows: list[tuple[int, int, int]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"cycle", "max", "min"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("cycle.csv must contain cycle,max,min columns")

        for row in reader:
            cycle_text = (row.get("cycle") or "").strip()
            if not cycle_text:
                continue
            cycle = int(cycle_text)
            max_text = (row.get("max") or "").strip()
            min_text = (row.get("min") or "").strip()

            # The supplied file uses cycle 0 as a metadata/start row.
            if cycle == 0 and (not max_text or not min_text):
                continue
            if not max_text or not min_text:
                raise ValueError(f"cycle {cycle} must have both max and min")
            rows.append((cycle, int(max_text), int(min_text)))

    if not rows:
        raise ValueError(f"no usable cycle rows found in {csv_path}")

    rows.sort(key=lambda item: item[0])
    if len({cycle for cycle, _, _ in rows}) != len(rows):
        raise ValueError("cycle numbers must be unique")

    markers: list[CycleMarker] = []
    current_start = int(start_index)
    for cycle, max_index, min_index in rows:
        if max_index <= current_start:
            raise ValueError(
                f"cycle {cycle}: max index {max_index} must be greater than "
                f"start index {current_start}"
            )
        if min_index <= max_index:
            raise ValueError(
                f"cycle {cycle}: min index {min_index} must be greater than "
                f"max index {max_index}"
            )
        markers.append(
            CycleMarker(
                cycle=cycle,
                start_index=current_start,
                max_index=max_index,
                min_index=min_index,
            )
        )
        current_start = min_index

    return tuple(markers)


def build_phase_order(
    n_phases: int,
    *,
    min_phase: int,
    max_phase: int,
    phase_labels: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return phase labels in cyclic order from minimum to maximum.

    By default, phase labels are ``1..n_phases``.  ``phase_labels`` can be
    supplied for zero-based or custom labels.  The order wraps around the end
    of the label sequence, so min=2 and max=1 with labels 1..10 gives
    ``(2, 3, ..., 10, 1)``.
    """

    if n_phases < 2:
        raise ValueError("n_phases must be at least 2")
    labels = tuple(range(1, n_phases + 1) if phase_labels is None else phase_labels)
    if len(labels) != n_phases:
        raise ValueError("phase_labels length must equal n_phases")
    if len(set(labels)) != len(labels):
        raise ValueError("phase_labels must be unique")
    if min_phase == max_phase:
        raise ValueError("min_phase and max_phase must be different")
    if min_phase not in labels or max_phase not in labels:
        raise ValueError("min_phase and max_phase must be present in phase_labels")

    min_position = labels.index(min_phase)
    max_position = labels.index(max_phase)
    order: list[int] = []
    position = min_position
    for _ in range(n_phases):
        order.append(labels[position])
        if position == max_position:
            return tuple(order)
        position = (position + 1) % n_phases

    raise ValueError("could not build a phase order")


def _nearest_phase_position(fraction: float, n_phases: int) -> int:
    """Map a 0..1 segment fraction to the nearest phase sample."""

    position = math.floor(fraction * (n_phases - 1) + 0.5)
    return min(max(position, 0), n_phases - 1)


def build_projection_phase_mapping(
    cycle_csv: str | Path,
    n_phases: int,
    *,
    min_phase: int,
    max_phase: int,
    phase_labels: Sequence[int] | None = None,
    start_index: int = 0,
    n_projections: int | None = None,
    angle_span_deg: float | None = 300.0,
) -> tuple[ProjectionPhase, ...]:
    """Build a mapping for every integer index in the inferred cycles.

    The output has one row per actual projection index.  The first cycle
    covers ``start_index..min[cycle 1]`` and later cycles start at the
    previous minimum.  The final minimum is treated as an exclusive boundary,
    so a 600-angle scan produces indices ``0..599`` when its final marker is
    600.  Shared boundaries are emitted once.
    """

    markers = load_cycle_markers(cycle_csv, start_index=start_index)
    final_boundary = markers[-1].min_index
    if n_projections is None:
        n_projections = final_boundary
    if n_projections != final_boundary:
        raise ValueError(
            "n_projections must equal the final min boundary in cycle.csv "
            f"({final_boundary})"
        )
    if n_projections <= start_index:
        raise ValueError("n_projections must be greater than start_index")
    if angle_span_deg is not None and angle_span_deg <= 0:
        raise ValueError("angle_span_deg must be positive")
    min_to_max = build_phase_order(
        n_phases,
        min_phase=min_phase,
        max_phase=max_phase,
        phase_labels=phase_labels,
    )
    max_to_min = tuple(reversed(min_to_max))

    result: list[ProjectionPhase] = []

    def append_segment(
        *,
        marker: CycleMarker,
        segment_start: int,
        segment_end: int,
        order: Sequence[int],
        branch: str,
        skip_start: bool,
    ) -> None:
        span = segment_end - segment_start
        first_index = segment_start + 1 if skip_start else segment_start
        for projection_index in range(first_index, segment_end + 1):
            fraction = 0.0 if span == 0 else (projection_index - segment_start) / span
            phase_position = _nearest_phase_position(fraction, len(order))
            result.append(
                ProjectionPhase(
                    projection_index=projection_index,
                    cycle=marker.cycle,
                    branch=branch,
                    phase_position=phase_position,
                    phase_label=order[phase_position],
                    phase_fraction=float(fraction),
                    angle_deg=(
                        None
                        if angle_span_deg is None
                        else float(projection_index * angle_span_deg / n_projections)
                    ),
                )
            )

    for marker_index, marker in enumerate(markers):
        append_segment(
            marker=marker,
            segment_start=marker.start_index,
            segment_end=marker.max_index,
            order=min_to_max,
            branch="min_to_max",
            skip_start=marker_index > 0,
        )
        append_segment(
            marker=marker,
            segment_start=marker.max_index,
            segment_end=marker.min_index,
            order=max_to_min,
            branch="max_to_min",
            skip_start=True,
        )

    return tuple(row for row in result if row.projection_index < n_projections)


def write_projection_phase_mapping(
    mapping: Iterable[ProjectionPhase],
    output_csv: str | Path,
) -> None:
    """Write a projection-to-phase mapping as a CSV file."""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "projection_index",
                "cycle",
                "branch",
                "phase_position",
                "phase_label",
                "phase_fraction",
                "angle_deg",
            ]
        )
        for row in mapping:
            writer.writerow(
                [
                    row.projection_index,
                    row.cycle,
                    row.branch,
                    row.phase_position,
                    row.phase_label,
                    f"{row.phase_fraction:.8f}",
                    "" if row.angle_deg is None else f"{row.angle_deg:.8f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-csv", default="data/cycle.csv")
    parser.add_argument("--output-csv", default="outputs/projection_phase_mapping.csv")
    parser.add_argument("--n-phases", type=int, required=True)
    parser.add_argument("--min-phase", type=int, required=True)
    parser.add_argument("--max-phase", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--n-projections",
        type=int,
        default=None,
        help="number of actual projections; defaults to the final min boundary",
    )
    parser.add_argument(
        "--angle-span-deg",
        type=float,
        default=300.0,
        help="scan angle span in degrees, using endpoint=False (default: 300)",
    )
    parser.add_argument(
        "--phase-index-base",
        type=int,
        choices=(0, 1),
        default=1,
        help="default phase labels are base-0 or base-1 (default: 1)",
    )
    args = parser.parse_args()

    labels = tuple(
        range(args.phase_index_base, args.phase_index_base + args.n_phases)
    )
    mapping = build_projection_phase_mapping(
        args.cycle_csv,
        args.n_phases,
        min_phase=args.min_phase,
        max_phase=args.max_phase,
        phase_labels=labels,
        start_index=args.start_index,
        n_projections=args.n_projections,
        angle_span_deg=args.angle_span_deg,
    )
    write_projection_phase_mapping(mapping, args.output_csv)
    print(f"wrote {len(mapping)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
