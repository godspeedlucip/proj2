"""Run the uniform 20-cycle phase simulation and matched GT pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from uniform_recon import (
    DEFAULT_ROUND_ROOT, DEFAULT_SETTINGS_CSV, discover_series,
    load_series_settings, select_series,
)
from uniform_cycle import prepare_schedule


PROJECT_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "list", "recon", "gt", "all"))
    parser.add_argument("--round-root", type=Path, default=DEFAULT_ROUND_ROOT)
    parser.add_argument("--settings-csv", type=Path, default=DEFAULT_SETTINGS_CSV)
    parser.add_argument("--calibration-csv", type=Path, default=PROJECT_ROOT / "settings" / "compare.csv")
    parser.add_argument("--schedule-dir", type=Path, default=PROJECT_ROOT / "settings")
    parser.add_argument("--recon-root", type=Path, default=PROJECT_ROOT / "outputs" / "recon")
    parser.add_argument("--gt-root", type=Path, default=PROJECT_ROOT / "outputs" / "gt")
    parser.add_argument("--n-phases", type=int, default=10)
    parser.add_argument("--series", nargs="+", default=["Control_1_7569/Heart_with_contrast_back"])
    parser.add_argument("--all-series", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.round_root = args.round_root.resolve()
    args.settings_csv = args.settings_csv.resolve()
    args.calibration_csv = args.calibration_csv.resolve()
    args.schedule_dir = args.schedule_dir.resolve()
    args.recon_root = args.recon_root.resolve()
    args.gt_root = args.gt_root.resolve()

    cycle_csv, mapping_csv = prepare_schedule(args.schedule_dir, n_phases=args.n_phases,
                                              max_phase=args.n_phases)
    print(f"Schedule: {cycle_csv}\nPhase mapping: {mapping_csv}", flush=True)
    if args.mode == "prepare":
        return 0

    round_root = args.round_root.resolve()
    selectors = None if args.all_series else args.series
    selected = select_series(discover_series(round_root), round_root, selectors)
    if not selected:
        parser.error(f"No contrast CT series found under {round_root}")
    settings = load_series_settings(args.settings_csv)
    expected_phases = tuple(range(1, args.n_phases + 1))
    for directory, phase_paths in selected:
        name = directory.relative_to(round_root).as_posix()
        if tuple(phase_paths) != expected_phases:
            parser.error(f"{name}: expected CT phases {expected_phases}, found {tuple(phase_paths)}")
        if name not in settings:
            parser.error(f"No settings row for {name}")
        row = settings[name]
        if (row.n_projections, row.angle_span_deg, row.min_phase, row.max_phase) != (600, 300.0, 1, args.n_phases):
            parser.error(f"{name}: settings must specify 600 views, 300 degrees and phases 1..{args.n_phases}")

    common = ["--round-root", str(round_root), "--settings-csv", str(args.settings_csv),
              "--calibration-csv", str(args.calibration_csv)]
    if not args.all_series:
        common += ["--series", *args.series]
    if args.overwrite:
        common += ["--overwrite"]

    if args.mode in ("list", "recon", "all"):
        if args.mode != "list" and importlib.util.find_spec("tigre") is None:
            parser.error("TIGRE is required for forward projection and FDK; use a Python/CUDA environment with tigre installed")
        command = [sys.executable, str(PROJECT_ROOT / "uniform_recon.py"),
                   *common, "--cycle-csv", str(cycle_csv), "--output-root", str(args.recon_root)]
        if args.mode == "list":
            command.append("--list")
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode or args.mode == "list":
            return result.returncode

    if args.mode in ("gt", "all"):
        command = [sys.executable, str(PROJECT_ROOT / "uniform_gt.py"),
                   *common, "--recon-root", str(args.recon_root), "--output-root", str(args.gt_root)]
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
