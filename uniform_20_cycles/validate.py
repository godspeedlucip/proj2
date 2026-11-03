"""Check the saved schedule, projections, FDK volumes and matched GT grids."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile


def validate_result(
    mapping_csv: Path,
    recon_dir: Path,
    gt_dir: Path,
    *,
    n_projections: int = 600,
    n_cycles: int = 20,
    phases: tuple[int, ...] = tuple(range(1, 11)),
) -> dict:
    """Return an audit summary or fail on a missing, corrupt or unmatched output."""
    if n_projections < 1 or n_cycles < 1 or n_projections % n_cycles:
        raise ValueError("projection count must be divisible by cycle count")
    width = n_projections // n_cycles
    with Path(mapping_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != n_projections:
        raise ValueError(f"mapping contains {len(rows)} rows; expected {n_projections}")
    counts = Counter()
    for index, row in enumerate(rows):
        phase = int(row["phase_label"])
        if (int(row["projection_index"]) != index
                or int(row["cycle"]) != index // width + 1
                or phase not in phases):
            raise ValueError(f"invalid mapping at projection {index}")
        counts[phase] += 1
    if any(counts[phase] == 0 for phase in phases):
        raise ValueError("a CT phase received zero projections")

    projections = np.load(Path(recon_dir) / "projections.npy", mmap_mode="r",
                          allow_pickle=False)
    if projections.ndim != 3 or projections.shape[0] != n_projections:
        raise ValueError(f"invalid projection shape: {projections.shape}")
    for start in range(0, n_projections, 32):
        if not np.isfinite(projections[start:start + 32]).all():
            raise ValueError(f"nonfinite projection values near index {start}")

    def check_volume(directory: Path, stem: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        tif_path = Path(directory) / f"{stem}.tif"
        nii_path = Path(directory) / f"{stem}.nii.gz"
        if not tif_path.is_file() or not nii_path.is_file():
            raise FileNotFoundError(f"missing TIFF/NIfTI pair for {stem} in {directory}")
        image = nib.load(str(nii_path))
        tiff = tifffile.memmap(tif_path)
        shape = tuple(int(value) for value in image.shape)
        if tiff.shape != shape or not np.isfinite(tiff[shape[0] // 2]).all():
            raise ValueError(f"invalid TIFF shape or values for {stem}")
        return shape, tuple(float(value) for value in image.header.get_zooms()[:3])

    reference_shape, reference_spacing = check_volume(recon_dir, "prior_image")
    for phase in phases:
        recon_shape, recon_spacing = check_volume(recon_dir, f"phase_{phase:02d}_reconstruction")
        gt_shape, gt_spacing = check_volume(gt_dir, f"phase_{phase:02d}_gt")
        if recon_shape != reference_shape or gt_shape != reference_shape:
            raise ValueError(f"phase {phase}: reconstruction/GT shape differs from prior")
        if not np.allclose(recon_spacing, reference_spacing, rtol=0, atol=1e-5):
            raise ValueError(f"phase {phase}: reconstruction spacing differs from prior")
        if not np.allclose(gt_spacing, recon_spacing, rtol=0, atol=1e-5):
            raise ValueError(f"phase {phase}: GT spacing differs from reconstruction")
    return {
        "status": "passed",
        "n_projections": n_projections,
        "n_cycles": n_cycles,
        "projections_per_cycle": width,
        "projection_shape": list(projections.shape),
        "phase_view_counts": {str(phase): counts[phase] for phase in phases},
        "volume_shape": list(reference_shape),
        "voxel_spacing_mm": list(reference_spacing),
        "reconstruction_directory": str(Path(recon_dir).resolve()),
        "gt_directory": str(Path(gt_dir).resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    project = Path(__file__).resolve().parent
    series = Path("Control_1_7569") / "Heart_with_contrast_back"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", type=Path,
                        default=project / "settings" / "projection_phase_mapping.csv")
    parser.add_argument("--recon-dir", type=Path, default=project / "outputs" / "recon" / series)
    parser.add_argument("--gt-dir", type=Path, default=project / "outputs" / "gt" / series)
    parser.add_argument("--n-projections", type=int, default=600)
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--n-phases", type=int, default=10)
    parser.add_argument("--report", type=Path, default=project / "outputs" / "validation_report.json")
    args = parser.parse_args(argv)
    result = validate_result(
        args.mapping_csv, args.recon_dir, args.gt_dir,
        n_projections=args.n_projections, n_cycles=args.n_cycles,
        phases=tuple(range(1, args.n_phases + 1)),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validation passed: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
