"""Compare sample FDK against its generator and prepare evaluator-only references."""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpi_moco.config import read_json, write_json
from dpi_moco.prepare import central_roi, load_prepared, save_array


def load_legacy(path, grid):
    image = nib.load(str(path))
    if (tuple(grid["shape_zyx"]) != image.shape or
            not np.allclose(image.header.get_zooms()[:3], grid["spacing_xyz_mm"][::-1], atol=1e-6, rtol=0)):
        raise ValueError(f"Legacy volume grid dimensions do not match: {path}")
    return np.asarray(image.dataobj, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--recon", type=Path, required=True)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, ((root, case),) = load_prepared(args.prepared)
    grid = case["grid"]
    real, pseudo = (case["branches"][name] for name in ("real", "pseudo"))
    records = [("pi", case["pi"], "prior_image.nii.gz")]
    records.extend((f"x_{int(p['phase_id']):02d}", p["x"],
                    f"phase_{int(p['phase_id']) + 1:02d}_reconstruction.nii.gz") for p in real)
    results = []
    for kind, current, previous in records:
        a = central_roi(np.load(root / current, mmap_mode="r"))
        b = central_roi(load_legacy(args.recon / previous, grid))
        difference = np.subtract(a, b, dtype=np.float64)
        results.append({"volume": kind, "new_roi_mean": float(a.mean(dtype=np.float64)),
                        "legacy_roi_mean": float(b.mean(dtype=np.float64)),
                        "max_abs_roi_difference": float(np.abs(difference).max()),
                        "roi_rmse": float(np.sqrt(np.square(difference).mean()))})
    rows = case["valid_views"]
    for phases in (real, pseudo):
        assert sorted(view for phase in phases for view in phase["views"]) == rows
    assert [len(p["views"]) for p in real] == [20, 80, 80, 40, 80, 80, 40, 80, 80, 20]
    assert [len(p["views"]) for p in pseudo] == [60] * 10
    assert 30 in real[0]["views"] and 30 in pseudo[9]["views"]
    assert 31 in real[1]["views"] and 31 in pseudo[0]["views"]
    write_json(args.out / "fdk_comparison.json", {"config": config, "units": "mm^-1",
               "roi": "central third along each axis", "real_counts": [len(p["views"]) for p in real],
               "pseudo_counts": [len(p["views"]) for p in pseudo], "legacy_compare": results,
               "note": "The legacy generator masks the circular FOV after reconstruction. The central ROI lies inside that mask."})
    if args.gt:
        reconstruction_settings = read_json(args.recon / "reconstruction_config.json")["settings"]
        if read_json(args.gt / "gt_config.json")["settings"] != reconstruction_settings:
            raise ValueError("Legacy reference and projection reconstruction settings differ")
        if not np.allclose(np.asarray(grid["origin_xyz_mm"]),
                           -(np.asarray(grid["shape_zyx"][::-1]) - 1) * np.asarray(grid["spacing_xyz_mm"]) / 2):
            raise ValueError("Legacy reference adapter expects a centered scanner grid")
        phases = []
        for p in real:
            phase = p["phase_id"]
            source = args.gt / f"phase_{phase + 1:02d}_gt.nii.gz"
            path = f"phase_{phase:02d}_reference.npy"
            save_array(args.out / path, load_legacy(source, grid))
            phases.append({"phase_id": phase, "units": "mm^-1", "grid": grid, "path": path,
                           "source": str(source.resolve())})
        write_json(args.out / "reference.json", {"scans": [{"scan_id": case["scan_id"], "phases": phases}],
                   "reference_type": "legacy TIGRE full-view, single-phase FDK, centered array-axis grid"})


if __name__ == "__main__":
    main()
