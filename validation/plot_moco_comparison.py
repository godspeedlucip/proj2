"""Render four matched Stage A methods from a validation scoresheet."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import h5py
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from rstar4d.moco.data import artifact_rows
from rstar4d.utils.io import read_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from audit_panel_alignment import require_matplotlib_panel_alignment

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    score = read_json(args.scores)
    keys = ("b", "x", "zero", "learned")
    titles = ("Frozen RSTAR", "Sparse FDK", "Zero-field average", "Learned-field fusion")
    with ExitStack() as stack:
        arrays = []
        labels = None
        for key in keys:
            item = artifact_rows(score["groups"][key]["path"])[args.case]
            if sha256(item["path"]) != score["groups"][key]["artifact_sha256"][args.case]:
                raise ValueError(f"Image digest disagrees with scoresheet for {key}")
            f = stack.enter_context(h5py.File(item["path"], "r"))
            meta = json.loads(f.attrs["metadata_json"])
            if labels is not None and labels != meta["phase_labels"]:
                raise ValueError("Phase order differs across methods")
            labels = meta["phase_labels"]
            arrays.append(f["prediction"])
        sample_indices = sorted({0, len(labels) // 2, len(labels) - 1})
        z = arrays[0].shape[1] // 2
        spacing = meta["geometry"]["voxel_spacing_zyx_mm"]
        fig, axes = plt.subplots(len(sample_indices), 4, figsize=(7.05, 5.86), squeeze=False)
        fig.subplots_adjust(left=.115, right=.98, top=.84, bottom=.145, wspace=.060, hspace=.075)
        fig.text(.115, .955, args.title, fontsize=12, weight="semibold", va="top")
        fig.text(.115, .904, f"Axial midplane, Z index {z}. Three ordered phases shown.", color="#53606b", fontsize=8)
        for r, time_index in enumerate(sample_indices):
            for c, (data, title) in enumerate(zip(arrays, titles)):
                axis = axes[r, c]
                plotted = axis.imshow(data[time_index, z], cmap="gray", vmin=0., vmax=.05,
                                      interpolation="nearest", aspect=spacing[1] / spacing[2])
                axis.set_axis_off()
                if r == 0:
                    axis.set_title(title, fontsize=8, pad=6, weight="semibold" if c == 3 else "normal")
                if c == 0:
                    axis.text(-.13, .5, f"Phase {labels[time_index]}", ha="right", va="center",
                              transform=axis.transAxes, color="#33414c")
        bar_axis = fig.add_axes([.36, .092, .36, .026])
        scale = fig.colorbar(plotted, cax=bar_axis, orientation="horizontal",
                             ticks=[0., .025, .05], format="%.3f")
        scale.set_label("Linear attenuation (mm⁻¹)", labelpad=3)
        scale.outline.set_linewidth(.5)
        require_matplotlib_panel_alignment(fig, axes=list(axes.flat),
                                           json_out=args.out.with_suffix(".alignment.json"), strict=True)
        fig.savefig(args.out.with_suffix(".pdf"))
        fig.savefig(args.out.with_suffix(".svg"))
        fig.savefig(args.out.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(args.out.with_suffix(".png"), dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    main()
