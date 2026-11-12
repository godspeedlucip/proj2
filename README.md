# Round contrast data generation

This is a copy of the Round contrast simulation pipeline. On the standalone
`get_data` branch, these files are at the repository root.

- `generate_round1_contrast_recon.py`: phase projections, full prior, and sparse FDK reconstructions.
- `generate_round1_contrast_gt.py`: matched per-phase GT from full-angle projection and FDK of each phase CT.
- `generate_round1_contrast_mask.py`: annotated phase segmentation masks mapped to the GT/reconstruction grid.
- `generate_pig_first_phase_prior.py`: shared HU-to-attenuation conversion and reconstruction postprocessing.
- `series_calibration.py`: per-series calibration lookup and post-FDK polynomial application.
- `utils/` and `moni/`: copied implementations used by the scripts above.
- `settings/`: Round settings, cycle CSVs, and Round 1–5 heart calibration CSVs.

Run from the branch checkout root. CT volumes are not included; place them under
`data/pig/` or point `--round-root` to their location. Pass all paths explicitly. Reconstruction and GT require either
`--calibration-csv` or `--no-calibration`. Each Round has its own calibration
CSV; the `series` column selects a separate polynomial for each control/stent
series.

```powershell
python generate_round1_contrast_recon.py --round-root data/pig/AHFP-Round-2 --settings-csv settings/heart_only/round2_heart_gt_settings.csv --cycle-csv settings/heart_only/cycle.csv --calibration-csv settings/heart_only/round2_heart_calibration.csv --output-root outputs/round2_contrast_recon --list
python generate_round1_contrast_gt.py --round-root data/pig/AHFP-Round-2 --settings-csv settings/heart_only/round2_heart_gt_settings.csv --calibration-csv settings/heart_only/round2_heart_calibration.csv --recon-root outputs/round2_contrast_recon --output-root outputs/round2_contrast_gt --list
python generate_round1_contrast_mask.py --round-root data/pig/AHFP-Round-2 --settings-csv settings/heart_only/round2_heart_gt_settings.csv --recon-root outputs/round2_contrast_recon --gt-root outputs/round2_contrast_gt --output-root outputs/round2_contrast_mask --list
```

Remove `--list` to generate data. Use the same settings and calibration CSVs for
both commands so the GT matches the sparse reconstruction grid. Each GT uses
all scan angles from its own phase, while the prior uses the phase-mixed scan
and each sparse reconstruction uses only its assigned phase angles. TIGRE and
a CUDA GPU are required for both generation commands. No CT volumes or
generated output images are included in this branch. In the original local
workspace, prefix script and settings paths in these examples with `get_data/`.

The optional `fov_source_pixels` CSV column sets the transverse physical FOV
in CSV-adjusted source voxel counts while `fov_output_pixels` keeps the output
matrix size fixed. For example, Round 5 Control 1 uses `300` source pixels on
a `256×256` output grid. Set `0` or omit the column for the original crop.
The requested FOV must fit within the unpadded detector width. Run the
reconstruction and GT with the same settings CSV so their grids match.

Source CT voxels equal to exactly `-2048 HU` are treated as missing data. They
are excluded from HU-to-attenuation conversion and are zero in the volumes
passed to TIGRE. Valid CT voxels are converted to attenuation before projection.
The calibration polynomial is applied to the reconstructed prior, sparse images,
and GT after FDK, while exact-zero background remains zero. With
`--no-calibration`, no polynomial adjustment is applied. GT reconstruction is
zeroed outside the valid source sampling region. Each `phase_XX_gt` has a matching
`phase_XX_gt_valid_mask.tif` and `.nii.gz`: value 1 means every interpolation
source was valid and the voxel lies inside the output FOV; value 0 means it
must be excluded from supervised loss or image metrics. Both files share the
same shape and voxel spacing.

When settings or calibration change, regenerate reconstruction before GT.
The GT configuration signature invalidates older direct-sampling GT outputs.

For segmentation labels, remove `--list` from the mask command. Only CT phases
with a matching `IMxxxxx_mask.nii.gz` are processed. The script preserves
integer class IDs, checks source CT/mask alignment and any existing GT or
reconstruction grid, and writes `phase_XX_mask.tif/.nii.gz`. Its default
resampling is linear interpolation of each foreground class followed by a
`0.5` score threshold; `--mask-resampling nearest` avoids interpolation, and
`polynomial`/`cubic` select cubic B-spline interpolation. This label threshold
does not affect the CT attenuation calibration described above.
