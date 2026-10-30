# Round contrast data generation

This is a copy of the Round contrast simulation pipeline. On the standalone
`get_data` branch, these files are at the repository root.

- `generate_round1_contrast_recon.py`: phase projections, full prior, and sparse FDK reconstructions.
- `generate_round1_contrast_gt.py`: matched per-phase GT without projection or FDK.
- `generate_pig_first_phase_prior.py`: shared attenuation calibration and reconstruction postprocessing.
- `utils/` and `moni/`: copied implementations used by the scripts above.
- `settings/`: snapshots of the six Round settings CSVs, `cycle.csv`, and `compare.csv`.

Run from the branch checkout root. CT volumes are not included; place them under
`data/pig/` or point `--round-root` to their location. Pass all paths explicitly
because the copied scripts retain defaults from their original workspace layout:

```powershell
python generate_round1_contrast_recon.py --round-root data/pig/AHFP-Round-2 --settings-csv settings/round2_contrast_recon_settings.csv --cycle-csv settings/cycle.csv --calibration-csv settings/compare.csv --output-root outputs/round2_contrast_recon --list
python generate_round1_contrast_gt.py --round-root data/pig/AHFP-Round-2 --settings-csv settings/round2_contrast_recon_settings.csv --calibration-csv settings/compare.csv --recon-root outputs/round2_contrast_recon --output-root outputs/round2_contrast_gt --list
```

Remove `--list` to generate data. Use the same settings and calibration CSVs for
both commands so the GT matches the sparse reconstruction. No CT volumes or
generated output images are included in this branch. In the original local
workspace, prefix script and settings paths in these examples with `get_data/`.
