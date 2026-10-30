# Round contrast data generation

This directory is a copy of the Round contrast simulation pipeline. The original
scripts and CSV files remain at the project root and under `data/`.

- `generate_round1_contrast_recon.py`: phase projections, full prior, and sparse FDK reconstructions.
- `generate_round1_contrast_gt.py`: matched per-phase GT without projection or FDK.
- `generate_pig_first_phase_prior.py`: shared attenuation calibration and reconstruction postprocessing.
- `utils/` and `moni/`: copied implementations used by the scripts above.
- `settings/`: snapshots of the six Round settings CSVs, `cycle.csv`, and `compare.csv`.

Run from the project root. By default, the scripts read the project's existing
`data/` directory and write to its `outputs/` directory; they also use the
original Round 1 settings, cycle, and calibration files. Pass the copied CSVs
explicitly when using this bundle's snapshots:

```powershell
python get_data/generate_round1_contrast_recon.py --round-root data/pig/AHFP-Round-2 --settings-csv get_data/settings/round2_contrast_recon_settings.csv --cycle-csv get_data/settings/cycle.csv --calibration-csv get_data/settings/compare.csv --output-root outputs/round2_contrast_recon --list
python get_data/generate_round1_contrast_gt.py --round-root data/pig/AHFP-Round-2 --settings-csv get_data/settings/round2_contrast_recon_settings.csv --calibration-csv get_data/settings/compare.csv --recon-root outputs/round2_contrast_recon --output-root outputs/round2_contrast_gt --list
```

Remove `--list` to generate data. Use the same settings and calibration CSVs for
both commands so the GT matches the sparse reconstruction. No CT volumes or
generated output images are duplicated in this folder.
