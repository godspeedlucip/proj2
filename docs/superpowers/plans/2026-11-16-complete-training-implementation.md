# Complete RSTAR4D Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver both raw CT/TIGRE and paired-data entry points through training, resume, inference and evaluation, using public notebooks as the algorithm authority.

**Architecture:** A locally runnable and installable `rstar4d` package. Preserve published network state keys and loss/layout semantics; add explicit data contracts, deterministic sampling, atomic checkpoints and metadata. Vendor the reviewed pig geometry/reconstruction helpers with source fingerprints rather than importing an external utils package at runtime.

**Tech Stack:** Python 3.9+, PyTorch 2.3.1, TIGRE, numpy/scipy, h5py, nibabel, tifffile, PyYAML, pytorch-msssim 1.0.0; built-in unittest for tests.

**Spec:** `docs/superpowers/specs/2026-11-16-rstar4d-pig-tigre-training-design.md`, with the user's latest instruction overriding conflicting loss/schedule defaults.

## Global Constraints

- Public notebook authority: circular padding, exact keys/2,118,693 parameters, Stage I skip z, Stage II Dirac/zero initialization, L1+0.1 SSIM with original stage layouts, Adam1.5e-4, epochs60/30, milestones20/30/40 and10/20/25.
- Explicit mu/calibrated_mu scaling; official legacy data divide1000 profile remains available. No silent mixing of intensity domains.
- Preserve full T=10/11. Pig local blocks16×128×128,32×96×96,64×64×64 are explicit memory adaptations.
- Original notebooks/weights and external source/data remain untouched. All generated files stay in workspace unless user explicitly chooses otherwise.
- No dependency on an external utils package at runtime; version and content fingerprints accompany datasets/checkpoints.
- Existing intentionally different GT/CBCT configurations are outside implementation testing.

## Review Focus

- Partial/empty valid masks must not leak invalid voxels into losses, sampling, or metrics.
- Epoch-boundary resume must match uninterrupted CPU runs and reject changed data/config/split.
- Mixed T and odd spatial dimensions must survive HDF5 sampling and tiled prediction without temporal padding.
- Source offsets/calibration order must match the existing GT generator, with no second offset in reconstruction.
- Interrupted generation or training must not mark partial artifacts complete or overwrite an existing run silently.

## Task 1: Package, configuration, exact model and losses

Files: pyproject.toml, rstar4d/config.py, models/*, training/losses.py, utils/*, tests/test_model.py.
Interfaces: create_model(stage,profile), forward_volume(model,x), initialize_stage2(model,state), compute_loss(pred,target,valid,stage,config), load_config(path,kind).
- [x] RED: strict public weight loading, CPU notebook-forward equivalence at T10/11, stage initialization isolation, original SSIM equivalence, invalid gradient exclusion and unknown config rejection.
- [x] Implement with source attribution; public-code and pig-local profiles.
- [x] GREEN: `python -m unittest tests.test_model -v`.

## Task 2: Data contracts, inventory, paired import and sampling

Files: data/{contracts,inventory,paired,sampling}.py; tests/test_data.py.
Interfaces: read_manifest(path), validate_manifest(path,split), import_paired(manifest,out), inventory_pig(...), PatchDataset(records,config,stage,epoch,length,samples=None).
- [x] RED: animal split leakage, HDF5 metadata/hash mismatch, missing TIFF metadata, variable-T synchronized patches, deterministic draws and invalid/small-volume handling.
- [x] Implement streaming HDF5, explicit source map, raw inventory, fixed validation lists.
- [x] GREEN: `python -m unittest tests.test_data -v`.

## Task 3: TIGRE generation and source compatibility

Files: simulation/{reference,phase_mapping,backend,pipeline,phantom}.py; tests/test_simulation.py; configs/simulation/*.
Interfaces: simulate_manifest(raw_manifest,config,out), make_phantom(out,...), reconstruct(...), validate_settings(...).
- [x] RED: original phase mapping/counts, exact detector padding, HU validity, calibration order, reconstruction parity and no input mutation.
- [x] Implement sequential view-batched simulation, cached atomic case files, valid masks, metadata and optional shared Poisson noise.
- [x] GREEN: real CUDA small dynamic phantom and reference parity; no mock projection used as end-to-end proof.

## Task 4: Training, checkpoint and pipeline

Files: training/{checkpoint,trainer,pipeline}.py; tests/test_training.py.
Interfaces: train(config,manifest,split,run_dir,stage,resume=None,init=None,stop_after_epoch=None), pipeline(...), load_checkpoint(path).
- [x] RED: deterministic uninterrupted/resume equivalence, config/data mutation rejection, accumulation remainder, NaN handling, stage transfer and no double initialization.
- [x] Implement exact public scheduler/loss plus explicit alternative profiles, animal-aware validation, RNG/checkpoint provenance and JSONL/CSV logs.
- [x] GREEN: both stages learn a nonconstant fixed fixture; CPU resume equality.

## Task 5: Inference, evaluation, CLI and documentation

Files: inference/*, evaluation/*, cli.py, __main__.py, README-training.md, configs/*, environment*.yml; tests/test_inference.py, tests/test_cli.py.
Interfaces: predict_manifest(checkpoint,manifest,out,...), evaluate_predictions(...), main(argv=None).
- [x] RED: tiled identity no seams on odd shapes and mixed T; original-domain metrics/empty ROI; projection eligibility and exact self projection; CLI errors/outputs.
- [x] Implement every Spec command, paired and phantom examples, environment diagnostics, benchmark, fixed-window previews and exports.
- [x] GREEN: all CLI help and programmatic integration checks.

## Task 6: Acceptance, independent review and delivery

- [x] Execute raw phantom→TIGRE→train both stages→infer→evaluate, and independently import paired data→same pipeline.
- [x] Run acceptance suite, GPU benchmarks, local source compatibility and a bounded real-source probe.
- [x] Fresh reviewer checks implementation against Spec/user precedence; fix material findings with regression tests.
- [x] Record exact completed checks and remaining external-data/performance limitations; provide concrete commands.

Completion evidence and exact scope: docs/reviews/2026-11-16-implementation-acceptance.md.
