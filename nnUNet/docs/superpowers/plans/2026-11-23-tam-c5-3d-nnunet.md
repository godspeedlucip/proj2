# TAM C5 3D nnU-Net Implementation Plan

**Goal** Implement and verify paired phase1 and phase5 segmentation with C1, C3, C4 and C5.

**Architecture** Reuse the PlainConvUNet, nnU-Net plans, augmentation and Dice + CE. Keep paired CT preparation, stage attention and native-geometry prediction behind dedicated entry points.

**Spec** [TAM C5 3D nnU-Net](../specs/2026-11-23-tam-c5-3d-nnunet.md)

## Constraints

Use one CT channel, mutually exclusive labels, 3d_fullres and one device per training run. Keep one frozen plans and patient partition per fold. Run the real CT experiment once its manifest, labels and patient assignments are available.

## Tasks

1. Verify then implement the paired CT manifest, fold preparation, common crop and native geometry metadata. Exercise leakage, invalid geometry and train-only statistics.
2. Verify then implement TAM3D and the paired PlainConvUNet builder. Exercise stage selection, deep supervision, gradients, case isolation and C1 parity.
3. Verify then implement paired sliding windows and per-phase native export. Exercise shared tile coordinates, mirroring, overlapping logits and different image grids.
4. Integrate the dedicated trainers and experiment CLI with the preprocessing, network and prediction interfaces. Verify synchronized augmentation, phasewise loss, checkpoint compatibility, native Dice and HD95, and normal nnU-Net network behavior. Run a synthetic train, reload, validate and predict workflow across all presets and document the real-data commands.
