# TAM Sparse Temporal Supervision Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans and test-driven development.

**Goal** Train 3D nnU-Net with two annotated phases and one rotating unannotated phase per forward, and export every requested phase.

**Architecture** Reuse the existing paired preparation, trainer, TAM presets, and sliding-window predictor. Store configured context phases in the frozen plans. Keep the two supervised targets in separate label channels, and use the remaining input slot for an image-only phase.

**Tech stack** nnU-Net v2, PyTorch, NumPy, SimpleITK, pytest.

**Spec** [TAM C5 specification, section 2.5](../specs/2026-11-23-tam-c5-3d-nnunet.md)

## Acceptance checks

- [x] Add and run an end-to-end failing test in `test_paired_tam_training.py` with ten CTs and only two label files per case. Assert prepared image and target shapes, actual forward with three inputs, checkpoint reload, and ten native-grid predictions.
- [x] Extend `paired_tam_contract.py`, `preprocessing/paired_tam.py` and `run/paired_tam.py` to accept named context phases, freeze them in the plans, and save their source provenance and image channels.
- [x] Extend `paired_tam.py` in network architecture and `nnUNetTrainerTAM.py` to accept a three-slot model for sparse plans. Crop two supervised slots and one sampled context slot at the same position. Apply deep-supervision loss to the two annotated slots only. Keep online validation on annotated targets.
- [x] Extend the paired predictor to process each context CT in a separate shared-anchor pass, validate destinations, and export in each phase's physical grid. Keep reported evaluation metrics on the two annotated phases.
- [x] Document the command and manifest, then run the repository test suite. Check both the existing two-phase control and the new checkpoint that consumes three inputs.
