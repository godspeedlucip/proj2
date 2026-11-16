# BIRD Reproduction Verification Plan

**Goal:** Audit the current implementation against the supplied journal paper and establish executable evidence for the full network and complete reconstruction pipeline.

**Scope:** The user's two GT configurations are intentional and are not findings, blockers, or targets for modification. No original data or study script is changed. Verification cannot establish paper-table reproduction without the authors' experimental data and a completed optimization.

- [x] Read journal equations 8–27, architecture figures and implementation section; request one independent read-only review.
- [x] Run existing tests and add independent DFT, general affine Jacobian and actual-network accumulated-gradient checks.
- [x] Run unmodified paper.json on CUDA: full network, 2048 rays × 512 samples and each regularizer on 262144 coordinates; save, resume and infer.
- [x] Run 16-level TIGRE model through a complete 20-step validation schedule, including midpoint resume and image inference.
- [x] Compare initial/trained projections on identical fixed inputs; measure execution chunk alternative without changing mathematical settings.
- [x] Save auditable reports, formula mapping, code fingerprints, runnable commands and explicit limits; package current code and small reports.

Artifacts: tests/test_paper_oracles.py; configs/verify_tigre.json; configs/paper_cuda_chunks.json; runs/verification_20261116/; REPRODUCTION_AUDIT.md. Runtime and quality claims must follow actual outputs.
