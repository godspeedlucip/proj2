# PriorNet Heart Mask Finetuning Implementation Plan

**Implementation status (2026-11-04):** Tasks 1–4 have been implemented on `prior_with_mask`. The focused suite passes (39 tests), and a synthetic CPU run produced separate base and finetune checkpoints. Reviewer findings were addressed with case-level validation, exact ROI MAE, deterministic stage-B seeding, checkpoint provenance, SophiaG curvature updates, and richer logs. Real-data effectiveness and anatomical alignment remain unverified until the user's data is available.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible two-stage PriorNet training path that uses optional heart masks for quality-aware region and boundary losses, while preserving mask-free inference.

**Architecture:** Keep `Dense2D(input, prior)` unchanged. Extend the NPZ reader to expose masks only in masked mode; put differentiable heart-region and exact-boundary losses in a focused module; add a separate training entry point for the 100-loop base stage and up-to-50-loop masked stage with explicit checkpoints, ablation modes, deterministic validation, and metrics.

**Tech Stack:** Python, NumPy, PyTorch, existing `SophiaG`, pytest. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-11-03-heart-mask-quality-aware-finetuning-design.md`

## Global Constraints

- Present data: roughly 20% of NPZ samples have `mask`; `mask_quality` is optional (`1=exact`, `0=rough`, absent means unknown/rough treatment). Only verified, aligned masks enter stage B.
- Stage A: 100 training loops with full-image MSE and the existing 10% random subset per loop. Stage B: at most 50 complete passes over the masked training list, starting from the same stage-A model checkpoint for every ablation.
- Stage B always keeps full-image MSE. `full`, `roi`, `boundary`, and `gradient` ablation modes add terms in that order. Only exact masks create boundary bands; rough and unknown masks receive a softened ROI term.
- Initial settings: `r=3` pixels, `lambda_exact=1.0`, `lambda_rough=0.5`, `lambda_boundary=1.0`; `lambda_gradient=0` until its scale is measured, then configure it explicitly. Stage-A learning rate `1e-4`, stage-B starting point `1e-5`; both configurable.
- Inference remains `Dense2D(input, prior)`; neither mask nor segmentation output is required at inference.
- No actual NPZ data or model checkpoint is present in this workspace. Unit/integration checks use synthetic samples; real-data alignment and model-quality claims require user data later.
- Keep the legacy S3 script intact. Its unused `network_discriminator` import and hard-coded real-data path make it an unsuitable foundation for the new entry point.

## File Map

| File | Responsibility |
| --- | --- |
| `utils/read_2d_npz_wy.py` | Backward-compatible reader, optional mask/quality contract, deterministic validation loading. |
| `utils/heart_mask_loss.py` | Region, exact boundary-band, and Sobel gradient reconstruction losses. |
| `train_prior_with_mask.py` | CLI, two-stage loop, sampling, checkpoints, validation, metrics, ablations, early stop. |
| `tests/test_heart_mask_data.py` | NPZ reader and data-contract tests. |
| `tests/test_heart_mask_loss.py` | Loss geometry, quality gating, numeric and gradient tests. |
| `tests/test_prior_with_mask_training.py` | Small-model stage routing, checkpoint and validation tests. |
| `docs/heart-mask-training.md` | Input format, command examples, audit steps, and experiment matrix. |

## Review Focus

1. **No newline at the end of a list file:** path must remain intact. Test in Task 1.
2. **Missing or unknown mask quality:** treat as rough; never apply exact-boundary terms. Tests in Tasks 1 and 2.
3. **Malformed or mis-sized mask:** fail with the NPZ path instead of silently training with a wrong mask. Test in Task 1; visual alignment remains a required manual check.
4. **Batch with no exact samples:** boundary and gradient contributions must be finite zero, with no division by zero. Test in Task 2.
5. **Stage/checkpoint confusion:** every ablation must start from the same stage-A weights with a fresh stage-B optimizer; saved metadata must identify stage and mode. Test in Task 3.

---

### Task 1: Read partial mask annotations safely

**Files:**
- Modify: `utils/read_2d_npz_wy.py:105`
- Create: `tests/test_heart_mask_data.py`

**Interfaces:**
- Preserve `npz_reader_2d_data_with_prior(data_with_prior_txt)` for the legacy three-field caller.
- Extend constructor to `npz_reader_2d_data_with_prior(data_with_prior_txt: str, *, training: bool = True, require_mask: bool = False)`.
- When `require_mask=True`, `__getitem__` returns `input`, `label`, `prior`, `mask` as `float32` tensors shaped `(1,H,W)` and `mask_quality` as scalar integer tensor (`1` exact, `0` rough, `-1` unknown). When false, retain the three-field return shape.
- Masked UTF-8 list rows may optionally append a tab and `0` or `1` to supply quality when it is absent from NPZ; conflicting NPZ/list values fail with the sample path.
- Validation mode (`training=False`) uses exactly `*50`, with no random `0.9..1.1` scaling. Mask values remain 0/1 in every mode.
- `require_mask=True` rejects missing, empty, all-one, nonbinary, nonfinite or shape-mismatched masks with the sample path in the error. It does not infer spatial alignment from shape.

- [ ] **Step 1: Write failing reader tests.** Make temporary NPZ files and list files; assert legacy keys, no-newline path loading, exact/rough/unknown quality mapping, unchanged mask under training scale, deterministic validation reads, and path-bearing errors for malformed masks.
- [ ] **Step 2: Run** `python -m pytest tests/test_heart_mask_data.py -q` **and confirm failure.**
- [ ] **Step 3: Implement the constructor and reader behavior.** Use `line.strip()` and an NPZ context manager. Keep the same image negative-value clipping. Validate masked data before tensor conversion.
- [ ] **Step 4: Run** `python -m pytest tests/test_heart_mask_data.py -q` **and confirm pass.**
- [ ] **Step 5: Commit** the reader and its tests.

### Task 2: Implement quality-aware reconstruction loss

**Files:**
- Create: `utils/heart_mask_loss.py`
- Create: `tests/test_heart_mask_loss.py`

**Interfaces:**
- `heart_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, quality: torch.Tensor, *, mode: str, radius: int, lambda_exact: float, lambda_rough: float, lambda_boundary: float, lambda_gradient: float, rough_sigma: float = 3.0) -> dict[str, torch.Tensor]`.
- Accept `[N,1,H,W]` images/masks; return scalar tensor entries `total`, `full`, `roi_exact`, `roi_rough`, `boundary`, `gradient` for logging and backpropagation.
- Modes: `full` = full MSE only; `roi` adds exact and rough ROI MSE; `boundary` adds exact boundary-band MSE; `gradient` adds exact boundary-band Sobel-L1. Mode selection never changes the input batch.
- Compute `B = dilate(M,r) - erode(M,r)` for exact masks with zero outside image bounds. Normalize each region/band term by its own pixel-weight sum per sample, then aggregate over the **whole batch** so varying quality mix cannot silently change term scale. For rough/unknown masks, blur the ROI and never derive a mask-contour band. Return finite zero for absent quality groups.

- [ ] **Step 1: Write failing loss tests.** Assert zero when `pred==target`, numerically correct normalized error on a tiny synthetic mask, no boundary/gradient contribution for rough or unknown masks, finite zero when there is no exact sample, and finite nonzero `pred.grad` near an exact boundary. Verify an error outside the band affects only full MSE.
- [ ] **Step 2: Run** `python -m pytest tests/test_heart_mask_loss.py -q` **and confirm failure.**
- [ ] **Step 3: Implement the loss module.** Use fixed horizontal/vertical Sobel kernels on both images. Compare image gradients inside `B`; do not compare image gradients to the binary mask edge. Validate `mode`, shape, radius, and nonnegative coefficients.
- [ ] **Step 4: Run** `python -m pytest tests/test_heart_mask_loss.py -q` **and confirm pass.**
- [ ] **Step 5: Commit** the loss module and tests.

### Task 3: Train both stages with explicit checkpoint behavior

**Files:**
- Create: `train_prior_with_mask.py`
- Create: `tests/test_prior_with_mask_training.py`

**Interfaces:**
- CLI options: `--stage {base,finetune,both}`, `--base-train-list`, `--base-valid-list`, `--masked-train-list`, `--masked-valid-list`, `--base-checkpoint`, `--output-dir`, `--device`, `--seed`, `--batch-size`, `--num-workers`, `--base-epochs` (default 100), `--finetune-epochs` (default 50), `--base-lr`, `--finetune-lr`, `--loss-mode {full,roi,boundary,gradient}`, all loss coefficients, and `--boundary-radius`.
- Expose `train_epoch(model, loader, optimizer, *, stage, loss_config, device) -> dict[str,float]` and `validate_epoch(model, loader, *, stage, loss_config, device) -> dict[str,float]` so a tiny injected model can test the loop without running `Dense2D`.
- Stage A uses a seeded random `max(1, len(train_dataset)//10)` subset per loop and full MSE, ignoring mask. Stage B uses every sample in the masked list each loop and the selected loss mode. Both use `SophiaG`; initialize a fresh optimizer/scheduler at the stage-B boundary.
- A checkpoint stores `state_dict`, `optimizer_state_dict`, `lr_scheduler`, `epoch`, `stage`, `loss_mode`, `config`, and validation metrics. Load an old `state_dict` checkpoint as stage-B initialization without depending on output-directory existence. Write `base/` and `finetune/<loss-mode>/` separately.
- Validate stage B on **both** lists: `base-valid-list` for full-population full-image MSE/MAE, and `masked-valid-list` for ROI and exact-band MSE/MAE/gradient MAE. Both reads are deterministic. Save best stage-B checkpoint by exact-band MSE when exact validation masks exist, otherwise by ROI MSE and mark boundary success as unassessed. Stop after configurable patience (default 8) without improvement or 50 loops; compare full-population MSE against the plain-MSE ablation before accepting a model.

- [ ] **Step 1: Write failing small-model tests.** Use a two-input dummy model and synthetic loaders to verify stage A ignores masks, stage B rejects unmasked samples, identical stage-A model initialization across ablation modes, stage-B optimizer reset, separate full-population and masked validation metrics, deterministic validation, checkpoint metadata, and best-checkpoint selection.
- [ ] **Step 2: Run** `python -m pytest tests/test_prior_with_mask_training.py -q` **and confirm failure.**
- [ ] **Step 3: Implement the CLI and stage runner.** Keep CPU support for tests and CUDA selection for real training. Do not invoke the legacy `test_real` function or overwrite original checkpoints.
- [ ] **Step 4: Run** `python -m pytest tests/test_prior_with_mask_training.py -q` **and confirm pass.**
- [ ] **Step 5: Run** `python train_prior_with_mask.py --help` **and confirm the two-stage and ablation arguments appear.**
- [ ] **Step 6: Commit** the new runner and tests.

### Task 4: Document data audit and experiment execution

**Files:**
- Create: `docs/heart-mask-training.md`
- Modify: `docs/superpowers/specs/2026-11-03-heart-mask-quality-aware-finetuning-design.md` only if implementation decisions differ from the draft

**Interfaces:**
- Document valid NPZ keys, quality codes, list files, aligned-mask requirement, stage-A exposure semantics, stage-B command examples, and the A/B/C/D/E comparison with the same base checkpoint.
- Include a preflight checklist reporting counts by independent patient, phase and mask quality; overlay inspection against `label`; disjoint patient IDs across train/validation. The program can check shape and values, but cannot prove anatomical alignment from arrays alone.
- State that no real-data training or boundary-quality claim is possible until NPZ lists, a base checkpoint or full training data, and independent exact-mask validation cases are available.

- [ ] **Step 1: Write the usage and audit document.** Include one command for base training, one for finetuning from an existing checkpoint, and the four ablation modes.
- [ ] **Step 2: Review the document against the spec and the actual CLI** for exact argument names and data assumptions.
- [ ] **Step 3: Run** `python -m pytest tests/test_heart_mask_data.py tests/test_heart_mask_loss.py tests/test_prior_with_mask_training.py -q` **and confirm pass.**
- [ ] **Step 4: Commit** documentation and any necessary spec correction.

## Final Verification

- Run all three new test files and `python train_prior_with_mask.py --help`.
- Use one tiny synthetic NPZ train/validation split for a CPU smoke run covering stage A and stage B, and verify separate checkpoint directories and logged metrics.
- Inspect the diff to confirm `PriorNet_pytorch.py` and the legacy S3 script did not change, and that no generated NPZ/checkpoint/cache file is staged.
- Report that clinical/image-quality efficacy remains untested without the user's data.
