# Final implementation review

Date: 2026-11-16. Reviewer: fresh-context `implementation_final_review` agent.

Verdict: **Changes required.** The implementation provides both major entry paths and a coherent public-code model/training/inference chain, but the six Important findings below prevent accepting the full data-integrity, sampling, resume, and interruption contracts. No Critical finding was identified.

## Scope and evidence

Reviewed the complete new package interfaces and implementation in data, simulation, training, inference, evaluation, CLI/configuration, associated tests/documentation, the implementation plan, Spec, and execution ledger. Public code takes precedence over older conflicting Spec defaults. Historical intentionally different GT/CBCT products are excluded.

All file SHA256 values listed in `docs/reviews/implementation-review-package.md` matched the files reviewed. The supplied `validation/implementation-tests.log` records 24 passing tests, including CUDA integration; I did not rerun that suite or use the GPU. Focused review reproductions ran with `D:/python/miniconda/env/env/common/python.exe -B -`, using in-memory arrays and mocked filesystem boundaries without modifying implementation or data.

## Critical

None identified.

## Important

### I1 — Paired import accepts contradictory temporal and calibration contracts

Location: `rstar4d/data/paired.py:80` (comparison field list), `rstar4d/data/paired.py:85`; related `rstar4d/data/contracts.py:28` (`validate_metadata`).

Trigger: supply equal-shape CBCT/target with source metadata whose target `phase_labels` are reversed, or whose post-FDK calibration polynomial differs, while the six currently compared fields agree. The importer ignores both differences and writes the common metadata over the imported arrays. It likewise does not compare source-grid/settings fields such as source spacing, voxel multiplier, padding, filter/version and validity provenance where these are outside `geometry`. `validate_metadata` accepts an empty geometry or the example's placeholder note and does not require meaningful calibration coefficients/provenance, so absent contracts do not compensate for the omitted comparisons.

Effect: a temporally mispaired or differently calibrated training pair is accepted and subsequently passes prepared-data validation. This is a newly supplied incompatible pair, not the out-of-scope historical products. It violates Spec §§5,8 and A09.

Evidence: an in-memory importer reproduction changed the target `phase_labels` from `[1,2]` to `[2,1]` and coefficients from `[1,0]` to `[2,0]`; the importer reached `write_case` with no error.

Fix: define and require a normalized, complete common-grid/intensity contract; compare per-source phase order, calibration coefficients/application/provenance, and the source/reconstruction settings required by the Spec before reading/writing the pair. Require actual spatial mapping fields rather than arbitrary nonempty JSON. Explicitly distinguish shared fields from source-specific provenance, and preserve the latter. Add rejection tests for reversed phases, differing coefficients and incomplete geometry in addition to spacing mismatch.

### I2 — Resume bypasses output-run ownership checks

Location: `rstar4d/training/trainer.py:83`, `rstar4d/training/trainer.py:125`, `rstar4d/training/trainer.py:128`.

Trigger: `train --resume runs/A/checkpoints/last.pt --run-dir runs/B`, where B already contains an unrelated run. The nonempty-directory guard is skipped whenever `resume` is truthy. Signature validation checks A's checkpoint against the provided data/config, not against B's ownership. Metadata in B is then overwritten; later checkpoint saves replace B's best/last and logs are appended to the unrelated history.

Effect: silent destruction/mixing of an existing run, contrary to the explicit review focus and Spec §11. Resuming into a new empty directory also returns paths to nonexistent best/last if the checkpoint is already complete; if not complete and validation never improves on the saved best, the new directory never receives `best.pt`.

Evidence: direct control-flow inspection; no destructive reproduction was performed.

Fix: persist a run identity and require the destination to belong to the resumed run; reject unrelated nonempty destinations before any writes. Support migration by verifying/copying the required checkpoint lineage and historical best, or explicitly require migration of the whole run directory. Do not return success paths unless their checkpoint artifacts exist and have been verified. Test foreign nonempty destinations and completed-checkpoint migration.

### I3 — Manifest reordering changes resumed draws without changing the resume signature

Location: `rstar4d/data/contracts.py:152`, `rstar4d/data/sampling.py:17`, `rstar4d/data/sampling.py:31`.

Trigger: two training cases belong to the same animal; reorder their manifest rows between completed epochs. `snapshot` sorts cases by case ID, so the resume signature remains unchanged. `PatchDataset` builds each animal's case-index list in input order, so the same random integer now selects a different case.

Effect: an accepted epoch-boundary resume no longer matches the uninterrupted run, despite unchanged case contents and split. This violates Spec §11 and the stated deterministic sampling contract.

Evidence: a two-case in-memory reproduction printed `same_signature=True`. Original eight draws were `b,a,b,b,a,a,a,b`; reordered draws were `a,b,a,a,b,b,b,a` with identical seed/epoch/draw IDs.

Fix: canonicalize record/case ordering before building the sampler, or include the effective sampling order in the signed checkpoint contract and reject changes. Prefer canonical ordering so harmless manifest relocation/reordering preserves reproducibility. Add a resume-equivalence test with at least two cases per animal and reordered manifest lines.

### I4 — Validity retries silently discard difficult animals and alter patch probabilities

Location: `rstar4d/data/sampling.py:39`–`40`.

Trigger: a selected case/shape fails `min_valid_fraction`. Each retry calls `proposal` again, redrawing the animal, case and patch shape instead of retrying spatial origins for the selected unit. Thus a case or shape with no qualifying patch can be bypassed indefinitely as long as another one succeeds within 32 attempts.

Effect: effective animal/case/shape probabilities are weighted by validity acceptance rate. Difficult animals and larger patches can disappear from training without a failure. This contradicts Spec §8's animal-uniform sampling, equal shape probabilities, and explicit prohibition on silently changing patch probabilities or skipping cases.

Evidence: a CPU in-memory fixture with two animals, one whose only configured patch has validity 1/16 and one fully valid, produced 100/100 accepted samples from the fully valid animal instead of surfacing the impossible selected sample.

Fix: select animal/case/patch shape once per draw, deterministically retry only its spatial origin up to `max_attempts`, and raise an error naming the case and shape if no patch qualifies. Validate this behavior with a mixed valid/impossible-case test and preserve deterministic replay.

### I5 — Case filtering happens after mandatory identity validation of excluded cases

Location: `rstar4d/simulation/pipeline.py:55`, `rstar4d/simulation/pipeline.py:203`–`208`.

Trigger: the inventory contains a known valid case plus another unresolved nested-animal identity; request only the known case with `--case-id` or a valid first case with `--limit 1`. `read_raw_manifest` rejects the unresolved identity before the requested subset is selected.

Effect: the documented bounded real-case workflow cannot operate on an otherwise valid selected case until all unrelated inventory ambiguities are resolved. Actual inventory has the described nested-ID category, so this affects ordinary use rather than only malformed synthetic inputs.

Evidence: a CPU reader reproduction with `selected/known` followed by `excluded/animal_id=None` failed with `excluded: explicit animal_id/identity_map required`; code applies filtering only after that call.

Fix: separate basic whole-manifest parsing/duplicate-ID validation from selected-case semantic preflight. Apply case/limit selection, then require identities, paths, phase coverage and geometry for every selected case. Unselected ambiguities must remain visible in inventory diagnostics and must still fail if later selected.

### I6 — Expansion failure leaves a stale generation-complete marker

Location: `rstar4d/simulation/pipeline.py:196`–`218`.

Trigger: follow the documented workflow: generate `--limit 1`, then rerun the same directory for all cases. The first run writes `generation_summary.json` with `complete:true`. The second call never marks the new requested generation incomplete; if any added case raises or the process is interrupted, the old completion summary remains alongside the newly partial case and incremental manifest.

Effect: consumers inspecting the directory's completion marker see a completed generation even though the latest requested operation failed. Per-case `.partial` handling is sound, but the overall completion state is stale. This violates the explicit interruption review focus.

Evidence: direct control-flow inspection; `generation_summary.json` is written only on success, without a start/failure transition or requested-case-set identity.

Fix: atomically record the current request ID/config/requested case set and `complete:false` before generation work; update completed case IDs incrementally and only mark complete after verifying the entire requested set. Preserve a prior successful summary under its own request/history identity if desired. Add a regression that succeeds for one case, fails during expansion, then verifies the current status is incomplete.

## Minor

### M1 — Projection metrics lack the promised aggregate report

Location: `rstar4d/evaluation/report.py:48`–`68`.

Per-case projection metrics are added to `cases.jsonl`, but `summary.json` is aggregated exclusively from the image-metric `flat` rows. With `--projection`, no case-equal/animal-equal projection MAE, RMSE or relative-L2 summary is produced; projection eligibility/constant-view counts are likewise absent from the summary. Spec §12 explicitly calls for both case and animal aggregation. Add a separate projection summary with applicable/skipped counts and null-aware statistics; keep line-integral units explicit.

### M2 — Acceptance evidence is narrower than the checklist currently suggests

Location: `tests/test_simulation.py:48`, `tests/test_acceptance.py:47`, `docs/superpowers/plans/2026-11-16-implementation-progress.md`.

`test_cuda_reference_reconstruction_and_mask` compares the new wrapper/mask and checks a water sphere, but does not invoke the external original reconstruction function for numerical parity. The source-offset self-projection test supplies nonzero source-offset *metadata* while projecting an unshifted output volume; it usefully tests that the evaluator ignores that metadata, but does not establish the full shifted-source→FDK/output-grid→projection chain. The ledger also still says Task 5 is in progress while the review package requests final acceptance. Preserve the valid 24-test evidence, but distinguish it from remaining original-function parity, shifted/odd-grid cases, and the root's separate bounded real-case validation. Update the final acceptance ledger with concrete evidence rather than treating those broader requirements as implied by current test names.

## Focus assessment and declined judgments

- For finite arrays, reviewed loss and default image-metric logic correctly exclude invalid voxels/windows; empty-valid loss errors and empty metric regions produce null. The sampling distribution defect is I4. No additional concrete mask-gradient defect was found.
- Existing tests exercise public state loading/forward/loss, Stage I/II initialization, T10/T11, odd spatial sizes and positive tiled coverage. I3 is outside the one-case-per-animal resume fixture.
- Static simulation code applies source offsets during projection, rebuilds centered reconstruction geometry, and applies calibration after FDK with GT validity zeroing first. This is consistent with the reviewed source design; it is not a new numerical parity claim.
- Declined to judge real held-out-animal recovery quality, complete 60/30-epoch convergence, medical usefulness, and paper-table reproduction; no such evidence was available or required for this code review.
- Declined to judge the root's still-running GPU real-case validation, new GPU memory measurements, and cross-version CUDA numerical parity. No GPU work was run by this reviewer.
- Declined to reinterpret historical intentional GT/CBCT configuration differences as bugs. They were expressly outside scope.

Acceptance recommendation: fix I1–I6 with focused regression tests, add the missing projection summary if claiming full Spec completion, and reconcile the evidence/ledger. Re-review the changed boundaries; the existing passing suite need not be repeatedly rerun without a concrete reason.
