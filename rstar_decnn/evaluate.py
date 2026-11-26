"""Freeze validation coefficients, then compare blinded animal-level pairs."""
from collections import defaultdict
from pathlib import Path
import csv
import math

import h5py
import numpy as np
import torch
from rstar4d.evaluation.image_metrics import image_metrics
from rstar4d.evaluation.lv_metrics import evaluate_lv, load_annotations, summarize_lv
from rstar4d.utils.io import atomic_json, atomic_jsonl, canonical_hash, read_json, sha256

from .data import load_rows, read_case, snapshot
from .predict import cache_rows, fuse


GROUPS = ("RSTAR", "RSTAR-STAM", "RSTAR + DeCNN", "RSTAR-STAM + DeCNN", "DeCNN")
COMPARISONS = ((GROUPS[1], GROUPS[0]), (GROUPS[2], GROUPS[0]), (GROUPS[3], GROUPS[1]), (GROUPS[4], GROUPS[1]))
METRICS = ("boundary_mae", "motion_error_ml", "mae", "ssim")


def evaluator_signature():
    root = Path(__file__).resolve().parent
    return {k: sha256(root / k) for k in ("evaluate.py", "predict.py", "data.py")}


def settings(window=(0.0, 0.05), tolerance_ml=0.0, minimum_motion_animals=1):
    if len(window) != 2 or not all(math.isfinite(v) for v in window) or window[1] <= window[0]:
        raise ValueError("provide a fixed increasing physical SSIM window")
    if not math.isfinite(tolerance_ml) or tolerance_ml < 0 or minimum_motion_animals < 1:
        raise ValueError("invalid pre-registered motion criteria")
    return {"window": list(window), "ssim_window": 11, "sigma": 1.5,
            "motion_tolerance_ml": tolerance_ml, "minimum_motion_animals": minimum_motion_animals}


def load_extractor(path, device):
    # File objects support Windows paths with non-ASCII characters in TorchScript.
    with Path(path).open("rb") as file:
        model = torch.jit.load(file, map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def predict(volume):
        source = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None].to(device)
        with torch.inference_mode():
            result = model(source)
        if result.shape == source.shape:
            result = result[0, 0]
        return result.cpu().numpy()

    return predict


def inputs(manifest, split, indexes, annotations, extractor_path, device, split_name, metric):
    rows, split = load_rows(manifest, split)
    selected = {r["case_id"]: r for r in rows}
    annotations_digest = sha256(annotations)
    masks = load_annotations(annotations, selected)
    extraction = load_extractor(extractor_path, device) if extractor_path else None
    cohorts, protocols = {}, {}
    for index in indexes:
        cache, protocol = cache_rows(index, rows, split, split_name)
        seed = str(protocol["seed"])
        if seed in cohorts:
            raise ValueError("duplicate seed in prediction caches")
        cohorts[seed] = cache
        protocols[seed] = protocol
    if not cohorts:
        raise ValueError("at least one prediction cache is required")
    stable = ("execution", "prior_mode", "transform", "engineering", "precision", "upstream", "decnn_prior_experiment")
    if len({canonical_hash({k: p.get(k) for k in stable}) for p in protocols.values()}) != 1:
        raise ValueError("paired seeds must use the same execution and intensity contracts")
    contract = {
        "data": canonical_hash(snapshot(rows, split)), "protocols": protocols,
        "annotations_sha256": annotations_digest,
        "extractor_sha256": sha256(extractor_path) if extractor_path else None,
        "extractor_device": str(device),
        "metrics": metric, "evaluator": evaluator_signature(),
    }
    return cohorts, contract, masks, extraction


def _score(p, sample, row, masks, metric, extractor, full):
    meta = row["metadata"]
    if not np.isfinite(p).all():
        raise ValueError(f"{row['case_id']}: nonfinite prediction")
    source = {"target": sample["y"], "valid_mask": sample["valid"]}
    lv = evaluate_lv(p, source, meta, masks, metric, extractor)
    result = {
        "case_id": row["case_id"], "animal_id": str(row["animal_id"]), "lv": lv,
        "boundary_mae": lv["boundary_mae"], "motion_error_ml": lv["motion"]["absolute_error_ml"],
    }
    if full:
        phase_metrics = [
            image_metrics(p[i:i+1], source["target"][i:i+1], source["valid_mask"][i:i+1],
                          metric["window"], win_size=metric["ssim_window"], sigma=metric["sigma"])
            for i in range(len(p))
        ]
        result["full_image_phases"] = phase_metrics
        for key in ("mae", "ssim"):
            vals = [m[key] for m in phase_metrics if m[key] is not None]
            result[key] = float(np.mean(vals)) if vals else None
    return result


def _scan(case, masks):
    row = case["row"]
    sample = read_case(row)
    if sample["y"] is None:
        raise ValueError(f"{row['case_id']}: evaluation requires a reference target")
    with h5py.File(case["path"], "r") as f:
        branches = {k: f[k][:] for k in ("R", "S", "D")}
    return branches, sample, masks.get(row["case_id"], {})


def mean_by_animal(results, metric):
    animals = defaultdict(lambda: defaultdict(list))
    for seed, scans in results.items():
        for scan in scans.values():
            if scan.get(metric) is not None:
                animals[scan["animal_id"]][seed].append(scan[metric])
    return {
        animal: float(np.mean([np.mean(values) for values in seeds.values()]))
        for animal, seeds in sorted(animals.items()) if len(seeds) == len(results)
    }


def pair_group(candidate, reference, metric):
    shared = defaultdict(lambda: defaultdict(list))
    missing = []
    for seed in sorted(reference):
        for case, baseline in reference[seed].items():
            other = candidate[seed][case]
            v, r = other.get(metric), baseline.get(metric)
            if r is not None and v is not None:
                shared[baseline["animal_id"]][seed].append(v - r)
            elif r is not None:
                missing.append({"seed": seed, "case_id": case, "animal_id": baseline["animal_id"]})
    for other in missing:
        shared.pop(other["animal_id"], None)
    by_animal = {
        animal: float(np.mean([np.mean(pairs) for pairs in seeds.values()]))
        for animal, seeds in sorted(shared.items()) if len(seeds) == len(reference)
    }
    return by_animal, missing


def additional_extraction_failures(candidate, reference):
    failures = []
    for seed, scans in reference.items():
        for case, baseline in scans.items():
            other = candidate[seed][case]
            statuses = {phase["phase_label"]: phase.get("extraction_status") for phase in other["lv"]["phases"]}
            for phase in baseline["lv"]["phases"]:
                label = phase["phase_label"]
                if phase.get("extraction_status") == "ok" and statuses.get(label) != "ok":
                    failures.append({"seed": seed, "case_id": case, "phase_label": label,
                                     "status": statuses.get(label)})
    return failures


def _candidate_table(by_alpha, grid, metric):
    baseline = by_alpha[0.0]
    motion_base = mean_by_animal(baseline, "motion_error_ml")
    table = []
    for alpha in grid:
        candidate = by_alpha[alpha]
        boundary = mean_by_animal(candidate, "boundary_mae")
        paired_boundary, missing_boundary = pair_group(candidate, baseline, "boundary_mae")
        motion, missing_motion = pair_group(candidate, baseline, "motion_error_ml")
        delta = float(np.mean(list(motion.values()))) if motion else None
        extra_extractions = additional_extraction_failures(candidate, baseline)
        cohort_match = not missing_motion and not extra_extractions and set(motion) == set(motion_base)
        evidence = cohort_match and len(motion) >= metric["minimum_motion_animals"]
        eligible = (evidence and delta <= metric["motion_tolerance_ml"]
                    and bool(boundary) and set(boundary) == set(paired_boundary) and not missing_boundary)
        statuses = summarize_lv([row for cases in candidate.values() for row in cases.values()])["extraction_status_counts"]
        table.append({
            "alpha": alpha, "boundary_mae": float(np.mean(list(boundary.values()))) if boundary else None,
            "n_boundary_animals": len(paired_boundary), "motion_delta_ml": delta, "n_motion_animals": len(motion),
            "motion_evidence": bool(evidence), "eligible": bool(eligible), "extra_missing_motion": missing_motion,
            "extraction_status_counts": statuses, "extra_extraction_failures": extra_extractions,
        })
    return table


def calibrate(manifest, split, indexes, annotations, extractor_path, out, *, device="cpu",
              metric=None, alpha_grid=None):
    metric = metric or settings()
    grid = sorted(set(alpha_grid if alpha_grid is not None else [i / 20 for i in range(21)]))
    if not grid or grid[0] != 0 or grid[-1] != 1 or not all(math.isfinite(x) for x in grid):
        raise ValueError("calibration grid must include both endpoints")
    cohorts, contract, masks, extractor = inputs(
        manifest, split, indexes, annotations, extractor_path, device, "val", metric,
    )
    results = {k: {a: {s: {} for s in cohorts} for a in grid} for k in ("R", "S")}
    for seed, cases in cohorts.items():
        for case_id, case in cases.items():
            images, sample, labels = _scan(case, masks)
            common_endpoint = _score(images["D"], sample, case["row"], labels, metric, extractor, False)
            for base in ("R", "S"):
                for alpha in grid:
                    p = fuse(images[base], images["D"], alpha, inference=True)
                    results[base][alpha][seed][case_id] = (
                        common_endpoint if alpha == 1 else _score(p, sample, case["row"], labels, metric, extractor, False)
                    )
    selections = {}
    for branch, data in results.items():
        table = _candidate_table(data, grid, metric)
        candidates = [row for row in table if row["eligible"]]
        exploration = [row for row in table if row["boundary_mae"] is not None]
        order = lambda row: (row["boundary_mae"], row["alpha"])
        selections[branch] = {
            "alpha": min(candidates, key=order)["alpha"] if candidates else None,
            "boundary_exploration_alpha": min(exploration, key=order)["alpha"] if exploration else None,
            "status": "locked" if candidates else "motion_unverified",
            "validation_grid": table,
        }
    value = {
        "version": 1, "kind": "validation_lock", "contract": contract, "choices": selections,
        "validation_indexes": [sha256(p) for p in indexes],
    }
    value["lock_sha256"] = canonical_hash(value)
    if Path(out).exists():
        if read_json(out) != value:
            raise FileExistsError("validation lock exists with a different contract or result")
    else:
        atomic_json(out, value)
    return value


def paired_summary(candidate, baseline, metric, draws=2000, seed=2026):
    values, missing = pair_group(candidate, baseline, metric)
    diff = np.array(list(values.values()))
    if not diff.size:
        return {"n_animals": 0, "mean_difference": None, "interval_95": None,
                "benefiting_animals": 0, "extra_missing": missing, "animals": values}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(draws)])
    return {
        "n_animals": len(diff), "mean_difference": float(diff.mean()),
        "interval_95": np.quantile(means, [0.025, 0.975]).tolist(),
        "benefiting_animals": int((diff > 0 if metric == "ssim" else diff < 0).sum()),
        "extra_missing": missing, "animals": values,
    }


def _profiles(candidates):
    if not candidates:
        return None
    return max(candidates, key=lambda p: (abs(p["reference_delta_ml"]), p["case_id"], p["seed"]))


def _profile(case, seed, scores, sample, images):
    motion = scores[GROUPS[0]]["lv"]["motion"]
    if motion.get("reference_delta_ml") is None:
        return None
    roles = {p["role"]: p["phase_label"] for p in scores[GROUPS[0]]["lv"]["phases"] if p.get("role")}
    y = sample["y"]
    ed, es = (sample["phase_labels"].index(roles[role]) for role in ("ED", "ES"))
    common = sample["valid"][ed] & sample["valid"][es]
    if not common.any():
        return None
    movement = np.where(common, np.abs(y[ed] - y[es]), -1)
    z, h, _ = np.unravel_index(movement.argmax(), movement.shape)
    affine = np.array(sample["geometry"]["voxel_to_scanner_zyx"])
    points = [list(map(float, (affine @ [z, h, w, 1])[:3])) for w in range(y.shape[-1])]
    return {
        "case_id": case, "seed": seed, "reference_delta_ml": motion["reference_delta_ml"],
        "scanner_zyx_mm": points, "z": int(z), "y": int(h),
        "phases": {
            role: {"phase_label": sample["phase_labels"][ti],
                   "valid": sample["valid"][ti, z, h].tolist(),
                   "input": sample["x"][ti, z, h].tolist(),
                   "reference": y[ti, z, h].tolist(),
                   **{name: p[ti, z, h].tolist() for name, p in images.items()}}
            for role, ti in (("ED", ed), ("ES", es))
        },
    }


def evaluate(manifest, split, indexes, annotations, extractor_path, lock, out, *, device="cpu"):
    value = read_json(lock)
    checksum = value.pop("lock_sha256")
    if value["version"] != 1 or value["kind"] != "validation_lock" or checksum != canonical_hash(value):
        raise ValueError("invalid validation lock")
    if any(sha256(path) in value["validation_indexes"] for path in indexes):
        raise ValueError("held-out and validation caches must differ")
    metric = value["contract"]["metrics"]
    cohorts, contract, masks, extractor = inputs(
        manifest, split, indexes, annotations, extractor_path, device, "test", metric,
    )
    if value["contract"] != contract:
        raise ValueError("held-out contract differs from validation lock")
    alpha = {key: choice["alpha"] for key, choice in value["choices"].items()}
    unlocked = [key for key, a in alpha.items() if a is None]
    groups = tuple(name for i, name in enumerate(GROUPS) if i not in
                   {2 if k == "R" else 3 for k in unlocked})
    results = {name: {seed: {} for seed in cohorts} for name in groups}
    profile_candidates = []
    performance = []
    for seed, cases in cohorts.items():
        for case_id, case in cases.items():
            raw, sample, labels = _scan(case, masks)
            p = {GROUPS[0]: raw["R"], GROUPS[1]: raw["S"], GROUPS[4]: raw["D"]}
            for i, key in ((2, "R"), (3, "S")):
                if alpha[key] is not None:
                    p[GROUPS[i]] = fuse(raw[key], raw["D"], alpha[key], inference=True)
            score = {name: _score(image, sample, case["row"], labels, metric, extractor, True)
                     for name, image in p.items()}
            for name, result in score.items():
                results[name][seed][case_id] = result
            profile = _profile(case_id, seed, score, sample, p)
            if profile:
                profile_candidates.append(profile)
            performance.append({k: case["metadata"][k] for k in
                                ("case_id", "seconds", "peak_cuda_allocated_bytes")} | {"seed": seed})
    summary = {}
    lines = []
    for name, by_seed in results.items():
        group = {}
        for key in METRICS:
            vals = mean_by_animal(by_seed, key)
            group[key] = {"n_animals": len(vals), "mean": float(np.mean(list(vals.values()))) if vals else None,
                          "animals": vals}
        rows = [{"group": name, "seed": seed, **row} for seed, scans in by_seed.items() for row in scans.values()]
        lines.extend(rows)
        status = summarize_lv(rows)
        group["boundary_status_counts"] = status["boundary_status_counts"]
        group["extraction_status_counts"] = status["extraction_status_counts"]
        group["motion_status_counts"] = status["motion_status_counts"]
        summary[name] = group
    comparisons = {
        candidate + " vs " + reference: {
            key: paired_summary(results[candidate], results[reference], key) for key in METRICS
        } for candidate, reference in COMPARISONS if candidate in results and reference in results
    }
    for candidate, reference in COMPARISONS:
        if candidate in results and reference in results:
            comparisons[candidate + " vs " + reference]["extra_extraction_failures"] = (
                additional_extraction_failures(results[candidate], results[reference])
            )
    primary = comparisons.get(GROUPS[3] + " vs " + GROUPS[1])
    decision = "motion_unverified"
    if primary:
        boundary, motion = primary["boundary_mae"], primary["motion_error_ml"]
        enough = (motion["n_animals"] >= metric["minimum_motion_animals"] and not motion["extra_missing"]
                  and motion["n_animals"] == summary[GROUPS[1]]["motion_error_ml"]["n_animals"]
                  and not primary["extra_extraction_failures"])
        if enough and boundary["mean_difference"] is not None:
            if alpha["S"] == 0:
                decision = "baseline_selected"
            elif alpha["S"] == 1:
                decision = "decnn_only_selected"
            elif boundary["mean_difference"] < 0 and motion["mean_difference"] <= metric["motion_tolerance_ml"]:
                interval = boundary["interval_95"]
                decision = "directional_improvement" if interval[0] <= 0 <= interval[1] else "combination_supported"
            else:
                decision = "combination_criteria_not_met"
    report = {
        "lock_sha256": checksum, "alpha": alpha, "groups": summary, "paired": comparisons, "decision": decision,
        "engineering": next(iter(contract["protocols"].values()))["engineering"], "contract": contract,
        "runtime": performance, "representative_motion_profile": _profiles(profile_candidates),
    }
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("evaluation output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(out / "cases.jsonl", lines)
    write_table(out / "comparison.csv", summary, comparisons, alpha)
    atomic_json(out / "summary.json", report)
    return report


def write_table(path, groups, comparisons, alpha):
    refs = dict(COMPARISONS)
    coefficients = {GROUPS[0]: 0, GROUPS[1]: 0, GROUPS[2]: alpha["R"], GROUPS[3]: alpha["S"], GROUPS[4]: 1}
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["group", "alpha", "metric", "mean", "n_animals", "reference", "paired_n_animals",
                  "paired_mean_difference", "interval_95_low", "interval_95_high", "benefiting_animals"]
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for name in GROUPS:
            for key in METRICS:
                row = {"group": name, "alpha": coefficients[name], "metric": key}
                if name in groups:
                    values = groups[name][key]
                    row.update(mean=values["mean"], n_animals=values["n_animals"])
                ref = refs.get(name)
                paired = comparisons.get(name + " vs " + ref, {}).get(key) if ref else None
                if paired:
                    low, high = paired["interval_95"] or (None, None)
                    row.update(reference=ref, paired_n_animals=paired["n_animals"],
                               paired_mean_difference=paired["mean_difference"], interval_95_low=low,
                               interval_95_high=high, benefiting_animals=paired["benefiting_animals"])
                writer.writerow(row)
