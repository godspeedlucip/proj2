from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import map_coordinates
import yaml

from .data import artifact_rows, atomic_path, case_rows
from .guard import context_for, enforce_split
from ..config import merge_strict, metric_defaults
from ..data.contracts import metadata
from ..evaluation.image_metrics import image_metrics
from ..evaluation.report import stats
from ..utils.io import atomic_json, atomic_jsonl, canonical_hash, read_json, resolve, sha256


def settings(path):
    defaults = {**metric_defaults(), "blood_pool_label": None, "annotations": None,
                "reference_group": "b", "blood_pool_mae_increase": 0., "tolerance_evidence": None,
                "phase_labels": None,
                "edge": {"half_length_mm": 5., "step_mm": .25, "plateau_mm": 1., "min_contrast": .001}}
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) if path else {}
    config = merge_strict(defaults, values or {})
    for key in ("annotations", "tolerance_evidence"):
        if config[key]:
            config[key] = str(resolve(config[key], Path(path).resolve().parent))
    edge = config["edge"]
    if (not np.isfinite(list(edge.values())).all() or min(edge.values()) <= 0
            or edge["plateau_mm"] >= edge["half_length_mm"] or 2 * edge["step_mm"] > edge["plateau_mm"]):
        raise ValueError("Invalid wall-sampling interval, plateau, step or contrast limit")
    if config["blood_pool_mae_increase"] < 0 or not np.isfinite(config["blood_pool_mae_increase"]):
        raise ValueError("Invalid blood-pool curve MAE tolerance")
    if config["blood_pool_mae_increase"] and not config["tolerance_evidence"]:
        raise ValueError("Nonzero blood-pool tolerance needs tolerance_evidence")
    if config["blood_pool_label"] is not None and (type(config["blood_pool_label"]) is not int
                                                    or config["blood_pool_label"] <= 0):
        raise ValueError("blood_pool_label must be a positive integer")
    phases = config["phase_labels"]
    if phases is not None and (not isinstance(phases, list) or not phases
                               or any(type(label) is not int for label in phases)
                               or len(set(phases)) != len(phases)):
        raise ValueError("phase_labels must be null or a nonempty list of unique integer labels")
    config["projection"] = False
    return config


def edge_measure(values, distances, plateau, minimum):
    first = values[distances <= distances[0] + plateau].mean()
    last = values[distances >= distances[-1] - plateau].mean()
    if abs(last - first) < minimum:
        raise ValueError("low contrast")
    normalized = (values - first) / (last - first)
    positions = []
    for threshold in (.1, .5, .9):
        crossing = []
        # Collapse an isolated exact threshold hit shared by two segments.
        for i, (a, b) in enumerate(zip(normalized[:-1], normalized[1:])):
            if (a <= threshold <= b or b <= threshold <= a) and a != b:
                crossing.append(float(np.interp(threshold, [min(a, b), max(a, b)],
                                                [distances[i], distances[i + 1]] if a < b
                                                else [distances[i + 1], distances[i]])))
        unique = {round(p, 10) for p in crossing}
        if len(unique) != 1 or np.any((normalized[1:] == threshold) & (normalized[:-1] == threshold)):
            raise ValueError("missing or multiple threshold crossings")
        positions.append(unique.pop())
    if positions != sorted(positions) or positions[0] >= positions[2]:
        raise ValueError("unordered thresholds")
    return positions[1], positions[2] - positions[0]


def profiles(images, support, affine, points, edge):
    reports = []
    inverse = np.linalg.inv(affine)
    steps = max(1, int(np.ceil(edge["half_length_mm"] / edge["step_mm"])))
    distances = np.linspace(-edge["half_length_mm"], edge["half_length_mm"], 2 * steps + 1)
    for point in points:
        center = np.asarray(point["center_scanner_zyx_mm"], dtype=float)
        normal = np.asarray(point["normal_scanner_zyx"], dtype=float)
        if (center.shape != (3,) or normal.shape != (3,) or not np.isfinite(np.r_[center, normal]).all()
                or np.linalg.norm(normal) == 0):
            raise ValueError("Invalid wall center or normal in annotations")
        xyz = center[:, None] + normal[:, None] / np.linalg.norm(normal) * distances
        voxel = inverse[:3, :3] @ xyz + inverse[:3, 3, None]
        measured = {}
        available = map_coordinates(support.astype(np.float32), voxel, order=1, mode="constant", cval=0.)
        for name, image in images.items():
            try:
                if np.any(available < 1. - 1e-6):
                    raise ValueError("sampling line outside scoring support")
                values = map_coordinates(image, voxel, order=1, mode="constant", cval=np.nan)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite profile")
                location, width = edge_measure(values, distances, edge["plateau_mm"], edge["min_contrast"])
                measured[name] = {"edge_mm": location, "width_mm": width}
            except ValueError as exc:
                measured[name] = {"reason": str(exc)}
        reports.append({"point_id": point["id"], "measurements": measured})
    return reports


def read_prediction(row, item, stack):
    if item is None:
        raise ValueError(f'{row["case_id"]}: missing group prediction')
    path = item["path"]
    if (sha256(path) != item.get("sha256") or row["sha256"] != item.get("source_sha256")
            or row["source_metadata_hash"] != item.get("source_metadata_hash")
            or str(row["animal_id"]) != str(item.get("animal_id"))):
        raise ValueError(f'{row["case_id"]}: prediction/source provenance mismatch')
    f = stack.enter_context(h5py.File(path, "r"))
    meta = metadata(f)
    for key in ("geometry", "phase_labels", "intensity_domain", "axis_order",
                "calibration_application", "normalization"):
        if meta.get(key) != row["metadata"][key]:
            raise ValueError(f'{row["case_id"]}: prediction metadata mismatch at {key}')
    pred = f["prediction"]
    if list(pred.shape) != row["shape"]:
        raise ValueError("Prediction TZYX shape mismatch")
    support = None
    if item.get("support_path"):
        if sha256(item["support_path"]) != item.get("support_sha256"):
            raise ValueError("MoCo evaluation support checksum mismatch")
        support = stack.enter_context(h5py.File(item["support_path"], "r"))
        if metadata(support) != row["metadata"] or list(support["external_weight"].shape) != row["shape"]:
            raise ValueError("Support geometry/phase/provenance mismatch")
    return pred, support


def aggregate_phase(rows):
    result = {}
    for metric in ("mae", "rmse", "ssim", "edge_location_error_mm", "edge_width_error_mm"):
        values = [row[metric] for row in rows if row.get(metric) is not None]
        result[metric] = float(np.mean(values)) if values else None
        result[metric + "_phases"] = len(values)
    curve = [(phase["phase_label"], phase["blood_pool_reference"], phase["blood_pool_prediction"])
             for phase in rows if phase.get("blood_pool_prediction") is not None]
    result["blood_pool_curve_mae"] = None
    result["blood_pool_peak_bias"] = None
    result["blood_pool_phases"] = [item[0] for item in curve]
    if curve:
        target = np.array([item[1] for item in curve])
        prediction = np.array([item[2] for item in curve])
        result["blood_pool_curve_mae"] = float(np.abs(target - prediction).mean())
        result["blood_pool_peak_bias"] = float(prediction.max() - target.max())
    return result


def registration_qa(row, item, context, annotations):
    if not context or not item.get("support_path") or item.get("output") not in ("d", "y_single", "y_multi"):
        return None
    summary_path = Path(item["support_path"]).parents[2] / "summary.json"
    summary = read_json(summary_path)
    if summary["request_signature"] != context["signature"]:
        raise ValueError("Field QC summary differs from prediction context")
    case = next(report for report in summary["cases"] if report["case_id"] == row["case_id"])
    pairs = {(pair["source"], pair["target"]): pair for p in case["phases"] for pair in p["pairs"]}
    affine = np.asarray(row["metadata"]["geometry"]["voxel_to_scanner_zyx"])
    inverse = np.linalg.inv(affine)
    bounds = np.asarray(row["shape"][1:]) - 1
    landmarks = []
    for landmark in annotations.get("landmarks", []):
        pair = (landmark["source_phase"], landmark["target_phase"])
        measured = {"id": landmark["id"], "source_phase": pair[0], "target_phase": pair[1]}
        landmarks.append(measured)
        if pair not in pairs:
            measured["reason"] = "phase pair unavailable"
            continue
        fixed = np.asarray(landmark["fixed_scanner_zyx_mm"], dtype=float)
        moving = np.asarray(landmark["moving_scanner_zyx_mm"], dtype=float)
        if fixed.shape != (3,) or moving.shape != (3,) or not np.isfinite(np.r_[fixed, moving]).all():
            raise ValueError("Invalid landmark coordinates")
        center = inverse[:3, :3] @ fixed + inverse[:3, 3]
        if np.any(center < 0) or np.any(center > bounds):
            measured["reason"] = "fixed landmark outside grid"
            continue
        field_key = pairs[pair]["field_key"]
        if field_key:
            path = Path(context["field_cache"]) / field_key[:2] / (field_key + ".npz")
            if not path.exists():
                measured["reason"] = "field absent from cache"
                continue
            with np.load(path, allow_pickle=False) as saved:
                field = saved["field"]
                if saved["key"].item() != field_key or field.shape != (3, *row["shape"][1:]):
                    raise ValueError("Landmark field provenance or layout mismatch")
                dz_dy_dx = np.array([map_coordinates(channel, center[:, None],
                                                      order=1, mode="nearest")[0]
                                     for channel in field[::-1]])
        else:
            dz_dy_dx = np.zeros(3)
        source_pos = center + dz_dy_dx
        if np.any(source_pos < 0) or np.any(source_pos > bounds):
            measured["reason"] = "warped landmark outside grid"
            continue
        predicted = affine[:3, :3] @ source_pos + affine[:3, 3]
        measured["tre_mm"] = float(np.linalg.norm(predicted - moving))
    return {"pairs": list(pairs.values()), "landmarks": landmarks,
            "landmarks_scored": sum("tre_mm" in item for item in landmarks),
            "landmarks_planned": len(landmarks)}


def read_case(row, items, contexts, case_annotations, config):
    labels = row["metadata"]["phase_labels"]
    phases = config["phase_labels"] or row["expected_phase_labels"]
    results = {name: {"full": [], "common_external": []} for name in items}
    detail = {"case_id": row["case_id"], "animal_id": row["animal_id"],
              "target_kind": row["metadata"]["target_kind"], "intensity_domain": row["metadata"]["intensity_domain"],
              "missing_labels": [label for label in phases if label not in labels],
              "scored_phase_labels": [label for label in labels if label in phases],
              "groups": results, "profiles": [], "registration_qa": {}}
    for name, item in items.items():
        qa = registration_qa(row, item, contexts[name], case_annotations)
        if qa is not None:
            detail["registration_qa"][name] = qa
    with ExitStack() as stack:
        reference = stack.enter_context(h5py.File(row["path"], "r"))
        if "target" not in reference:
            detail["reason"] = "no GT"
            return detail
        sources = {name: read_prediction(row, item, stack) for name, item in items.items()}
        for i, label in enumerate(labels):
            if label not in phases:
                continue
            gt = np.asarray(reference["target"][i], dtype=np.float32)
            gt_valid = np.asarray(reference["valid_mask"][i], dtype=bool)
            coverage = {name: (np.asarray(support["external_weight"][i]) > 0
                               if support is not None else None)
                        for name, (_, support) in sources.items()}
            common = gt_valid.copy()
            included = [mask for mask in coverage.values() if mask is not None]
            for mask in included:
                common &= mask
            if not included:
                common.fill(False)
            images = {name: np.asarray(data[i], dtype=np.float32) for name, (data, _) in sources.items()}
            blood_label = config["blood_pool_label"]
            blood_pool = (np.asarray(reference["roi_labels"][i]) == blood_label
                          if blood_label is not None and "roi_labels" in reference else np.zeros_like(gt_valid))
            points = case_annotations.get("wall_points", {}).get(str(label), [])
            affine = np.asarray(row["metadata"]["geometry"]["voxel_to_scanner_zyx"])
            gt_count = int(gt_valid.sum())
            for domain, valid in (("full", gt_valid), ("common_external", common)):
                edge_profiles = profiles({"gt": gt, **images}, valid, affine, points, config["edge"])
                if edge_profiles:
                    detail["profiles"].append({"phase_label": label, "domain": domain, "points": edge_profiles})
                common_edges = [item["measurements"] for item in edge_profiles
                                if all("edge_mm" in val for val in item["measurements"].values())]
                for name, image in images.items():
                    masks = {"all": None}
                    if "roi_labels" in reference and config["roi_labels"]:
                        mask_labels = np.asarray(reference["roi_labels"][i])
                        for key, roi in config["roi_labels"].items():
                            masks[str(roi)] = (mask_labels == int(key))[None]
                    score = {roi: image_metrics(image[None], gt[None], valid[None], config["window"],
                                                roi=mask, win_size=config["ssim_window"], sigma=config["sigma"])
                             for roi, mask in masks.items()}
                    blood = blood_pool & valid
                    edge_errors = [abs(item[name]["edge_mm"] - item["gt"]["edge_mm"]) for item in common_edges]
                    width_errors = [abs(item[name]["width_mm"] - item["gt"]["width_mm"]) for item in common_edges]
                    external = coverage[name]
                    support = sources[name][1]
                    fusion_valid = np.asarray(support["valid_mask"][i], dtype=bool) if support is not None else None
                    results[name][domain].append({
                        "phase_label": label, **score["all"], "roi": score,
                        "edge_location_error_mm": float(np.mean(edge_errors)) if edge_errors else None,
                        "edge_width_error_mm": float(np.mean(width_errors)) if width_errors else None,
                        "edge_points": len(edge_errors), "edge_points_planned": len(points),
                        "blood_pool_prediction": float(image[blood].mean(dtype=np.float64)) if blood.any() else None,
                        "blood_pool_reference": float(gt[blood].mean(dtype=np.float64)) if blood.any() else None,
                        "blood_pool_voxels": int(blood.sum()),
                        "external_coverage": float((external & gt_valid).sum()) / gt_count
                        if external is not None and gt_count else None,
                        "fusion_valid_coverage": float((fusion_valid & gt_valid).sum()) / gt_count
                        if fusion_valid is not None and gt_count else None,
                        "reinjection_fallback_fraction": float((gt_valid & ~external).sum()) / gt_count
                        if external is not None and gt_count and items[name].get("output") in ("y_single", "y_multi")
                        else None,
                    })
    detail["case_equal_phase"] = {name: {domain: aggregate_phase(values) for domain, values in doms.items()}
                                  for name, doms in results.items()}
    return detail


def load_groups(requests):
    groups, contexts = {}, {}
    for request in requests:
        name, separator, path = request.partition("=")
        if not separator or not name or not path or name == "gt" or name in groups:
            raise ValueError("--predictions requires unique NAME=MANIFEST arguments (gt is reserved)")
        groups[name] = artifact_rows(path)
        contexts[name] = context_for(path)
    return groups, contexts


METRICS = ("mae", "rmse", "ssim", "edge_location_error_mm", "edge_width_error_mm",
           "blood_pool_curve_mae", "blood_pool_peak_bias")


def summarize(cases, groups, reference):
    cohorts = defaultdict(list)
    for case in cases:
        if case.get("case_equal_phase"):
            cohorts[case["target_kind"], case["intensity_domain"]].append(case)
    results = {}
    for (target, intensity), members in cohorts.items():
        results[f"{target}/{intensity}"] = result = {}
        for group in groups:
            result[group] = domains = {}
            for domain in ("full", "common_external"):
                domains[domain] = scores = {}
                for metric in METRICS:
                    animals = defaultdict(list)
                    values, differences = [], []
                    for case in members:
                        row = case["case_equal_phase"]
                        value = row[group][domain][metric]
                        if value is not None:
                            values.append(value)
                            animals[str(case["animal_id"])].append(value)
                            baseline = row.get(reference, {}).get(domain, {}).get(metric)
                            if baseline is not None:
                                differences.append(value - baseline)
                    scores[metric] = {"scan_equal": stats(values), "delta_to_reference": stats(differences),
                                      "animal_equal": stats([float(np.mean(v)) for v in animals.values()])}
    return results


def write_tables(out, cases):
    fields = ["case_id", "animal_id", "target_kind", "intensity_domain", "group", "domain",
              "phase_label", "mae", "rmse", "ssim", "valid_voxels", "ssim_centers",
              "edge_location_error_mm", "edge_width_error_mm", "edge_points", "edge_points_planned",
              "blood_pool_prediction", "blood_pool_reference", "blood_pool_voxels",
              "external_coverage", "fusion_valid_coverage", "reinjection_fallback_fraction", "reason"]
    with atomic_path(out / "phase_metrics.csv") as tmp, tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            identity = {k: case[k] for k in fields[:4]}
            for group, domains in case["groups"].items():
                for domain, phases in domains.items():
                    writer.writerows({**identity, "group": group, "domain": domain, **phase} for phase in phases)


def evaluate_moco(requests, manifest, metric_path, out, split, lock_path):
    lock = enforce_split(manifest, split, lock_path)
    config = settings(metric_path)
    settings_inputs = {key: sha256(config[key]) if config[key] else None
                       for key in ("annotations", "tolerance_evidence")}
    if lock and (config != lock["metric_config"] or settings_inputs != lock["metric_input_sha256"]):
        raise ValueError("Test metrics, annotation file or tolerance evidence differs from the lock")
    groups, contexts = load_groups(requests)
    group_kinds = {name: {item.get("output") for item in items.values()} for name, items in groups.items()}
    if any(len(kinds) != 1 for kinds in group_kinds.values()):
        raise ValueError("A scoresheet group must have the same output kind across cases")
    group_kinds = {name: next(iter(kinds)) for name, kinds in group_kinds.items()}
    selected_manifest_rows = case_rows(manifest)
    for name, context in contexts.items():
        if context:
            mismatched = [case for case, item in groups[name].items()
                          if item.get("moco_signature") != context["signature"]]
            if mismatched:
                raise ValueError(f"{name}: prediction manifest disagrees with its MoCo run context")
        if lock:
            matches = [entry for entry in lock["runs"] if context and
                       all(entry.get(key) == context.get(key) for key in (
                           "config", "register_checkpoint_sha256", "baseline_provider"))]
            if not any(group_kinds[name] in entry["outputs"] for entry in matches):
                raise ValueError(f"{name}: prediction run or output differs from the validation lock")
    if lock and set(group_kinds.values()) != set(lock["required_outputs"]):
        raise ValueError("Test evaluation must supply the outputs listed in the validation lock")
    annotation_cases = read_json(config["annotations"])["cases"] if config["annotations"] else {}
    ids = {r["case_id"] for r in selected_manifest_rows}
    common = ids.intersection(*(set(items) for items in groups.values()))
    if common != ids:
        missing = {name: sorted(ids - set(items)) for name, items in groups.items() if ids - set(items)}
        raise ValueError(f"Explicit comparison manifest must match all groups; missing cases: {missing}")
    reports = []
    for row in selected_manifest_rows:
        case = row["case_id"]
        annotations = annotation_cases.get(case, {})
        if annotations and annotations.get("source_sha256") != row["sha256"]:
            raise ValueError(f"{case}: annotation source SHA-256 mismatch")
        items = {name: group[case] for name, group in groups.items()}
        reports.append(read_case(row, items, contexts, annotations, config))
    out = Path(out).resolve()
    data_path = out / "cases.jsonl"
    atomic_jsonl(data_path, reports)
    write_tables(out, reports)
    summary = {"split": split, "cases": {row["case_id"]: row["sha256"] for row in selected_manifest_rows},
               "source_manifest": str(Path(manifest).resolve()), "source_manifest_sha256": sha256(manifest),
               "metric_config": config, "metric_input_sha256": settings_inputs,
               "case_results": str(data_path), "case_results_sha256": sha256(data_path),
               "contexts": contexts, "group_outputs": group_kinds,
               "groups": {name: {"path": str(Path(request.partition("=")[2]).resolve()),
                                 "artifact_sha256": {case: items[case]["sha256"] for case in sorted(ids)}}
                          for request, (name, items) in zip(requests, groups.items())},
               "cohorts": summarize(reports, groups, config["reference_group"])}
    atomic_json(out / "summary.json", summary)
    return {"summary": str(out / "summary.json"), "matched_scans": len(reports),
            "cohorts": sorted(summary["cohorts"])}


def candidate_comparison(cases, baseline, single, multi, tolerance):
    location_gains = {"full": [], "common_external": []}
    reason = None
    for case in cases:
        data = case.get("case_equal_phase", {})
        if not data:
            return None, "missing reference"
        for domain in ("full", "common_external"):
            b, s, m = [data[group][domain] for group in (baseline, single, multi)]
            values = ("edge_location_error_mm", "edge_width_error_mm", "blood_pool_curve_mae")
            if any(entry[key] is None for entry in (b, s, m) for key in values):
                return None, "wall or blood-pool annotations unavailable on the comparison domain"
            if m["edge_width_error_mm"] > min(b["edge_width_error_mm"], s["edge_width_error_mm"]):
                reason = "wall width error increased"
            if m["blood_pool_curve_mae"] > min(b["blood_pool_curve_mae"], s["blood_pool_curve_mae"]) + tolerance:
                reason = "blood-pool curve MAE exceeded tolerance"
            location_gains[domain].append([b["edge_location_error_mm"] - m["edge_location_error_mm"],
                                           s["edge_location_error_mm"] - m["edge_location_error_mm"]])
    if reason:
        return None, reason
    gains = {domain: np.mean(values, axis=0) for domain, values in location_gains.items()}
    if any(min(values) <= 0 for values in gains.values()):
        return None, "no location-error improvement over both controls"
    return float(gains["full"][0]), None


def lock_moco(validation, out):
    summary = read_json(validation)
    if summary.get("split") != "val":
        raise ValueError("Lock requires an evaluate-moco --split-name val summary")
    if sha256(summary["case_results"]) != summary["case_results_sha256"]:
        raise ValueError("Validation case scores changed")
    from ..utils.io import read_jsonl
    cases = read_jsonl(summary["case_results"])
    baseline = summary["metric_config"]["reference_group"]
    contexts = summary["contexts"]
    if summary["group_outputs"].get(baseline) != "b" or not contexts.get(baseline):
        raise ValueError("Lock requires a b reference from a traceable infer-moco run")
    candidates = defaultdict(dict)
    for group, context in contexts.items():
        output = summary["group_outputs"][group]
        if context and output in ("y_single", "y_multi"):
            candidates[context["signature"]][output] = group
    decisions = []
    best = None
    for pair in candidates.values():
        if set(pair) != {"y_single", "y_multi"}:
            raise ValueError("C candidate must have same-run y_single and y_multi")
        single, multi = pair["y_single"], pair["y_multi"]
        alpha = contexts[multi]["config"]["reinjection"]["alpha"]
        if alpha == 0:
            continue
        gain, reason = candidate_comparison(cases, baseline, single, multi,
                                            summary["metric_config"]["blood_pool_mae_increase"])
        decisions.append({"alpha": alpha, "location_gain_mm": gain, "reason": reason})
        if gain is not None and (best is None or (gain, -alpha) > best[0]):
            best = ((gain, -alpha), multi)
    keys = ("config", "register_checkpoint_sha256", "baseline_provider")
    baseline_context = contexts[baseline]
    runs = [{"config": {**baseline_context["config"], "outputs": ["b"]},
             "register_checkpoint_sha256": None, "baseline_provider": baseline_context["baseline_provider"],
             "outputs": ["b"]}]
    if best:
        runs.append({**{k: contexts[best[1]][k] for k in keys}, "outputs": ["y_single", "y_multi"]})
    body = {"schema": 1, "validation_summary_sha256": sha256(validation), "validation_cases": summary["cases"],
            "best_alpha": -best[0][1] if best else 0., "selected_output": "y_multi" if best else "b",
            "candidates": decisions, "metric_config": summary["metric_config"],
            "metric_input_sha256": summary["metric_input_sha256"], "runs": runs,
            "required_outputs": ["b", "y_single", "y_multi"] if best else ["b"]}
    path = Path(out).resolve()
    atomic_json(path, {**body, "lock_sha256": canonical_hash(body)})
    return {"lock": str(path), "selected_output": body["selected_output"], "selected_alpha": body["best_alpha"]}
