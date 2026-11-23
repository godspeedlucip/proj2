import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects

from nnunetv2.inference.paired_tam import check_prediction_destinations
from nnunetv2.utilities.paired_tam_contract import PHASES
from nnunetv2.utilities.paired_tam_io import read_image


def segmentation_metrics(reference, prediction, spacing):
    n_reference, n_prediction = int(reference.sum()), int(prediction.sum())
    if n_reference == n_prediction == 0:
        return None, None
    dice = 2 * np.count_nonzero(reference & prediction) / (n_reference + n_prediction)
    if min(n_reference, n_prediction) == 0:
        return float(dice), None

    # The tight joint bbox preserves the complete surfaces with zero exterior padding.
    bbox = find_objects(reference | prediction, max_label=1)[0]
    reference, prediction = reference[bbox], prediction[bbox]
    reference_surface = reference & ~binary_erosion(reference)
    prediction_surface = prediction & ~binary_erosion(prediction)
    to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)[reference_surface]
    to_reference = distance_transform_edt(~reference_surface, sampling=spacing)[prediction_surface]
    hd95 = max(np.percentile(to_prediction, 95), np.percentile(to_reference, 95))
    return float(dice), float(hd95)


def _summarize(source, group_keys, metrics=("dice", "hd95_mm")):
    groups = defaultdict(list)
    for row in source:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summaries = []
    for group, values in sorted(groups.items()):
        summary = dict(zip(group_keys, group))
        for metric in metrics:
            defined = [row[metric] for row in values if row[metric] is not None]
            summary[metric] = sum(defined) / len(defined) if defined else None
            summary[f"{metric}_defined"] = len(defined)
        summary["count"] = len(values)
        summaries.append(summary)
    return summaries


def paired_patient_deltas(reference, candidate):
    phase_structure = ("phase", "label", "structure")
    patient_phase_structure = ("patient_id", *phase_structure)

    def index(rows):
        return {tuple(row[key] for key in patient_phase_structure): row for row in rows}

    reference, candidate = index(reference), index(candidate)
    if reference.keys() != candidate.keys():
        raise ValueError("Ablation arms must score the same patients, phases and structures.")
    metrics = ("dice_delta", "hd95_mm_delta")
    per_patient = []
    for key in sorted(reference):
        row = dict(zip(patient_phase_structure, key))
        for original, delta in zip(("dice", "hd95_mm"), metrics):
            baseline, observed = reference[key][original], candidate[key][original]
            row[delta] = observed - baseline if observed is not None and baseline is not None else None
        per_patient.append(row)
    both_phases = _summarize(per_patient, ("patient_id", "label", "structure"), metrics)
    return {
        "per_patient": per_patient,
        "by_phase_and_structure": _summarize(per_patient, phase_structure, metrics),
        "both_phases_by_structure": _summarize(both_phases, ("label", "structure"), metrics),
    }


def evaluate_pairs(predictor, records, split, output_directory, save_probabilities=False, *, source_records=None):
    output_directory = Path(output_directory)
    check_prediction_destinations(
        [record["case_id"] for record in records], records if source_records is None else source_records,
        output_directory, predictor.dataset_json["file_ending"], save_probabilities, phases=predictor.phase_order,
    )
    label_manager = predictor.plans_manager.get_label_manager(predictor.dataset_json)
    label_names = {
        label: name for name, label in predictor.dataset_json["labels"].items()
        if label in label_manager.foreground_labels
    }
    rows = []
    for record in records:
        predictions = predictor.predict_case(record, output_directory, save_probabilities)
        for phase in PHASES:
            reference = read_image(record[phase]["label"])
            prediction = read_image(predictions[phase])
            if not reference.IsSameImageGeometryAs(prediction):
                raise ValueError(f"{record['case_id']} {phase}: prediction geometry differs from reference.")
            truth = sitk.GetArrayViewFromImage(reference)
            pred = sitk.GetArrayViewFromImage(prediction)
            valid = truth != label_manager.ignore_label if label_manager.has_ignore_label else True
            spacing = reference.GetSpacing()[::-1]
            for label, name in label_names.items():
                reference_fg, prediction_fg = truth == label, (pred == label) & valid
                dice, hd95 = segmentation_metrics(reference_fg, prediction_fg, spacing)
                rows.append({
                    "case_id": record["case_id"], "patient_id": record["patient_id"],
                    "phase": phase, "label": label, "structure": name, "dice": dice, "hd95_mm": hd95,
                    "empty_reference": not reference_fg.any(), "empty_prediction": not prediction_fg.any(),
                })

    per_patient = _summarize(rows, ("patient_id", "phase", "label", "structure"))
    # Average phases within patient before calculating the cohort means.
    both_phases = _summarize(per_patient, ("patient_id", "label", "structure"))
    summary = {
        "split": split,
        "model": predictor.checkpoint_metadata,
        "phase_order": list(PHASES),
        "num_patients": len({row["patient_id"] for row in rows}),
        "per_case": rows,
        "per_patient": per_patient,
        "by_phase_and_structure": _summarize(per_patient, ("phase", "label", "structure")),
        "both_phases_by_structure": _summarize(both_phases, ("label", "structure")),
        "lowest_dice_cases": sorted((row for row in rows if row["dice"] is not None),
                                    key=lambda row: row["dice"])[:10],
        "highest_hd95_cases": sorted((row for row in rows if row["hd95_mm"] is not None),
                                     key=lambda row: row["hd95_mm"], reverse=True)[:10],
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    return summary
