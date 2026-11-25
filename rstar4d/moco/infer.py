from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import time
import zipfile

import h5py
import numpy as np
import torch

from .data import artifact_rows, atomic_path, baseline_artifacts, case_rows, geometry_support, phase, validate_artifact
from .fusion import Accumulator, field_quality, ordered_pairs, reinject
from .registration import RegisNet3D, estimate_field
from .settings import dependencies
from .train import load_registration, normalize
from ..training.trainer import select_device
from ..utils.io import atomic_json, atomic_jsonl, canonical_hash, read_json, sha256


def array_options(shape):
    chunks = (1, min(shape[1], 16), min(shape[2], 128), min(shape[3], 128))
    return {"chunks": chunks, "compression": "gzip", "compression_opts": 1, "fletcher32": True}


class Fields:
    def __init__(self, config, model, checkpoint, root, device):
        self.settings = config["registration"]
        self.model = model
        self.device = device
        self.root = Path(root)
        self.checkpoint = checkpoint
        self.hits = self.calls = self.computed = 0
        if model is not None:
            model.register_forward_hook(self.count_forward)

    def count_forward(self, _model, _args, _output):
        self.calls += 1

    def get(self, row, input_hash, source, target, load):
        if self.model is None:
            return None, None, "zero"
        key = canonical_hash({
            "implementation": "regisnet-field-v1", "checkpoint": self.checkpoint,
            "case": row["case_id"], "grid": row["metadata"]["geometry"], "input": input_hash,
            "input_kind": self.settings["input"], "normalization": self.settings["normalization"],
            "tile": self.settings["inference_tile"], "halo": self.settings["inference_halo"],
            "limit": self.settings["max_displacement_vox"], "device": str(self.device),
            "runtime": str(torch.__version__), "cuda": torch.version.cuda,
            "ordered_source_target": [source, target],
        })
        filename = self.root / key[:2] / (key + ".npz")
        if filename.exists():
            try:
                with np.load(filename, allow_pickle=False) as saved:
                    field = saved["field"]
                    if (saved["key"].item() == key and field.dtype == np.float32
                            and field.shape == (3, *row["shape"][1:]) and np.isfinite(field).all()):
                        self.hits += 1
                        return torch.from_numpy(field).to(self.device)[None], key, "cache"
            except (ValueError, OSError, KeyError, zipfile.BadZipFile):
                pass
        moving, fixed = [normalize(torch.from_numpy(load(label)).to(self.device)[None, None],
                                   self.settings["normalization"]) for label in (source, target)]
        field = estimate_field(self.model, moving, fixed, self.settings["max_displacement_vox"],
                               self.settings["inference_tile"], self.settings["inference_halo"])
        if not torch.isfinite(field).all():
            raise FloatingPointError(f'{row["case_id"]}: nonfinite field {source} -> {target}')
        with atomic_path(filename) as temporary, temporary.open("wb") as file:
            np.savez(file, key=key, field=field[0].cpu().numpy())
        self.computed += 1
        return field, key, "computed"


def target_file(filename, signature):
    receipt = filename.with_suffix(".json")
    if filename.exists() and receipt.exists():
        try:
            record = read_json(receipt)
        except (OSError, ValueError):
            return None
        if record.get("signature") == signature and record.get("sha256") == sha256(filename):
            return record["qa"]
    return None


def fuse_target(config, row, labels, group, target, datasets, support, fields, input_hash):
    reg_input = config["registration"]["input"]
    fusion_source = config["fusion"]["source"]
    label_index = {label: i for i, label in enumerate(labels)}
    expected_index = {label: i for i, label in enumerate(row["expected_phase_labels"])}

    def load(kind, label):
        return phase(datasets[kind], label_index[label])

    device = fields.device
    index = label_index[target]
    target_support = support(index)
    acc = Accumulator(torch.from_numpy(load(fusion_source, target)).to(device),
                      torch.from_numpy(target_support).to(device))
    qc = []
    spacing = row["metadata"]["geometry"]["voxel_spacing_zyx_mm"]
    for source in group:
        field, key, status = fields.get(row, input_hash, source, target, lambda label: load(reg_input, label))
        adjacent = abs(expected_index[source] - expected_index[target]) == 1
        acc.add(torch.from_numpy(load(fusion_source, source)).to(device),
                torch.from_numpy(support(label_index[source])).to(device), field, adjacent=adjacent)
        if field is None:
            n = int(target_support.sum())
            stats = dict(target_voxels=n, mean_displacement_mm=0. if n else None,
                         max_displacement_mm=0. if n else None, out_of_bounds_fraction=0. if n else None,
                         folded_fraction=0. if n else None)
        else:
            stats = field_quality(field[0].cpu().numpy(), target_support, spacing)
        qc.append({"source": source, "target": target, "range": "near" if adjacent else "far",
                   "field_key": key, "field_status": status, **stats})
    combined = {kind: value.cpu().numpy() for kind, value in acc.result().items()
                if kind != "raw_sum" or config["fusion"]["save_raw_sum"]}
    combined["geometry_support"] = target_support
    external = combined["external_weight"] > 0
    for which, kind in (("y_single", "c"), ("y_multi", "d")):
        if which in config["outputs"]:
            baseline = load("b", target)
            source = combined["d"] if kind == "d" else load(kind, target)
            reinjection = config["reinjection"]
            combined[which] = reinject(baseline, source, external, reinjection["alpha"],
                                       reinjection["sigma_mm"], spacing)
    qa = {
        "phase_label": target, "pairs": qc,
        "geometry_voxels": int(target_support.sum()), "valid_voxels": int(combined["valid_mask"].sum()),
        "external_voxels": int(external.sum()), "near_voxels": int((combined["near_weight"] > 0).sum()),
        "far_voxels": int((combined["far_weight"] > 0).sum()),
        "external_weight_sum": float(combined["external_weight"].sum(dtype=np.float64)),
        "self_only_voxels": int((combined["valid_mask"] & ~external).sum()),
    }
    if "raw_sum" in combined:
        valid = combined["valid_mask"]
        difference = np.abs(combined["raw_sum"][valid] - combined["d"][valid])
        qa["raw_sum_vs_d"] = {"mean_abs": float(difference.mean()) if difference.size else None,
                              "max_abs": float(difference.max()) if difference.size else None}
    return combined, qa


def write_output(path, row, kind, targets, metadata, source=None):
    shape = tuple(row["shape"])
    limits = [float("inf"), float("-inf")]
    with atomic_path(path) as temporary, h5py.File(temporary, "w") as file:
        file.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        pred = file.create_dataset("prediction", shape=shape, dtype="f4", **array_options(shape))
        for i in range(shape[0]):
            if source is not None:
                value = phase(source, i)
            else:
                with h5py.File(targets[i], "r") as staged:
                    value = staged[kind][:]
            pred[i] = value
            limits[0] = min(limits[0], float(value.min()))
            limits[1] = max(limits[1], float(value.max()))
    return limits


def write_support(path, row, targets):
    shape = tuple(row["shape"])
    with atomic_path(path) as temporary, h5py.File(temporary, "w") as file:
        file.attrs["metadata_json"] = json.dumps(row["metadata"], ensure_ascii=False, allow_nan=False)
        for i, target in enumerate(targets):
            with h5py.File(target, "r") as staged:
                for name in staged.keys() - {"d", "y_single", "y_multi"}:
                    if i == 0:
                        file.create_dataset(name, shape=shape, dtype=staged[name].dtype,
                                            **array_options(shape))
                    file[name][i] = staged[name][:]


@torch.inference_mode()
def infer_moco(config, manifest, out, *, baseline=None, baseline_checkpoint=None, correction=None,
               register_checkpoint=None, field_cache=None):
    sources, fuse, _ = dependencies(config)
    rows = case_rows(manifest)
    out = Path(out).resolve()
    b_rows, provider = {}, None
    if "b" in sources:
        b_rows, provider = baseline_artifacts(config, rows, manifest, out / "baseline_cache",
                                              baseline, baseline_checkpoint)
    c_rows = artifact_rows(correction) if "c" in sources and correction else {}
    c_details = {row["case_id"]: validate_artifact(c_rows.get(row["case_id"]), row, "c",
                                                 config["pigae"]["mode"])
                 for row in rows} if "c" in sources else {}
    learned = fuse and config["registration"]["mode"] == "learned"
    dev = select_device(config["device"]) if learned else torch.device("cpu")
    torch.set_num_threads(config["threads"])
    model, checkpoint = None, None
    if learned:
        reg_provider = provider if config["registration"]["input"] == "b" else None
        saved = load_registration(register_checkpoint, config, rows, reg_provider)
        checkpoint = sha256(register_checkpoint)
        if any(row.get("scan_split_name") == "test" for row in rows):
            development = saved.get("data_snapshot", {})
            seen = {c["sha256"] for split in development.values() for c in split["cases"]}
            if any(row["sha256"] in seen for row in rows):
                raise ValueError("Test manifest overlaps registration training or validation input")
        model = RegisNet3D().to(dev)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
    fields = Fields(config, model, checkpoint, field_cache or out.parent / "fields", dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    signature_payload = {
        "schema": 1, "implementation": {name: sha256(Path(__file__).with_name(name))
                                        for name in ("infer.py", "fusion.py", "data.py", "registration.py")},
        "config": config, "source_manifest_sha256": sha256(manifest), "register_checkpoint_sha256": checkpoint,
        "baseline_provider": provider, "inference_device": str(dev), "torch": str(torch.__version__),
        "cases": [{**{key: row[key] for key in ("case_id", "animal_id", "sha256", "expected_phase_labels")},
                   "b_sha256": b_rows.get(row["case_id"], {}).get("sha256"),
                   "c_sha256": c_rows.get(row["case_id"], {}).get("sha256")}
                  for row in rows],
    }
    signature = canonical_hash(signature_payload)
    context = out / "inference_context.json"
    if context.exists() and read_json(context)["signature"] != signature:
        raise ValueError("Output directory belongs to a different infer-moco request")
    atomic_json(context, {**signature_payload, "signature": signature, "source_manifest": str(Path(manifest).resolve()),
                          "field_cache": str(fields.root.resolve())})
    records = {kind: [] for kind in config["outputs"]}
    case_reports = []
    resumed = 0
    for row in rows:
        case_start = time.perf_counter()
        case = row["case_id"]
        labels = row["metadata"]["phase_labels"]
        targets, phase_reports = [], []
        support_file = None
        geometry_rule = None
        case_report = {"case_id": case, "animal_id": row["animal_id"], "labels": labels,
                       "label_to_index": {str(label): i for i, label in enumerate(labels)},
                       "missing_labels": [label for label in row["expected_phase_labels"] if label not in labels],
                       "phases": phase_reports, "ranges": {}, "b": b_rows.get(case), "c": c_rows.get(case)}
        with ExitStack() as stack:
            source_file = stack.enter_context(h5py.File(row["path"], "r")) if "x" in sources or fuse else None
            datasets = {}
            if "x" in sources:
                datasets["x"] = source_file["cbct"]
            for kind, artifacts, name in [("b", b_rows, "prediction"), ("c", c_rows, "correction")]:
                if kind in sources and (fuse or kind == "c"):
                    file = stack.enter_context(h5py.File(artifacts[case]["path"], "r"))
                    datasets[kind] = file[name]
            if fuse:
                support, geometry_rule = geometry_support(source_file, row)
                pairing = ordered_pairs(labels, row["expected_phase_labels"], config["fusion"]["pairs"])
                estimate_input_hash = (b_rows[case]["sha256"] if learned and config["registration"]["input"] == "b"
                                       else row["sha256"])
                for target in labels:
                    path = out / "targets" / case / f"{target}.h5"
                    token = canonical_hash([signature, case, target])
                    qa = target_file(path, token)
                    if qa is None:
                        combined, qa = fuse_target(config, row, labels, pairing[target], target, datasets,
                                                   support, fields, estimate_input_hash)
                        with atomic_path(path) as temporary, h5py.File(temporary, "w") as result:
                            for name, image in combined.items():
                                result.create_dataset(name, data=image, compression="gzip",
                                                      compression_opts=1, fletcher32=True)
                        atomic_json(path.with_suffix(".json"), {"signature": token, "sha256": sha256(path), "qa": qa})
                    else:
                        resumed += 1
                        qa["resumed"] = True
                    targets.append(path)
                    phase_reports.append(qa)
                support_file = out / "support" / case / "coverage.h5"
                write_support(support_file, row, targets)
            support_sha = sha256(support_file) if support_file else None
            for kind in records:
                if kind == "b":
                    item = dict(b_rows[case])
                else:
                    path = out / kind / case / "prediction.h5"
                    meta = {**row["metadata"], "moco": {
                        "output": kind, "signature": signature, "geometry_support_rule": geometry_rule,
                        "registration_mode": config["registration"]["mode"] if fuse else None,
                        "registration_checkpoint_sha256": checkpoint, "pigae": c_details.get(case),
                        "expected_phase_labels": row["expected_phase_labels"],
                    }}
                    case_report["ranges"][kind] = write_output(path, row, kind, targets, meta,
                                                              datasets.get("x") if kind == "x" else None)
                    item = {"case_id": case, "animal_id": row["animal_id"], "path": str(path),
                            "sha256": sha256(path), "source_sha256": row["sha256"],
                            "source_metadata_hash": row["source_metadata_hash"],
                            "checkpoint_sha256": provider["checkpoint_sha256"] if provider else None}
                records[kind].append({**item, "output": kind, "moco_signature": signature,
                                      "support_path": str(support_file) if support_file else None,
                                      "support_sha256": support_sha})
        case_report["seconds"] = time.perf_counter() - case_start
        case_report["geometry_support_rule"] = geometry_rule
        case_reports.append(case_report)
    manifests = {}
    for kind, items in records.items():
        path = out / kind / "manifest.jsonl"
        atomic_jsonl(path, items)
        manifests[kind] = str(path)
    result = {"manifests": manifests, "request_signature": signature,
              "registration_mode": config["registration"]["mode"] if fuse else None,
              "fields_total": sum(len(p["pairs"]) for row in case_reports for p in row["phases"]),
              "forward_calls": fields.calls, "fields_computed": fields.computed,
              "cache_hits": fields.hits, "resumed_targets": resumed,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
              "cases": case_reports}
    atomic_json(out / "summary.json", result)
    return result
