"""Phasewise inference and physical-domain prediction caches."""
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import math
import os
import time

import h5py
import numpy as np
import torch
from rstar4d.inference.sliding_window import gaussian_weight, starts
from rstar4d.utils.io import atomic_jsonl, canonical_hash, read_jsonl, resolve, sha256

from . import checkpoint
from .backbone import check_case, normalize, physical, predict_backbone, restore_backbone
from .data import load_rows, read_case, snapshot, units


def fuse(backbone, decnn, alpha, *, inference=False):
    if inference:
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if alpha == 0 and backbone is not None:
            return backbone
        if alpha == 1 and decnn is not None:
            return decnn
    if backbone is None or decnn is None or backbone.shape != decnn.shape:
        raise ValueError("fusion requires aligned physical-domain predictions")
    # Keeping both operands in the training graph also gives alpha a gradient at zero.
    return (1 - alpha) * backbone + alpha * decnn


def decnn_volume(model, x, prior, transform, *, mode, tile, overlap, device, amp):
    if x.ndim != 4 or prior.shape not in (x.shape[1:], x.shape):
        raise ValueError("DeCNN prediction requires aligned TZYX and ZYX arrays")
    if mode not in {"full", "tiled"}:
        raise ValueError("inference mode must be full or tiled")
    device = torch.device(device)
    model.eval()
    shape = x.shape[1:]

    def run(gated, mean):
        arrays = [np.ascontiguousarray(normalize(a, transform), dtype=np.float32) for a in (gated, mean)]
        g, p = (torch.from_numpy(a)[None, None].to(device) for a in arrays)
        with torch.inference_mode(), torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            result = model(g, p)[0, 0].float().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite DeCNN prediction")
        return result

    result = np.empty(x.shape, np.float32)
    if mode == "full":
        for phase, gated in enumerate(x):
            result[phase] = run(gated, prior if prior.ndim == 3 else prior[phase])
    else:
        positions = [starts(n, k, o, 16) for n, k, o in zip(shape, tile, overlap)]
        tiles = []
        weights = gaussian_weight(tile)
        denominator = np.zeros(shape, np.float32)
        for origin in product(*positions):
            region = tuple(slice(a, min(a + k, n)) for a, k, n in zip(origin, tile, shape))
            real = tuple(s.stop - s.start for s in region)
            pad = tuple((0, k - n) for k, n in zip(tile, real))
            crop = tuple(slice(0, n) for n in real)
            tiles.append((region, pad, crop))
            denominator[region] += weights[crop]
        if (denominator <= 0).any():
            raise RuntimeError("DeCNN tiles do not cover the image")
        for phase, gated in enumerate(x):
            result[phase].fill(0)
            phase_prior = prior if prior.ndim == 3 else prior[phase]
            for region, pad, crop in tiles:
                p = np.pad(phase_prior[region], pad, mode="edge")
                g = np.pad(gated[region], pad, mode="edge")
                result[phase][region] += run(g, p)[crop] * weights[crop]
            result[phase] /= denominator
    return physical(result, transform)


def predict_case(model, spec, sample, transform, *, alpha, backbone_options, decnn_options, branches=None):
    def rstar():
        return predict_backbone(model.backbone, sample["x"], sample["a_s"], spec, **backbone_options)

    def enhanced():
        return decnn_volume(model.decnn, sample["x"], sample["a_d"], transform, **decnn_options)

    if branches is not None:
        funcs = {"S": rstar, "D": enhanced}
        return {key: funcs[key]() for key in branches}
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return fuse(rstar() if alpha != 1 else None, enhanced() if alpha != 0 else None, alpha, inference=True)


def run_predict(model_path, manifest, split, split_name, out, *, rstar_path=None, alpha=None,
                cache=False, device="cuda", amp=False, mode="tiled",
                tile=(32, 128, 128), overlap=(16, 64, 64), d_tile=None, d_overlap=None,
                baseline_design=None):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("prediction output directory must be empty")
    rows, split = load_rows(manifest, split)
    all_data = snapshot(rows, split)
    selected = [r for r in rows if split_name == "all" or str(r["animal_id"]) in split[split_name]]
    if not selected:
        raise ValueError("empty inference split")
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    value = checkpoint.load(model_path)
    if value["decnn"] is None:
        raise ValueError("Stage I checkpoint has no combination prediction")
    model = checkpoint.model_from(value, dev).eval()
    spec, config = value["backbone"], value["config"]
    del value["weights"]
    if value["data"]["split"] != split and split_name != "all":
        raise ValueError("inference split differs from training split")
    if cache and (value["policy"] != "frozen" or value["data"] != all_data):
        raise ValueError("ablation caches require the frozen recipe and the original cohort")
    alpha = value["alpha"] if alpha is None else float(alpha)
    if alpha < 0 or alpha > 1 or not math.isfinite(alpha):
        raise ValueError("alpha must be in [0, 1]")
    if cache and rstar_path is None:
        raise ValueError("three-branch cache requires an original Stage II RSTAR checkpoint")
    rstar = r_spec = r_source = None
    if rstar_path is not None:
        if not cache:
            raise ValueError("--rstar is the baseline for --cache")
        rstar, r_spec, r_source = restore_backbone(
            rstar_path, joint=False, prior_mode=config["prior_mode"], engineering=config["engineering"],
        )
        rstar.to(dev).eval()
        if spec["model_profile"] != r_spec["model_profile"]:
            raise ValueError("original RSTAR and RSTAR-STAM require matched profiles")
        if not config["engineering"]:
            expected = {"training_start", "update_budget", "animal_split", "matched_stam_training_start",
                        "matched_stam_update_budget"}
            if not isinstance(baseline_design, dict) or not expected.issubset(baseline_design):
                raise ValueError("formal ablation requires a documented baseline training design")
            if baseline_design["animal_split"] != split:
                raise ValueError("baseline design has a different animal split")
            if (baseline_design["training_start"] != baseline_design["matched_stam_training_start"]
                or baseline_design["update_budget"] != baseline_design["matched_stam_update_budget"]):
                raise ValueError("baseline and STAM design differ in training start or budget")
            if r_source["split"] != split:
                raise ValueError("baseline checkpoint must have the locked animal split")

    backbone_options = {"mode": mode, "tile": tuple(tile), "overlap": tuple(overlap), "device": dev, "amp": amp}
    decnn_options = {**backbone_options, "tile": tuple(d_tile or tile), "overlap": tuple(d_overlap or overlap)}
    protocol = {
        "cohort": canonical_hash(all_data), "split_hash": canonical_hash(split),
        "combined_checkpoint_sha256": sha256(model_path), "source_stam": value["source"],
        "original_rstar": r_source, "baseline_design": baseline_design,
        "upstream": canonical_hash(value["upstream"]), "prior_mode": config["prior_mode"],
        "transform": config["transform"], "engineering": config["engineering"], "seed": config["seed"],
        "decnn_prior_experiment": config["decnn_prior"],
        "execution": {k: {"mode": mode, "tile": list(opts["tile"]), "overlap": list(opts["overlap"])}
                      for k, opts in (("S", backbone_options), ("D", decnn_options))},
        "precision": "amp" if amp and dev.type == "cuda" else "fp32", "device": str(dev),
        "hardware": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
    }
    records = []
    for row in selected:
        check_case(spec, row)
        if row["backbone_prior_mode"] != config["prior_mode"]:
            raise ValueError(f"{row['case_id']}: unexpected backbone prior mode")
        if units(row["metadata"]) != config["transform"]["unit"]:
            raise ValueError(f"{row['case_id']}: DeCNN intensity unit differs")
        if not config["engineering"] and row["engineering"]:
            raise ValueError(f"{row['case_id']}: engineering fixture cannot be used for formal predictions")
        if config["decnn_prior"] != "self" and ((row["decnn_prior"]["kind"] == "shared") != (config["decnn_prior"] == "shared")):
            raise ValueError(f"{row['case_id']}: DeCNN prior experiment differs")
        if r_spec:
            check_case(r_spec, row)
        sample = read_case(row, supervision=False)
        if config["decnn_prior"] == "self":
            sample["a_d"] = sample["x"]
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        started = time.perf_counter()
        if cache:
            images = predict_case(model, spec, sample, config["transform"], alpha=alpha,
                                  backbone_options=backbone_options, decnn_options=decnn_options, branches=("S", "D"))
            images["R"] = predict_backbone(rstar, sample["x"], sample["a_s"], r_spec, **backbone_options)
        else:
            images = {"prediction": predict_case(model, spec, sample, config["transform"], alpha=alpha,
                                                  backbone_options=backbone_options, decnn_options=decnn_options)}
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - started
        metadata = {
            "case_id": row["case_id"], "animal_id": str(row["animal_id"]), "source_sha256": row["sha256"],
            "source_contract_hash": canonical_hash(row["metadata"]), "phase_labels": sample["phase_labels"],
            "geometry": sample["geometry"], "decnn_prior": {k: v for k, v in row["decnn_prior"].items() if k != "path"},
            "decnn_prior_experiment": config["decnn_prior"], "intensity_domain": spec["intensity_domain"],
            "unit": units(row["metadata"]), "calibration": row["metadata"]["calibration"],
            "protocol": protocol, "split_name": split_name, "alpha": None if cache else alpha,
            "seconds": elapsed,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
        }
        path = out / row["case_id"] / "prediction.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
        try:
            with h5py.File(temp, "w") as f:
                f.attrs["metadata_json"] = json.dumps(metadata, allow_nan=False)
                for label, image in images.items():
                    f.create_dataset(label, data=image, compression="gzip", compression_opts=1)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        records.append({
            "case_id": row["case_id"], "animal_id": str(row["animal_id"]), "sha256": sha256(path),
            "source_sha256": row["sha256"], "path": path.relative_to(out).as_posix(),
            "phase_labels": sample["phase_labels"], "datasets": sorted(images),
            "protocol_hash": canonical_hash(protocol), "metadata_hash": canonical_hash(metadata),
        })
    index = out / "manifest.jsonl"
    atomic_jsonl(index, records)
    return index


def cache_rows(index, rows, split, split_name):
    from .data import check_grid

    index = Path(index).resolve()
    expected = {row["case_id"]: row for row in rows if str(row["animal_id"]) in split[split_name]}
    if not expected:
        raise ValueError(f"empty {split_name} split")
    records = {}
    protocol = None
    cohort_hash = canonical_hash(snapshot(rows, split))
    for item in read_jsonl(index):
        case = item["case_id"]
        if case not in expected or case in records:
            raise ValueError(f"{case}: unexpected or duplicate cached case")
        row = expected[case]
        path = resolve(item["path"], index.parent)
        if (sha256(path) != item["sha256"] or row["sha256"] != item["source_sha256"]
            or str(row["animal_id"]) != item["animal_id"]):
            raise ValueError(f"{case}: cached prediction checksum or source differs")
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["metadata_json"])
            if (canonical_hash(meta) != item["metadata_hash"] or meta["case_id"] != case
                or meta["animal_id"] != str(row["animal_id"])
                or meta["source_sha256"] != row["sha256"] or meta["split_name"] != split_name
                or meta["source_contract_hash"] != canonical_hash(row["metadata"])
                or meta["phase_labels"] != row["metadata"]["phase_labels"]
                or item["phase_labels"] != meta["phase_labels"]):
                raise ValueError(f"{case}: cached input, split or phase order differs")
            check_grid(meta["geometry"], row["metadata"]["geometry"])
            if meta["decnn_prior"] != {k: v for k, v in row["decnn_prior"].items() if k != "path"}:
                raise ValueError(f"{case}: cached DeCNN prior differs")
            if set(item["datasets"]) != {"R", "S", "D"} or any(k not in f or f[k].shape != tuple(row["shape"]) for k in ("R", "S", "D")):
                raise ValueError(f"{case}: three aligned prediction arrays are required")
        current = meta["protocol"]
        if (canonical_hash(current) != item["protocol_hash"] or current["cohort"] != cohort_hash
            or current["split_hash"] != canonical_hash(split) or (protocol is not None and protocol != current)):
            raise ValueError(f"{case}: cache protocol differs")
        protocol = current
        records[case] = {"path": path, "row": row, "metadata": meta}
    if set(records) != set(expected):
        raise ValueError(f"cache does not cover {split_name}: missing {sorted(set(expected) - set(records))}")
    return records, protocol
