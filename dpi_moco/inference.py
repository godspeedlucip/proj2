from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import atomic_path, file_digest, fingerprint, network_units, physical_units, read_json, software, write_json
from .data import load_array, load_manifest
from .losses import warp_pull
from .models import RegisNet3D, RestoreNet2D
from .prepare import prepare_scan, roi_mean, save_array


def select_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported training and inference devices are CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    return device


def weights_contract(config, stage):
    keys = {"restore": ("intensity", "critic_profile", "pigae_mode"),
            "register": ("intensity", "regis_variant", "pigae_mode"),
            "denoise": ("intensity", "critic_profile", "gamma")}[stage]
    return {key: config[key] for key in keys}


def load_weights(path, stage, config, device, *, domain, restore_id=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = weights_contract(config, stage)
    if payload["stage"] != stage or payload["contract"] != expected or payload["domain"] != domain:
        raise ValueError(f"Weight stage, domain or training configuration mismatch: {path}")
    if stage == "register" and payload["restore_sha256"] != restore_id:
        raise ValueError("Register weights were trained against a different RestoreNet")
    model = RegisNet3D() if stage == "register" else RestoreNet2D()
    model.load_state_dict(payload["model"])
    return model.to(device).eval().requires_grad_(False)


def tensor_volume(volume, device):
    return torch.from_numpy(np.array(volume, dtype=np.float32, copy=True)).to(device)[None, None]


@torch.inference_mode()
def apply_slicewise(model, volume, config, device, batch_size):
    output = np.empty(volume.shape, dtype=np.float32)
    model.eval()
    for start in range(0, len(volume), batch_size):
        patch = torch.from_numpy(np.array(volume[start:start + batch_size], dtype=np.float32, copy=True))
        pixels = network_units(patch[:, None].to(device), config)
        output[start:start + batch_size] = physical_units(model(pixels)[:, 0], config).cpu().numpy()
    return output


def enhanced_paths(case_root, case, model, restore_id, config, device, batch_size):
    enhancement_id = fingerprint({"source": case["cache_id"], "restore": restore_id,
                                  "intensity": config["intensity"], "device": device.type,
                                  "torch": str(torch.__version__), "batch_size": batch_size})
    base = case_root / "enhanced"
    legacy = base / enhancement_id
    root = legacy if (legacy / "phase_00.npy").is_file() else base / enhancement_id[:24]
    paths = []
    for phase in case["branches"]["real"]:
        path = root / f"phase_{phase['phase_id']:02d}.npy"
        if not path.is_file():
            volume = load_array(case_root / phase["c"])
            save_array(path, apply_slicewise(model, volume, config, device, batch_size))
        paths.append(path)
    return paths


def noise_seed(scan_id, phase, seed):
    value = f"{seed}\0{scan_id}\0{phase}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def add_volume_noise(volume, gamma, seed):
    rng = np.random.default_rng(seed)
    return volume + np.float32(gamma) * rng.standard_normal(volume.shape, dtype=np.float32)


@torch.inference_mode()
def fuse_target(register, c_paths, b_paths, target, config, device, on_pair=None, alternatives=None):
    fixed = network_units(tensor_volume(load_array(b_paths[target]), device), config)
    accumulated = tensor_volume(load_array(c_paths[target]), device)
    alternatives = alternatives or {}
    other_sums = {name: tensor_volume(load_array(paths[target]), device) for name, paths in alternatives.items()}
    zero_motion = np.array(load_array(c_paths[target])) if alternatives else None
    for source, corrected in enumerate(c_paths):
        if source == target:
            continue
        moving = network_units(tensor_volume(load_array(b_paths[source]), device), config)
        field = register(moving, fixed)
        corrected_volume = load_array(corrected)
        warped = warp_pull(tensor_volume(corrected_volume, device), field)
        accumulated.add_(warped)
        if alternatives:
            zero_motion += corrected_volume
            for name, paths in alternatives.items():
                other_sums[name].add_(warp_pull(tensor_volume(load_array(paths[source]), device), field))
        if on_pair is not None:
            on_pair(source, field[0].cpu().numpy(), warped[0, 0].cpu().numpy())
    if config["fusion_mode"] == "mean":
        accumulated.div_(len(c_paths))
        for volume in other_sums.values():
            volume.div_(len(c_paths))
        if alternatives:
            zero_motion /= len(c_paths)
    outputs = {name: volume[0, 0].cpu().numpy() for name, volume in other_sums.items()}
    if alternatives:
        outputs["d_zero_motion"] = zero_motion
    return accumulated[0, 0].cpu().numpy(), outputs


def field_statistics(field, grid):
    spacing = np.asarray(grid["spacing_xyz_mm"], dtype=np.float32)
    magnitudes = np.sqrt(sum(np.square(component * scale) for component, scale in zip(field, spacing)))
    percentile = np.percentile(magnitudes, [5, 50, 95, 99]).tolist()
    depth = field.shape[1]
    folds = 0
    for start in range(0, depth, 8):
        stop = min(depth, start + 8)
        lower, upper = max(0, start - 1), min(depth, stop + 1)
        slab = field[:, lower:upper]
        jacobian = np.empty((*slab.shape[1:], 3, 3), dtype=np.float32)
        for component in range(3):
            for coord, axis in enumerate((2, 1, 0)):
                if slab.shape[axis + 1] > 1:
                    jacobian[..., component, coord] = np.gradient(slab[component], axis=axis)
                else:
                    jacobian[..., component, coord] = 0
                if component == coord:
                    jacobian[..., component, coord] += 1
        folds += np.count_nonzero(np.linalg.det(jacobian[start - lower:stop - lower]) <= 0)
    return {"displacement_mm_percentiles": dict(zip(("p05", "p50", "p95", "p99"), percentile)),
            "fold_fraction": float(folds / magnitudes.size)}


def save_multiplanar(folder, volume, window):
    low, high = window
    for axis, name in enumerate(("axial", "coronal", "sagittal")):
        image = np.take(volume, volume.shape[axis] // 2, axis=axis)
        image = (np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        with atomic_path(folder / f"{name}.png") as temporary:
            Image.fromarray(image).save(temporary, format="PNG")


def infer(args, config):
    device = select_device(args.device)
    scans = list(load_manifest(args.manifest))
    domains = {scan["domain"] for scan in scans}
    if len(domains) != 1:
        raise ValueError("Use one scanner domain per inference run with its corresponding weights")
    domain = next(iter(domains))
    restore_id = file_digest(args.restore)
    weight_ids = {"restore": restore_id, "register": file_digest(args.register),
                  "denoise": file_digest(args.denoise)}
    output = Path(args.out).resolve()
    scan_records = []
    restore, register, denoise = (load_weights(getattr(args, stage), stage, config, device,
                                              domain=domain, restore_id=restore_id)
                                  for stage in ("restore", "register", "denoise"))
    for scan in scans:
        case_file = prepare_scan(scan, config, output / "preparation", training=False)
        case = read_json(case_file)
        case_root = case_file.parent
        phases = case["branches"]["real"]
        b_paths = enhanced_paths(case_root, case, restore, restore_id, config, device, args.slice_batch)
        c_paths = [case_root / phase["c"] for phase in phases]
        x_paths = [case_root / phase["x"] for phase in phases]
        alternatives = {"d_from_b": b_paths, "d_from_x": x_paths} if args.motion_ablation else None
        root = output / scan["scan_id"]
        pi = load_array(case_root / case["pi"])
        pi_roi = roi_mean(pi)
        if args.keep_intermediates:
            save_array(root / "pi.npy", pi)
        prediction_records = []
        for target, phase in enumerate(phases):
            folder = root / f"phase_{target:02d}"
            metadata = {"phase_id": target, "units": "mm^-1", "grid": scan["grid"]}
            pair_stats = []

            def save_pair(source, field, warped):
                pair = folder / "pairs" / f"from_{source:02d}"
                if args.keep_fields:
                    save_array(pair / "field.npy", field)
                    save_array(pair / "warped_c.npy", warped)
                    write_json(pair / "metadata.json", {"source_phase": source, "target_phase": target,
                               "components": ["dx", "dy", "dz"], "direction": "target-to-source pull",
                               "units": "voxels", "grid": scan["grid"]})
                if args.motion_qc:
                    pair_stats.append({"source_phase": source, **field_statistics(field, scan["grid"])})

            fused, ablations = fuse_target(register, c_paths, b_paths, target, config, device,
                                           on_pair=save_pair if args.keep_fields or args.motion_qc else None,
                                           alternatives=alternatives)
            seed = noise_seed(scan["scan_id"], target, config["seed"])
            noisy = add_volume_noise(fused, config["gamma"], seed)
            prediction = apply_slicewise(denoise, noisy, config, device, args.slice_batch)
            save_array(folder / "dpri.npy", prediction)
            for name, volume in ablations.items():
                save_array(folder / f"{name}.npy", volume)
            if args.motion_qc:
                save_multiplanar(folder, prediction, case["slice_display"]["window"])
            if args.keep_intermediates:
                for name, source in (("x", case_root / phase["x"]), ("c", c_paths[target]),
                                     ("b", b_paths[target])):
                    save_array(folder / f"{name}.npy", load_array(source))
                save_array(folder / "d.npy", fused)
                save_array(folder / "d_noisy.npy", noisy)
            fused_roi = roi_mean(fused)
            write_json(folder / "metadata.json", {**metadata, "noise_seed": seed, "pi_roi_mean": pi_roi,
                       "c_roi_mean": phase["stats"]["c"]["roi_mean"], "d_roi_mean": fused_roi,
                       "d_over_pi_roi_mean": fused_roi / pi_roi if pi_roi else None,
                       "dpri_roi_mean": roi_mean(prediction),
                       "display_window": case["slice_display"]["window"] if args.motion_qc else None,
                       "motion_qc": pair_stats if args.motion_qc else None})
            prediction_records.append({**metadata, "path": (folder / "dpri.npy").relative_to(output).as_posix()})
            print(f"{scan['scan_id']}: inferred phase {target:02d}", flush=True)
        scan_records.append({**{key: scan[key] for key in ("scan_id", "patient_id", "domain")},
                             "phases": prediction_records})
    write_json(output / "predictions.json", {"config": config, "weight_sha256": weight_ids,
                                            "software": software(device), "scans": scan_records})
