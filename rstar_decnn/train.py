"""Frozen, direct Stage II, Tetris and warmup/unfreeze training."""
from pathlib import Path
import json
import random
import uuid

import h5py
import numpy as np
import torch
import yaml
from rstar4d.models.model import attach_stam, create_model, initialize_stage2
from rstar4d.models.stam import architecture as stam_architecture
from rstar4d.training.losses import compute_loss
from rstar4d.utils.io import atomic_json, canonical_hash, read_json, sha256

from . import checkpoint
from .backbone import check_case, normalize, physical, restore_backbone, upstream_identity
from .data import load_rows, range_counts, read_case, sample_training, snapshot, units, validate_transform
from .model import Combined
from .predict import decnn_volume, fuse, predict_case


RECIPES = {"frozen": "frozen", "scratch_stage2": "random_joint", "tetris": "stage1",
           "warmup_unfreeze": "preheat"}


def read_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    if config.get("backbone_checkpoint"):
        config["backbone_checkpoint"] = str((path.parent / config["backbone_checkpoint"]).resolve())
    return validate_config(config)


def validate_config(c):
    if c["recipe"] not in RECIPES or c["decnn_prior"] not in {"all_views", "shared", "self"}:
        raise ValueError("unknown training recipe or prior experiment")
    if c["steps"] < 1 or c["validation_every"] < 1 or c["checkpoint_every"] < 1:
        raise ValueError("training and checkpoint intervals must be positive")
    if c["recipe"] == "tetris" and not 0 < c["stage1_steps"] < c["steps"]:
        raise ValueError("Tetris requires Stage I and Stage II updates")
    if c["recipe"] == "warmup_unfreeze" and not 0 < c["warmup_steps"] < c["steps"]:
        raise ValueError("warmup/unfreeze requires both training stages")
    if c["recipe"] in {"frozen", "warmup_unfreeze"} and not c.get("backbone_checkpoint"):
        raise ValueError("this recipe requires an existing Stage II Joint STAM checkpoint")
    if c["recipe"] in {"scratch_stage2", "tetris"} and c.get("backbone_checkpoint"):
        raise ValueError("from-scratch recipes require random initialization")
    if not np.isfinite([c["alpha"], c["decnn_mse_weight"], c["gamma"]]).all() or not (
        0 <= c["alpha"] <= 1 and c["decnn_mse_weight"] > 0 and 0 < c["gamma"] <= 1
    ):
        raise ValueError("invalid alpha, auxiliary loss weight or learning-rate decay")
    if c["stam"]["mode"] != "joint" or c["stam"].get("sam_checkpoint"):
        raise ValueError("provide Joint STAM execution options without external SAM weights")
    for name in ("threads", "phase_batch", "stage1_steps", "warmup_steps",
                 "steps", "validation_every", "checkpoint_every"):
        if type(c[name]) is not int or c[name] < (0 if name.endswith("_steps") else 1):
            raise ValueError(f"invalid {name}")
    expected = {"decnn", "backbone", "stam", "sam", "alpha"}
    if set(c["learning_rates"]) != expected or any(
        not np.isfinite(rate) or rate <= 0 for rate in c["learning_rates"].values()
    ):
        raise ValueError("provide positive learning rates for all named parameter groups")
    if sorted(set(c["milestones"])) != c["milestones"] or any(step < 1 for step in c["milestones"]):
        raise ValueError("milestones must be sorted distinct positive update counts")
    if not c["stage2_patches"]:
        raise ValueError("provide at least one Stage II training patch")
    for patch in c["stage2_patches"]:
        if len(patch) != 3 or any(n < 16 or n % 16 for n in patch):
            raise ValueError("Stage II patches must be real 3D blocks divisible by 16")
    if len(c["stage1_patch"]) != 3 or c["stage1_patch"][0] != 1 or min(c["stage1_patch"][1:]) < 8:
        raise ValueError("Stage I patch requires Z=1 and XY >= 8")
    validate_transform(c["transform"])
    return c


def _source_contract(config, rows):
    meta = rows[0]["metadata"]
    return {
        "model_profile": config["model_profile"],
        **{k: meta[k] for k in ("normalization", "intensity_domain", "calibration_application")},
    }


def _with_stam(spec, config):
    return {
        **spec, "stage": 2, "config": {"stam": config["stam"]},
        "stam": {"architecture": stam_architecture("joint"), "sam_source": {"kind": "random"}},
    }


def _fresh(config, rows, split):
    policy = RECIPES[config["recipe"]]
    spec = _source_contract(config, rows)
    source = {"sam": {"kind": "random"}, "prior_mode": config["prior_mode"]}
    if policy in {"frozen", "preheat"}:
        backbone, spec, source = restore_backbone(
            config["backbone_checkpoint"], joint=True, prior_mode=config["prior_mode"],
            engineering=config["engineering"],
        )
        if spec["model_profile"] != config["model_profile"]:
            raise ValueError("configured model profile differs from original STAM")
        if not config["engineering"]:
            if source.get("split") != split or not source.get("completed_epoch") or not source.get("data_snapshot"):
                raise ValueError("formal STAM checkpoint requires target-domain training provenance and the locked animal split")
    else:
        spec["stage"] = 1 if policy == "stage1" else 2
        backbone = create_model(spec["stage"], spec["model_profile"],
                                config["stam"] if spec["stage"] == 2 else None)
        if spec["stage"] == 2:
            spec = _with_stam(spec, config)
    model = Combined(backbone, config["alpha"], config["upsampling"], decnn=policy != "stage1")
    model.set_policy(policy)
    return model, spec, source


def parameter_groups(model, c):
    groups = {name: [] for name in ("decnn", "backbone", "stam", "sam", "alpha")}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.stam.sam."):
            group = "sam"
        elif name.startswith("backbone.stam."):
            group = "stam"
        else:
            group = name.partition(".")[0]
        groups[group].append(param)
    return [{"name": name, "params": p, "lr": c["learning_rates"][name],
             "initial_lr": c["learning_rates"][name]} for name, p in groups.items() if p]


def optimizer_for(model, c):
    return torch.optim.Adam(parameter_groups(model, c))


def set_learning_rate(optimizer, config, stage_updates):
    factor = config["gamma"] ** sum(step <= stage_updates for step in config["milestones"])
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def transition(model, spec, optimizer, config, step, stage_updates, device):
    if model.policy == "stage1" and step == config["stage1_steps"]:
        backbone = create_model(2, spec["model_profile"])
        initialize_stage2(backbone, model.backbone.state_dict())
        attach_stam(backbone, config["stam"])
        model = Combined(backbone, float(model.alpha.detach()), config["upsampling"]).to(device)
        model.set_policy("random_joint")
        spec = _with_stam(spec, config)
        optimizer = optimizer_for(model, config)
        stage_updates = 0
    elif model.policy == "preheat" and step == config["warmup_steps"]:
        model.set_policy("unfrozen")
        present = {group["name"] for group in optimizer.param_groups}
        for group in parameter_groups(model, config):
            if group["name"] not in present:
                optimizer.add_param_group(group)
        set_learning_rate(optimizer, config, stage_updates)
    return model, spec, optimizer, stage_updates


def _task_loss(pred, target, valid, stage, norm, config):
    params = {**config["loss"], "normalization_scale": norm["scale"], "normalization_offset": norm["offset"]}
    # The public loss scales intensities by 1000 before SSIM's second
    # moments. Even ordinary attenuation targets can overflow fp16 there.
    with torch.autocast(pred.device.type, enabled=False):
        return compute_loss(pred, target, valid, stage, params)["total"]


def _prediction_block(values, batch_count):
    return values.reshape(batch_count, -1, 1, *values.shape[-3:]).transpose(1, 2)


def update(model, batch, spec, optimizer, scaler, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    valid = batch["valid"]
    y = batch["y"]
    transform = config["transform"]
    count = int(valid.sum())
    if count == 0:
        raise ValueError("training patch has no valid voxels")
    stats = range_counts(y, valid, transform) if model.decnn is not None else {}
    if stats.get("below_relu", 0):
        raise ValueError(f"g(Y) falls outside ReLU support: {stats}")
    norm = spec["normalization"]
    amp = torch.autocast(y.device.type, enabled=scaler.is_enabled())
    frozen = model.policy in {"frozen", "preheat"}
    if model.decnn is None:
        with amp:
            p = model.normalized_backbone(batch, norm)
            loss = _task_loss(p, normalize(y, norm), valid, 1, norm, config)
        scaler.scale(loss).backward()
        value = float(loss.detach())
    elif frozen:
        value = 0.0
        for phase, x, prior in model.phase_batches(batch, config["phase_batch"]):
            mask = valid[:, :, phase]
            if not bool(mask.any()):
                continue
            if config["decnn_prior"] == "self":
                prior = x
            with amp:
                raw = model.normalized_decnn(x, prior, transform)
                raw = _prediction_block(raw.float(), y.shape[0])
                diff = (raw - normalize(y[:, :, phase], transform))[mask]
                mse = diff.square().sum() / count
            scaler.scale(mse).backward()
            value += float(mse.detach())
    else:
        outputs = []
        with amp:
            for _, x, prior in model.phase_batches(batch, config["phase_batch"]):
                if config["decnn_prior"] == "self":
                    prior = x
                raw = model.normalized_decnn(x, prior, transform, config["decnn_recompute"])
                outputs.append(_prediction_block(raw.float(), y.shape[0]))
            d = torch.cat(outputs, dim=2)
            s = model.normalized_backbone(batch, norm).float()
            pred = fuse(physical(s, norm), physical(d, transform), model.alpha)
            loss = _task_loss(normalize(pred, norm), normalize(y, norm), valid, 2, norm, config)
            loss = loss + config["decnn_mse_weight"] * (d - normalize(y, transform))[valid].square().mean()
        scaler.scale(loss).backward()
        value = float(loss.detach())
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite training loss")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    succeeded = scaler.get_scale() >= previous_scale
    if succeeded:
        with torch.no_grad():
            model.alpha.clamp_(0, 1)
    return value, stats, succeeded


def inference_options(config, device):
    options = config["validation_inference"]
    common = {"mode": options["mode"], "amp": config["amp"], "device": device}
    return ({**common, "tile": options["backbone_tile"], "overlap": options["backbone_overlap"]},
            {**common, "tile": options["decnn_tile"], "overlap": options["decnn_overlap"]})


def _validate(model, spec, rows, split, config, device):
    backbone_options, decnn_options = inference_options(config, device)
    animals = {}
    for row in rows:
        animal = str(row["animal_id"])
        if animal not in split["val"]:
            continue
        sample = read_case(row)
        transform = config["transform"]
        stats = range_counts(sample["y"], sample["valid"], transform)
        if stats["below_relu"] or not stats["valid_voxels"]:
            raise ValueError(f"{row['case_id']}: invalid validation support: {stats}")
        if config["decnn_prior"] == "self":
            sample["a_d"] = sample["x"]
        if model.policy in {"frozen", "preheat"}:
            options = {**decnn_options}
            p = decnn_volume(model.decnn, sample["x"], sample["a_d"], transform, **options)
            error = (normalize(p, transform) - normalize(sample["y"], transform))[sample["valid"]]
            score = float(np.square(error.astype(np.float64)).mean())
        else:
            p = predict_case(model, spec, sample, transform, alpha=model.alpha.detach(),
                             backbone_options=backbone_options, decnn_options=decnn_options)
            score = float(np.abs((p - sample["y"])[sample["valid"]]).mean(dtype=np.float64))
        animals.setdefault(animal, []).append(score)
    if not animals:
        raise ValueError("empty validation split")
    return float(np.mean([np.mean(scores) for scores in animals.values()]))


def _verify_dataset(rows, split, config):
    if not split["train"] or not split["val"]:
        raise ValueError("training requires nonempty animal train and validation splits")
    if not config["engineering"]:
        seen = {}
        for row in rows:
            animal = str(row["animal_id"])
            hashes = {row["sha256"], *row["metadata"]["source_hashes"].values()}
            for item in hashes:
                if item in seen and animal != seen[item]:
                    raise ValueError(f"{row['case_id']}: duplicate input assigned to different animals")
                seen[item] = animal
    for row in rows:
        if row["backbone_prior_mode"] != config["prior_mode"] or units(row["metadata"]) != config["transform"]["unit"]:
            raise ValueError(f"{row['case_id']}: prior mode or intensity unit differs")
        if row["engineering"] != config["engineering"]:
            raise ValueError(f"{row['case_id']}: data engineering status differs")
        if config["decnn_prior"] != "self" and ((row["decnn_prior"]["kind"] == "shared") != (config["decnn_prior"] == "shared")):
            raise ValueError(f"{row['case_id']}: requested prior ablation differs from prepared source")
        if str(row["animal_id"]) in split["train"] + split["val"]:
            with h5py.File(row["path"], "r") as f:
                if "target" not in f:
                    raise ValueError(f"{row['case_id']}: train and val require paired targets")
    training_rows = [row for row in rows if str(row["animal_id"]) in split["train"]]
    for row in training_rows:
        phases = row["shape"][0]
        group_sizes = [min(config["phase_batch"], phases - i) for i in range(0, phases, config["phase_batch"])]
        for patch in config["stage2_patches"]:
            if any(n > limit for n, limit in zip(patch, row["shape"][1:])):
                raise ValueError(f"{row['case_id']}: requested real 3D patch exceeds scan dimensions")
            if min(group_sizes) * int(np.prod(np.array(patch) // 16)) <= 1:
                raise ValueError(f"{row['case_id']}: every phase microbatch needs at least two bottleneck values")


def run_train(config, manifest, split, out, *, resume=None, stop_after=None):
    config = validate_config(config)
    out = Path(out).resolve()
    rows, split = load_rows(manifest, split)
    _verify_dataset(rows, split, config)
    data = snapshot(rows, split)
    signature = canonical_hash({"config": config, "data": data})
    device = torch.device(config["device"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"] and device.type == "cuda")
    upstream = upstream_identity()
    run_file, last = out / "run.json", out / "last.pt"

    if resume is None:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("new training output directory must be empty")
        model, spec, source = _fresh(config, rows, split)
        model.to(device)
        optimizer = optimizer_for(model, config)
        state = {
            "run_id": uuid.uuid4().hex, "step": 0, "stage_updates": 0, "best": None,
            "range_counts": {}, "history": [], "signature": signature,
        }
    else:
        saved = checkpoint.load(resume, training=True)
        state = saved["training"]
        if (state["signature"] != signature or saved["data"] != data or saved["config"] != config
            or not (out / "run.json").is_file() or read_json(run_file)["run_id"] != state["run_id"]
            or last.resolve() != Path(resume).resolve()):
            raise ValueError("resume requires the original run directory, last checkpoint, data and config")
        if state["best"] is not None:
            best_path = out / state["best"]["path"]
            if not best_path.is_file() or sha256(best_path) != state["best"]["sha256"]:
                raise ValueError("resume requires the verified historical best artifact")
        model = checkpoint.model_from(saved, device)
        spec, source = saved["backbone"], saved["source"]
        optimizer = optimizer_for(model, config)
        if [g["name"] for g in optimizer.param_groups] != state["groups"]:
            raise ValueError("optimizer groups differ from saved phase")
        optimizer.load_state_dict(state.pop("optimizer"))
        scaler.load_state_dict(state.pop("scaler"))
        checkpoint.restore_rng(state.pop("rng"), rng, device)
        if state["scheduler"]["stage_updates"] != state["stage_updates"]:
            raise ValueError("scheduler state differs from training stage")
        del saved

    for row in rows:
        check_case(spec, row)
    if resume is None:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(run_file, {"run_id": state["run_id"], "signature": signature, "config": config,
                               "data": data, "upstream": upstream, "source": source})

    def save_last():
        an = checkpoint.payload(model, spec, config, data, upstream, source, model.policy)
        an["training"] = {
            **state, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "groups": [g["name"] for g in optimizer.param_groups], "rng": checkpoint.rng_state(rng, device),
            "scheduler": {"stage_updates": state["stage_updates"], "milestones": config["milestones"],
                          "gamma": config["gamma"], "learning_rates": [g["lr"] for g in optimizer.param_groups]},
            "transitions": {
                "stage2_initialized": spec["stage"] == 2,
                "unfreeze_applied": model.policy == "unfrozen",
            },
        }
        checkpoint.publish(last, an)

    final = config["steps"] if stop_after is None else min(config["steps"], int(stop_after))
    if not 0 <= state["step"] <= final:
        raise ValueError("stop step precedes the restored training state")
    failures = 0
    while state["step"] < final:
        old = model.policy
        model, spec, optimizer, state["stage_updates"] = transition(
            model, spec, optimizer, config, state["step"], state["stage_updates"], device,
        )
        if old != model.policy:
            state["best"] = None
            state["history"].append({"step": state["step"], "transition": model.policy,
                                     "alpha": float(model.alpha.detach())})
            save_last()
        if model.policy == "stage1":
            patch = config["stage1_patch"]
        else:
            patch = config["stage2_patches"][int(rng.integers(len(config["stage2_patches"])))]
        batch = sample_training(rows, split, patch, rng, device, flips=config["flips"])
        loss, counts, succeeded = update(model, batch, spec, optimizer, scaler, config)
        if not succeeded:
            failures += 1
            if failures >= 16:
                raise FloatingPointError("16 consecutive AMP optimizer overflows")
            continue
        failures = 0
        for key, count in counts.items():
            state["range_counts"][key] = state["range_counts"].get(key, 0) + count
        state["step"] += 1
        state["stage_updates"] += 1
        step = state["step"]
        set_learning_rate(optimizer, config, state["stage_updates"])
        validation = model.decnn is not None and (step % config["validation_every"] == 0 or step == config["steps"])
        if validation:
            score = _validate(model, spec, rows, split, config, device)
            event = {"step": step, "policy": model.policy, "validation": score, "last_training_loss": loss,
                     "alpha": float(model.alpha.detach())}
            state["history"].append(event)
            print(json.dumps(event), flush=True)
            if state["best"] is None or score < state["best"]["validation"]:
                path = out / f"best-{model.policy}-{step:07}.pt"
                checkpoint.publish(path, checkpoint.payload(model, spec, config, data, upstream, source, model.policy))
                state["best"] = {"validation": score, "path": path.name, "sha256": sha256(path)}
        if validation or step == final or step % config["checkpoint_every"] == 0:
            save_last()
    if not last.exists():
        save_last()
    summary = {"last": str(last), "step": state["step"], "policy": model.policy,
               "best": str(out / state["best"]["path"]) if state["best"] else None,
               "range_counts": state["range_counts"], "history": state["history"]}
    atomic_json(out / "summary.json", summary)
    return summary
