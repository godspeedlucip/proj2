from pathlib import Path

import yaml

from .data import atomic_path, run_lock, split_scans
from .settings import dependencies
from ..config import load_config


def run_moco(args):
    command = args.command
    if command == "lock-moco":
        from .evaluate import lock_moco
        return lock_moco(args.validation, args.out)
    with run_lock(args.out):
        return run_directory_command(args)


def run_directory_command(args):
    command = args.command
    if command == "split-moco":
        return split_scans(args.manifest, args.scan_split, args.out)
    if command == "evaluate-moco":
        from .evaluate import evaluate_moco
        return evaluate_moco(args.predictions, args.manifest, args.metric_config, args.out,
                             args.split_name, args.test_lock)
    config = load_config(args.config, "moco")
    if args.device:
        config["device"] = args.device
    sources, _, _ = dependencies(config)
    effective = Path(args.out) / "effective.yaml"
    if effective.exists() and yaml.safe_load(effective.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory contains a different effective MoCo configuration")
    with atomic_path(effective) as temporary:
        temporary.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if args.dry_run:
        return {"effective_config": str(effective.resolve()), "inference_dependencies": sorted(sources)}
    if command == "train-register":
        from .train import train_registration
        result = train_registration(config, args.manifest, args.val_manifest, args.out, baseline=args.baseline,
                                    val_baseline=args.val_baseline, baseline_checkpoint=args.baseline_checkpoint,
                                    resume=args.resume, stop_after_epoch=args.stop_after_epoch)
        return {key: value for key, value in result.items() if key != "history"}
    if args.manifest:
        from .guard import enforce_split
        enforce_split(args.manifest, args.split_name, args.test_lock, config=config,
                      checkpoint=getattr(args, "register_checkpoint", None),
                      baseline_checkpoint=getattr(args, "baseline_checkpoint", None),
                      baseline=getattr(args, "baseline", None), command=command)
    if command == "prepare-moco":
        from .prepare import prepare_moco
        result = prepare_moco(config, args.manifest, args.out, raw_manifest=args.raw_manifest, external=args.external_c)
        return {key: value for key, value in result.items() if key != "cases"}
    if not args.manifest:
        raise ValueError("infer-moco needs --manifest")
    from .infer import infer_moco
    result = infer_moco(config, args.manifest, args.out, baseline=args.baseline,
                        baseline_checkpoint=args.baseline_checkpoint, correction=args.correction,
                        register_checkpoint=args.register_checkpoint, field_cache=args.field_cache)
    return {key: value for key, value in result.items() if key != "cases"}
