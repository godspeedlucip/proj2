"""Prepare, train, and predict paired phase1/phase5 CT cases."""
import argparse
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from nnunetv2.evaluation.paired_tam import evaluate_pairs, paired_patient_deltas
from nnunetv2.inference.paired_tam import PairedTAMPredictor, check_prediction_destinations
from nnunetv2.preprocessing.paired_tam import prepare_dataset, read_manifest, validate_sources, validate_split
from nnunetv2.training.nnUNetTrainer import nnUNetTrainerTAM as trainers
from nnunetv2.utilities.paired_tam_contract import PHASES, TRAINER_PRESETS, json_digest


TRAINERS = {preset: getattr(trainers, name) for name, preset in TRAINER_PRESETS.items()}


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("Expected a 32-bit unsigned seed.")
    return number


def add_fold(parser):
    parser.add_argument("--fold", type=int, default=0)


def add_inference_options(parser):
    add_fold(parser)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--save-probabilities", action="store_true")


def add_training_options(parser, seed=None):
    add_fold(parser)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=positive_integer, default=1000)
    parser.add_argument("--train-iterations", type=positive_integer, default=250)
    parser.add_argument("--val-iterations", type=positive_integer, default=50)
    parser.add_argument("--seed", type=seed_value, default=seed)


@contextmanager
def seeded_augmentation(seed):
    # Worker queues are asynchronous. A single process keeps augmentation order paired across arms.
    previous = os.environ.get("nnUNet_n_proc_DA")
    if seed is not None:
        os.environ["nnUNet_n_proc_DA"] = "0"
    try:
        yield
    finally:
        if seed is not None:
            if previous is None:
                os.environ.pop("nnUNet_n_proc_DA", None)
            else:
                os.environ["nnUNet_n_proc_DA"] = previous


def stage_reference_dataset(args):
    """Expose only the training patients to the ordinary nnU-Net planner."""
    records = read_manifest(args.manifest, require_labels=True)
    split = validate_split(trainers.read_json(args.splits), records, args.fold)
    dataset_json = trainers.read_json(args.dataset_json)
    if not re.fullmatch(r"Dataset\d{3}_.+", args.raw_dataset.name):
        raise ValueError("Reference dataset folder must have the nnU-Net DatasetNNN_Name naming format.")
    dataset_json["dataset"] = {
        f"{case_id}_{phase}": {
            "images": [records[case_id][phase]["image"]], "label": records[case_id][phase]["label"],
        }
        for case_id in split["train"] for phase in PHASES
    }
    dataset_json["numTraining"] = len(dataset_json["dataset"])
    sources = [path for entry in dataset_json["dataset"].values()
               for path in (*entry["images"], entry["label"])]
    if os.name == "nt" and dataset_json["file_ending"].lower() in (".nii", ".nii.gz") and any(
            not path.isascii() for path in sources):
        dataset_json["overwrite_image_reader_writer"] = "NibabelIO"
    args.raw_dataset.mkdir(parents=True, exist_ok=True)
    with (args.raw_dataset / "dataset.json").open("x", encoding="utf-8") as file:
        json.dump(dataset_json, file, indent=2)


def train(args):
    with seeded_augmentation(args.seed):
        return train_preset(args)


def find_checkpoint(model, fold):
    folder = model / f"fold_{fold}"
    checkpoints = [folder / f"checkpoint_{kind}.pth" for kind in ("final", "latest", "best")]
    return max((path for path in checkpoints if path.is_file()),
               key=lambda path: path.stat().st_mtime_ns, default=None)


def train_preset(args):
    trainer = TRAINERS[args.preset].from_prepared(args.prepared, args.model, args.fold, torch.device(args.device))
    trainer.num_epochs = args.epochs
    trainer.num_iterations_per_epoch = args.train_iterations
    trainer.num_val_iterations_per_epoch = args.val_iterations
    trainer.training_seed = args.seed
    checkpoint = find_checkpoint(args.model, args.fold)
    if args.resume:
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint to resume in {trainer.output_folder}.")
        trainer.load_checkpoint(checkpoint)
        if trainer.current_epoch > trainer.num_epochs:
            raise ValueError("The requested epoch count precedes the checkpoint epoch.")
    elif checkpoint is not None:
        raise FileExistsError(f"Existing checkpoint {checkpoint}. Use --resume or choose a new model directory.")
    if trainer.current_epoch < trainer.num_epochs:
        trainer.run_training()
    trainer.perform_actual_validation(args.save_probabilities)
    return trainer


def ablate(args):
    folder = args.output / f"fold_{args.fold}"
    recipe = {
        "fold": args.fold, "seed": args.seed, "device": args.device,
        "epochs": args.epochs, "train_iterations": args.train_iterations, "val_iterations": args.val_iterations,
        "plans_sha256": json_digest(trainers.read_json(args.prepared / "plans.json")),
        "dataset_sha256": json_digest(trainers.read_json(args.prepared / "dataset.json")),
        "splits_sha256": json_digest(trainers.read_json(args.prepared / "splits_final.json")),
        "fingerprint_sha256": json_digest(trainers.read_json(args.prepared / "dataset_fingerprint.json")),
    }
    recipe_path = folder / "ablation_recipe.json"
    folder.mkdir(parents=True, exist_ok=True)
    if recipe_path.exists():
        if trainers.read_json(recipe_path) != recipe:
            raise ValueError("Ablation recipe differs. Use a new output directory.")
        if not args.resume:
            raise FileExistsError(f"Ablation already started at {folder}. Use --resume.")
    else:
        with recipe_path.open("x", encoding="utf-8") as stream:
            json.dump(recipe, stream, indent=2)

    with seeded_augmentation(args.seed):
        for preset in TRAINERS:
            model = folder / preset
            settings = vars(args) | {"preset": preset, "model": model,
                                     "resume": args.resume and find_checkpoint(model, args.fold) is not None}
            train_preset(argparse.Namespace(**settings))
            if args.device == "cuda":
                torch.cuda.empty_cache()

    arms = {}
    patient_rows = {}
    for preset in TRAINERS:
        model = folder / preset
        predictor = PairedTAMPredictor.from_model_folder(
            model, args.fold, "checkpoint_final.pth", torch.device(args.device),
        )
        output = model / f"fold_{args.fold}" / "test"
        summary = evaluate(argparse.Namespace(prepared=args.prepared, fold=args.fold, output=output,
                                               split="test", save_probabilities=args.save_probabilities), predictor)
        patient_rows[preset] = summary["per_patient"]
        arms[preset] = {
            "settings": predictor.checkpoint_metadata["paired_tam"],
            "parameter_count": sum(parameter.numel() for parameter in predictor.network.parameters()),
            "test_summary": str((output / "summary.json").relative_to(folder)),
            "by_phase_and_structure": summary["by_phase_and_structure"],
            "both_phases_by_structure": summary["both_phases_by_structure"],
        }
        del predictor
    report = {
        "recipe": recipe, "arms": arms,
        "paired_deltas_vs_C1": {preset: paired_patient_deltas(patient_rows["C1"], patient_rows[preset])
                                for preset in TRAINERS if preset != "C1"},
        "paired_deltas_by_added_stage": {
            "C4_minus_C3": paired_patient_deltas(patient_rows["C3"], patient_rows["C4"]),
            "C5_minus_C4": paired_patient_deltas(patient_rows["C4"], patient_rows["C5"]),
        },
    }
    with (folder / "ablation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def evaluate(args, predictor):
    records = read_manifest(args.prepared / "paired_manifest.json", require_labels=True)
    split = validate_split(trainers.read_json(args.prepared / "splits_final.json"), records, args.fold)
    fingerprint = trainers.read_json(args.prepared / "dataset_fingerprint.json")
    validate_sources(records, fingerprint)
    preparation = {"manifest": records, "split": split, "fingerprint": fingerprint}
    expected = predictor.checkpoint_metadata["paired_tam_training"]
    if (not expected or json_digest(preparation) != expected["preparation_sha256"]
            or json_digest(trainers.read_json(args.prepared / "plans.json"))
            != predictor.checkpoint_metadata["paired_tam"]["plans_sha256"]
            or json_digest(trainers.read_json(args.prepared / "dataset.json"))
            != predictor.checkpoint_metadata["paired_tam"]["dataset_sha256"]):
        raise ValueError("Evaluation requires the same prepared fold as training.")
    output = args.output or (args.model / f"fold_{args.fold}" /
                             ("validation" if args.split == "val" else "test"))
    summary = evaluate_pairs(predictor, [records[key] for key in split[args.split]],
                             args.split, output, args.save_probabilities, source_records=records.values())
    print(json.dumps(summary["by_phase_and_structure"], indent=2))
    return summary


def profile(args):
    measurements = {}
    presets = TRAINERS if args.preset == "all" else [args.preset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for preset in presets:
        with TemporaryDirectory(dir=args.output.parent, prefix=".tam-profile-") as temp_model:
            trainer = TRAINERS[preset].from_prepared(args.prepared, temp_model, args.fold, torch.device(args.device))
            trainer.initialize()
            train_loader, val_loader = trainer.get_dataloaders()
            try:
                measurements[preset] = {
                    "settings": trainer.checkpoint_contract(),
                    "parameter_count": sum(parameter.numel() for parameter in trainer.network.parameters()),
                    **trainer.profile_training_step(next(train_loader), args.steps),
                }
                trainer.set_deep_supervision_enabled(False)
                predictor = trainer.make_predictor()
                record = trainer.records[trainer.split["val"][0]]
                predictor.predict_case(record, temp_model)
            finally:
                for loader in (train_loader, val_loader):
                    if hasattr(loader, "_finish"):
                        loader._finish()
            del trainer, predictor
            if args.device == "cuda":
                torch.cuda.empty_cache()
    with args.output.open("w", encoding="utf-8") as file:
        device = torch.cuda.get_device_name(0) if args.device == "cuda" else args.device
        json.dump({"device": device, "presets": measurements}, file, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    reference = commands.add_parser("reference-dataset", help="Stage only training patients for the nnU-Net planner.")
    for name in ("manifest", "splits", "dataset_json", "raw_dataset"):
        reference.add_argument(name, type=Path)
    add_fold(reference)

    prep = commands.add_parser("prepare", help="Freeze CT statistics and preprocess one patient fold.")
    for name in ("manifest", "splits", "plans", "dataset_json", "output"):
        prep.add_argument(name, type=Path)
    add_fold(prep)

    training = commands.add_parser("train", help="Train and validate a paired TAM preset.")
    training.add_argument("prepared", type=Path)
    training.add_argument("model", type=Path)
    training.add_argument("--preset", choices=TRAINERS, required=True)
    add_training_options(training)

    ablation = commands.add_parser("ablate", help="Run C1/C3/C4/C5 and compare paired held-out patients.")
    ablation.add_argument("prepared", type=Path)
    ablation.add_argument("output", type=Path)
    add_training_options(ablation, seed=12345)

    predict = commands.add_parser("predict", help="Export two native-grid predictions per manifest case.")
    for name in ("model", "manifest", "output"):
        predict.add_argument(name, type=Path)
    add_inference_options(predict)

    score = commands.add_parser("evaluate", help="Report native-grid Dice and HD95 for validation or test patients.")
    score.add_argument("prepared", type=Path)
    score.add_argument("model", type=Path)
    score.add_argument("--split", choices=("val", "test"), default="test")
    score.add_argument("--output", type=Path)
    add_inference_options(score)

    probe = commands.add_parser("profile", help="Measure full training steps and check paired sliding-window inference.")
    probe.add_argument("prepared", type=Path)
    probe.add_argument("output", type=Path)
    probe.add_argument("--preset", choices=[*TRAINERS, "all"], default="C5")
    probe.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    probe.add_argument("--steps", type=positive_integer, default=3)
    add_fold(probe)

    args = parser.parse_args(argv)
    if args.fold < 0:
        parser.error("--fold must be non-negative.")
    if getattr(args, "device", None) == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment. CPU is supported for smoke tests.")

    if args.command == "reference-dataset":
        stage_reference_dataset(args)
    elif args.command == "prepare":
        prepare_dataset(args.manifest, args.splits, args.plans, args.dataset_json, args.output, args.fold)
    elif args.command == "train":
        train(args)
    elif args.command == "ablate":
        ablate(args)
    elif args.command == "profile":
        profile(args)
    else:
        predictor = PairedTAMPredictor.from_model_folder(
            args.model, args.fold, args.checkpoint, torch.device(args.device),
        )
        if args.command == "evaluate":
            evaluate(args, predictor)
        else:
            records = read_manifest(args.manifest)
            check_prediction_destinations(records, records.values(), args.output,
                                          predictor.dataset_json["file_ending"], args.save_probabilities)
            for record in records.values():
                predictor.predict_case(record, args.output, args.save_probabilities)


if __name__ == "__main__":
    main()
