"""RSTAR-DeCNN command-line workflow."""
import argparse
import json
from pathlib import Path

from rstar4d.utils.io import read_json

from .data import prepare
from .evaluate import calibrate, evaluate, settings
from .predict import run_predict
from .train import read_config, run_train


def main(argv=None):
    cli = argparse.ArgumentParser(prog="python -m rstar_decnn")
    commands = cli.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="prepare the two FDK priors from immutable scans")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True, help="prepared JSONL manifest path")
    prep.add_argument("--prior-mode", default="phase_weighted_fdk")
    prep.add_argument("--prior-index", type=Path)
    prep.add_argument("--shared-prior", action="store_true")
    prep.add_argument("--raw-manifest", type=Path)
    prep.add_argument("--cycle-csv", type=Path)
    prep.add_argument("--simulation-seed", type=int)
    prep.add_argument("--engineering", action="store_true")

    def common(parser):
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--split", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)

    train = commands.add_parser("train", help="train or resume any of the four recipes")
    common(train)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--stop-after", type=int)

    predict = commands.add_parser("predict", help="predict fused volumes or cache separate branches")
    common(predict)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--split-name", choices=("train", "val", "test", "all"), default="test")
    predict.add_argument("--rstar", type=Path, help="matched ordinary RSTAR Stage II checkpoint")
    predict.add_argument("--baseline-design", type=Path, help="JSON baseline training provenance")
    predict.add_argument("--cache", action="store_true")
    predict.add_argument("--alpha", type=float)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--amp", action="store_true")
    predict.add_argument("--mode", choices=("full", "tiled"), default="tiled")
    predict.add_argument("--tile", type=int, nargs=3, default=(32, 128, 128))
    predict.add_argument("--overlap", type=int, nargs=3, default=(16, 64, 64))
    predict.add_argument("--decnn-tile", type=int, nargs=3)
    predict.add_argument("--decnn-overlap", type=int, nargs=3)

    def metrics(parser):
        common(parser)
        parser.add_argument("--cache", type=Path, action="append", required=True, help="one index per paired seed")
        parser.add_argument("--annotations", type=Path, required=True)
        parser.add_argument("--extractor", type=Path, help="frozen TorchScript binary LV contour model")
        parser.add_argument("--device", default="cpu")

    cal = commands.add_parser("calibrate", help="lock separate RSTAR and RSTAR-STAM coefficients on validation")
    metrics(cal)
    cal.add_argument("--motion-tolerance-ml", type=float, default=0)
    cal.add_argument("--minimum-motion-animals", type=int, default=1)
    cal.add_argument("--window", type=float, nargs=2, default=(0, 0.05))
    holdout = commands.add_parser("evaluate", help="evaluate locked coefficients on held-out animals")
    metrics(holdout)
    holdout.add_argument("--lock", type=Path, required=True)

    a = cli.parse_args(argv)
    if a.command == "prepare":
        result = prepare(a.manifest, a.out, prior_mode=a.prior_mode, prior_index=a.prior_index,
                         shared=a.shared_prior, raw_manifest=a.raw_manifest, cycle_csv=a.cycle_csv,
                         seed=a.simulation_seed, engineering=a.engineering)
    elif a.command == "train":
        result = run_train(read_config(a.config), a.manifest, a.split, a.out, resume=a.resume, stop_after=a.stop_after)
    elif a.command == "predict":
        result = run_predict(
            a.checkpoint, a.manifest, a.split, a.split_name, a.out,
            rstar_path=a.rstar, alpha=a.alpha, cache=a.cache, device=a.device, amp=a.amp,
            mode=a.mode, tile=a.tile, overlap=a.overlap, d_tile=a.decnn_tile, d_overlap=a.decnn_overlap,
            baseline_design=read_json(a.baseline_design) if a.baseline_design else None,
        )
    elif a.command == "calibrate":
        value = calibrate(
            a.manifest, a.split, a.cache, a.annotations, a.extractor, a.out, device=a.device,
            metric=settings(a.window, a.motion_tolerance_ml, a.minimum_motion_animals),
        )
        result = {"path": str(a.out), "alpha": {k: v["alpha"] for k, v in value["choices"].items()}}
    else:
        value = evaluate(a.manifest, a.split, a.cache, a.annotations, a.extractor, a.lock, a.out, device=a.device)
        result = {"path": str(a.out / "summary.json"), "decision": value["decision"]}
    print(json.dumps(result, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
