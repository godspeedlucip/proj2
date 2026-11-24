from __future__ import annotations

import argparse
from pathlib import Path

from .config import read_config


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def add_output(command):
    command.add_argument("--out", required=True, type=Path)


def main():
    parser = argparse.ArgumentParser(prog="python -m dpi_moco")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Bin views and reconstruct training input volumes")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--config", type=Path)
    add_output(prepare)
    train = commands.add_parser("train", help="Train a single stage")
    train.add_argument("stage", choices=("restore", "register", "denoise"))
    train.add_argument("--prepared", required=True, type=Path)
    add_output(train)
    train.add_argument("--restore", type=Path, help="Frozen G.pt used for register")
    train.add_argument("--domain", help="Scanner domain from the prepared manifest")
    train.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--epochs", type=positive_int, help="Total LR schedule length, defaults to 50")
    train.add_argument("--stop-after-epoch", type=positive_int, help="Stop early without changing the schedule")
    train.add_argument("--resume", nargs="?", const="latest", metavar="CHECKPOINT",
                       help="Resume latest completed epoch or an explicit checkpoint path")
    train.add_argument("--init-weights", type=Path, help="Initialize model weights for a new training run")
    train.add_argument("--batch-size", type=positive_int)
    train.add_argument("--patch-size", type=positive_int)
    train.add_argument("--slice-batch", type=positive_int)
    train.add_argument("--lr-start", type=float)
    train.add_argument("--lr-end", type=float)
    train.add_argument("--max-batches", type=positive_int, help="Limit batches per epoch for smoke runs")
    train.add_argument("--log-every", type=positive_int, help="Steps between TensorBoard batch loss points")
    train.add_argument("--loss", choices=("mse", "mse_gan", "full"))
    train.add_argument("--deterministic", action="store_true", default=None)
    infer = commands.add_parser("infer", help="Reconstruct a scan with three frozen stages")
    infer.add_argument("--manifest", required=True, type=Path)
    infer.add_argument("--config", type=Path)
    for stage in ("restore", "register", "denoise"):
        infer.add_argument(f"--{stage}", required=True, type=Path)
    add_output(infer)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--slice-batch", type=positive_int, default=4)
    infer.add_argument("--keep-intermediates", action="store_true")
    infer.add_argument("--keep-fields", action="store_true")
    infer.add_argument("--motion-qc", action="store_true", help="Export multiplanar views and displacement statistics")
    infer.add_argument("--motion-ablation", action="store_true", help="Re-use the same fields to fuse x and b, and save zero-field fusion of c")
    evaluate = commands.add_parser("evaluate", help="Compare predictions with same-grid phase references")
    evaluate.add_argument("--pred", required=True, type=Path)
    evaluate.add_argument("--reference", required=True, type=Path)
    evaluate.add_argument("--protocol", type=Path)
    evaluate.add_argument("--out", type=Path)
    evaluate.add_argument("--stages", nargs="+", default=["dpri"],
                          choices=("x", "b", "d", "d_noisy", "dpri", "d_from_x", "d_from_b", "d_zero_motion"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from .prepare import prepare as run
            run(args.manifest, read_config(args.config), args.out)
        elif args.command == "train":
            from .training import train as run
            run(args)
        elif args.command == "infer":
            from .inference import infer as run
            run(args, read_config(args.config))
        elif args.command == "evaluate":
            from .evaluate import evaluate as run
            run(args)
    except (ValueError, RuntimeError, FloatingPointError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
