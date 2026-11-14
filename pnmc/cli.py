"""Command-line entry points for PN-Net inference and PNMC reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from pnmc.config import NetworkConfig, ReconstructionConfig, RegistrationConfig
from pnmc.data.manifest import load_case_manifest, load_network_manifest
from pnmc.inference import infer_case
from pnmc.motion.stage import register_case
from pnmc.pipeline import run_case


def _write_audit(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def run_candidates(cases, output: Path, *, runner, loader=None) -> dict:
    """Try audited cases by case ID and preserve each exclusion reason."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = []
    audit = {"status": "running", "candidates": [], "selected_case_id": None}
    for item in cases:
        try:
            case = loader(item) if loader is not None else item
            ready.append(case)
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            audit["candidates"].append({"path": str(item), "status": "excluded",
                                        "reason": str(exc)})
    for index, case in enumerate(sorted(ready, key=lambda item: item.case_id)):
        try:
            result = runner(case, output / ("case_{:03d}".format(index)))
            if result.get("status") != "passed":
                raise ValueError("candidate did not reach passed status")
            audit["candidates"].append({"case_id": case.case_id,
                                        "status": "passed"})
            audit["selected_case_id"] = case.case_id
            audit["status"] = "passed"
            _write_audit(output / "selection_audit.json", audit)
            return result
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            audit["candidates"].append({"case_id": case.case_id,
                                        "status": "excluded", "reason": str(exc)})
            _write_audit(output / "selection_audit.json", audit)
    audit["status"] = "failed"
    _write_audit(output / "selection_audit.json", audit)
    raise ValueError("all physical candidates failed; see selection_audit.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer", help="Run PN-Net on a manifest split")
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--split", choices=("train", "valid", "test"), required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")

    register = commands.add_parser("register", help="Register enhanced phase volumes")
    register.add_argument("--case", type=Path, required=True)
    register.add_argument("--enhanced", type=Path, required=True)
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--device", default="cuda:0")

    reconstruct = commands.add_parser("reconstruct", help="Run full PNMC")
    reconstruct.add_argument("--case", type=Path, action="append", required=True,
                             help="Physical case manifest; repeat for audited fallback")
    reconstruct.add_argument("--checkpoint", type=Path, required=True)
    reconstruct.add_argument("--network-config", type=Path, required=True)
    reconstruct.add_argument("--config", type=Path, required=True)
    reconstruct.add_argument("--output", type=Path, required=True)
    reconstruct.add_argument("--device", default="cuda:0")
    reconstruct.add_argument("--resume", action="store_true",
                             help="Reuse verified stages and complete targets")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "infer":
            manifest = load_network_manifest(args.manifest)
            result = infer_case(manifest, args.checkpoint, args.output,
                                split=args.split, device=args.device)
        elif args.command == "register":
            case = load_case_manifest(args.case, require_physics=True)
            reconstruction = ReconstructionConfig.from_json(args.config)
            registration = RegistrationConfig.from_dict(
                reconstruction.registration or {})
            report = register_case(case, args.enhanced, registration,
                                   args.output, device=args.device)
            result = {"status": report["status"], "case_id": report["case_id"],
                      "report": str(args.output.resolve() / "registration.json")}
        else:
            network = NetworkConfig.from_json(args.network_config)
            network = NetworkConfig.from_dict({**asdict(network),
                                               "device": args.device})
            reconstruction = ReconstructionConfig.from_json(args.config)
            if len(args.case) == 1:
                case = load_case_manifest(args.case[0], require_physics=True)
                result = run_case(case, checkpoint=args.checkpoint,
                                  network_config=network,
                                  reconstruction_config=reconstruction,
                                  output=args.output, resume=args.resume)
            else:
                result = run_candidates(
                    args.case, args.output,
                    loader=lambda path: load_case_manifest(path, require_physics=True),
                    runner=lambda case, folder: run_case(
                        case, checkpoint=args.checkpoint, network_config=network,
                        reconstruction_config=reconstruction, output=folder,
                        resume=args.resume))
                result = {"status": result["status"],
                          "case_id": result["case_id"], "output": str(args.output)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, "{} failed: {}\n".format(args.command, exc))


if __name__ == "__main__":
    raise SystemExit(main())
