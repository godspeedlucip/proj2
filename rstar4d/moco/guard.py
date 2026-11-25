from __future__ import annotations

from pathlib import Path

from .data import artifact_rows
from .settings import dependencies
from ..data.contracts import read_manifest
from ..utils.io import canonical_hash, read_json, sha256


def enforce_split(manifest, split, lock_path, *, config=None, checkpoint=None,
                  baseline_checkpoint=None, baseline=None, command=None):
    rows = read_manifest(manifest)
    declared = {row.get("scan_split_name") for row in rows}
    if split == "all" and "test" in declared:
        raise ValueError("Tagged test cases require --split-name test and --test-lock")
    if split in ("val", "test") and declared - {None, split}:
        raise ValueError(f"Manifest contains cases outside the requested {split} submanifest")
    if split != "test":
        return None
    if not lock_path:
        raise ValueError("Test workflow requires --test-lock from validation")
    lock = read_json(lock_path)
    body = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("schema") != 1 or lock.get("lock_sha256") != canonical_hash(body):
        raise ValueError("Invalid MoCo validation lock")
    validation_cases = lock["validation_cases"]
    if any(row["case_id"] in validation_cases or row.get("sha256") in validation_cases.values() for row in rows):
        raise ValueError("Test inputs overlap validation case IDs or paired-file hashes")
    if command not in ("infer-moco", "prepare-moco"):
        return lock
    sources, fused, _ = dependencies(config)
    candidates = [item for item in lock["runs"] if item["config"] == config]
    if not candidates:
        raise ValueError("Test run differs from the effective configuration locked on validation")
    if command == "infer-moco":
        learned = fused and config["registration"]["mode"] == "learned"
        digest = sha256(checkpoint) if checkpoint and learned else None
        if digest not in {item["register_checkpoint_sha256"] for item in candidates}:
            raise ValueError("Test registration checkpoint differs from the validation lock")
        if "b" in sources:
            if baseline_checkpoint:
                baseline_digests = {sha256(baseline_checkpoint)}
            elif baseline:
                artifacts = artifact_rows(baseline)
                baseline_digests = {artifacts.get(row["case_id"], {}).get("checkpoint_sha256") for row in rows}
            else:
                raise ValueError("Test output requires frozen baseline")
            expected = {item["baseline_provider"]["checkpoint_sha256"]
                        for item in candidates if item.get("baseline_provider")}
            if not baseline_digests or baseline_digests - expected:
                raise ValueError("Test Stage II checkpoint differs from the validation lock")
    return lock


def context_for(manifest):
    path = Path(manifest).resolve()
    if path.is_dir():
        path /= "manifest.jsonl"
    context = path.parent.parent / "inference_context.json"
    return read_json(context) if context.exists() else None
