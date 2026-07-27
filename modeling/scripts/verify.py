#!/usr/bin/env python3
"""Verify every recorded artifact and model-source SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    arguments = parser.parse_args()
    run_dir = Path(arguments.artifacts).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    checked = 0
    for relative, expected in manifest["artifact_hashes"].items():
        path = run_dir / relative
        checked += 1
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(f"artifact:{relative}")
    for relative, expected in manifest["sources"]["modeling_code_hashes"].items():
        path = PROJECT_ROOT / relative
        checked += 1
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(f"source:{relative}")
    if mismatches:
        raise RuntimeError("checksum mismatch: " + ", ".join(mismatches))
    print(f"verified {checked} files; 0 checksum mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

