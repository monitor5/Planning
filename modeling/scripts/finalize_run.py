#!/usr/bin/env python3
"""Refresh source/artifact hashes after a run without retraining it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "modeling" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elder_guardian.pipeline import (  # noqa: E402
    _artifact_hashes,
    _write_json,
    model_code_hashes,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    arguments = parser.parse_args()
    run_dir = Path(arguments.artifacts).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"]["modeling_code_hashes"] = model_code_hashes(PROJECT_ROOT)
    manifest["artifact_hashes"] = _artifact_hashes(run_dir)
    _write_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

