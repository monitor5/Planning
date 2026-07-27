#!/usr/bin/env python3
"""Read an immutable exact run and print its top audited candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--top", type=int, default=10)
    arguments = parser.parse_args()
    run_dir = Path(arguments.artifacts).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected_hash = manifest["artifact_hashes"]["exact_candidate_results.csv"]
    results_path = run_dir / "exact_candidate_results.csv"
    actual_hash = sha256_file(results_path)
    if actual_hash != expected_hash:
        raise RuntimeError("exact_candidate_results.csv checksum mismatch")
    scenario = arguments.scenario or manifest["config"]["base_scenario"]
    results = pd.read_csv(results_path, low_memory=False)
    subset = results.loc[(results["scenario"] == scenario) & results["valid"]]
    columns = [
        "gid",
        "borough",
        "latitude",
        "longitude",
        "baseline_gini",
        "candidate_gini",
        "delta_gini",
        "new_capacity",
    ]
    print(
        subset.nlargest(arguments.top, "delta_gini")[columns].to_string(index=False)
    )
    print(
        "\n주의: WALK_EUCLIDEAN_FEASIBILITY_UNKNOWN_SCREENING — "
        "필지·현행 정원·다중교통 검증 전 최종 부지가 아닙니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

