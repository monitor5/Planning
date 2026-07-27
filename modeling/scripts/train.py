#!/usr/bin/env python3
"""Train and validate the Elder Guardian exact/GNN model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "modeling" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elder_guardian.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "modeling" / "config" / "base.json"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--population-root")
    parser.add_argument("--facility-master")
    parser.add_argument("--historical-capacity")
    parser.add_argument("--output-root")
    arguments = parser.parse_args()
    output = run_pipeline(
        project_root=PROJECT_ROOT,
        config_path=arguments.config,
        run_id=arguments.run_id,
        population_root=arguments.population_root,
        facility_master_csv=arguments.facility_master,
        historical_capacity_csv=arguments.historical_capacity,
        output_root=arguments.output_root,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

