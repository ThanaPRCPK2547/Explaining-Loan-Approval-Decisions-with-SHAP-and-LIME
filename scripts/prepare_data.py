"""CLI entry point for running the data preparation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.prepare import run_data_preparation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare credit approval dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to YAML configuration file (relative to project root).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_data_preparation(config_path=config_path)


if __name__ == "__main__":
    main()
