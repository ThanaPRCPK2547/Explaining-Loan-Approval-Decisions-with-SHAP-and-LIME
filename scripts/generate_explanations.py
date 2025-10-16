"""CLI entry point for generating model explanations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.explain import run_explanations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP and LIME explanations.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to YAML configuration file (relative to project root).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Specific model names to explain (default: all available).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_explanations(config_path=config_path, model_names=args.models)


if __name__ == "__main__":
    main()
