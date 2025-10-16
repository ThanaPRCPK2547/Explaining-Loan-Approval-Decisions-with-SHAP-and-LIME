"""CLI entry point to create interactive explainability visualizations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.visualization.shap_plots import generate_shap_feature_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP visualization HTML files.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to YAML configuration file (relative to project root).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to visualize (e.g., logistic_regression).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top features to include in the bar chart.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    output_path = Path(args.output) if args.output else None
    if output_path and not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    html_path = generate_shap_feature_importance(
        model_name=args.model,
        config_path=config_path,
        output_path=output_path,
        top_n=args.top_n,
    )
    print(f"Visualization saved to {html_path}")


if __name__ == "__main__":
    main()
