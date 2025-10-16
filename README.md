# Explaining Loan Approval Decisions with SHAP and LIME

This project builds an interpretable credit approval pipeline that couples traditional machine-learning models with post-hoc explainability using SHAP and LIME. The repository provides reproducible data-preparation, modelling, and interpretation scripts alongside ready-to-visualise HTML reports for understanding model behaviour and applicant-level decisions.

## Project Structure

```
├── data/                  # Raw input CSVs (application & credit records)
├── configs/               # Pipeline configuration (YAML, env examples)
├── src/                   # Reusable Python modules
│   ├── data/              # Loading and cleaning utilities
│   ├── features/          # (reserved for feature engineering helpers)
│   ├── models/            # Training, evaluation, and explanations
│   └── visualization/     # Plotting helpers (e.g., SHAP charts)
├── scripts/               # CLI entry points wrapping src modules
├── artifacts/             # Saved model pipelines and metrics
├── reports/               # Generated figures, explanations, summaries
├── tests/                 # Pytest suite
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Tooling configuration (black, isort, ruff)
```

## Quick Start
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Prepare the datasets: `python scripts/prepare_data.py`.
4. Train the models and store artifacts: `python scripts/train_model.py`.
5. Produce SHAP/LIME outputs: `python scripts/generate_explanations.py --models logistic_regression random_forest`.
6. Build interactive feature-importance charts (optional): `python scripts/visualize_explanations.py --model logistic_regression`.

Generated figures and reports are saved under `reports/explanations/`; trained pipelines reside in `artifacts/models/`.
