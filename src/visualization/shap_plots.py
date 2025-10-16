"""Utilities for building interactive SHAP-based visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from src.models.explain import (
    compute_shap_values,
    get_feature_names,
    load_explain_config,
    load_validation_data,
    select_model_files,
)


def generate_shap_feature_importance(
    model_name: str,
    config_path: Path,
    output_path: Optional[Path] = None,
    top_n: int = 15,
) -> Path:
    """Create a horizontal bar chart of mean |SHAP| by feature."""
    config = load_explain_config(config_path)
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = (project_root / config.processed_dir).resolve()
    artifacts_dir = (project_root / config.artifacts_dir).resolve()
    output_dir = output_path.parent if output_path else (project_root / config.output_dir)

    validation = load_validation_data(processed_dir, config.target_column)
    x_val = validation.drop(columns=[config.target_column])

    model_files = select_model_files(artifacts_dir, [model_name])
    pipeline = joblib.load(model_files[0])
    preprocessor = pipeline.named_steps["preprocess"]

    # Sample rows for SHAP calculation
    sample_size = min(len(x_val), config.shap_sample_size)
    shap_sample = x_val.sample(n=sample_size, random_state=config.random_state)

    features = get_feature_names(preprocessor, shap_sample)
    shap_payload = compute_shap_values(pipeline, shap_sample, features)

    shap_values = np.asarray(shap_payload["values"])
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    summary_df = pd.DataFrame(
        {"feature": features, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)

    summary_df = summary_df.head(top_n).iloc[::-1]  # reverse for horizontal bar

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_path or (output_dir / f"{model_name}_shap_feature_importance.html")

    fig = px.bar(
        summary_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=f"Mean |SHAP| Feature Importance: {model_name}",
        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"},
    )

    figure_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    table_html = summary_df.iloc[::-1].to_html(
        index=False,
        float_format=lambda x: f"{x:0.4f}",
        columns=["feature", "mean_abs_shap"],
    )

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SHAP Feature Importance — {model_name}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0 auto;
      padding: 24px;
      max-width: 960px;
      line-height: 1.6;
      background-color: #f9fafb;
    }}
    h1 {{
      margin-bottom: 0.2em;
    }}
    h2 {{
      margin-top: 2em;
    }}
    .note {{
      padding: 12px 16px;
      background: #eef2ff;
      border-left: 4px solid #4f46e5;
      border-radius: 4px;
      margin-bottom: 24px;
    }}
    ol {{
      margin-left: 20px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 16px;
      background: white;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 8px;
      text-align: left;
    }}
    th {{
      background-color: #f3f4f6;
    }}
    footer {{
      margin-top: 32px;
      font-size: 0.85em;
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <h1>SHAP Feature Importance</h1>
  <p class="note">
    กราฟและตารางนี้สรุปอันดับความสำคัญของฟีเจอร์ต่อการทำนายของโมเดล <strong>{model_name}</strong>
    โดยใช้ค่าเฉลี่ยขนาดสัมบูรณ์ของ SHAP (mean |SHAP|) ยิ่งค่าเฉลี่ยสูงเท่าไร ฟีเจอร์นั้นยิ่งส่งผลต่อการพยากรณ์มากขึ้น.
  </p>
  <ol>
    <li>วางเมาส์บนแท่งแผนภูมิแต่ละรายการเพื่อดูค่าที่แน่นอน.</li>
    <li>mean |SHAP| ใช้เปรียบเทียบความสำคัญโดยรวม ช่วยให้มองเห็นว่าฟีเจอร์ใดมีอิทธิพลสูงสุด.</li>
    <li>ใช้ตารางด้านล่างเพื่อดูค่าเชิงตัวเลขหรือบันทึกข้อมูลเพิ่มเติม.</li>
  </ol>
  {figure_html}
  <h2>Top {len(summary_df)} Features</h2>
  {table_html}
  <footer>สร้างโดย pipeline อธิบายผล SHAP/LIME · ใช้ข้อมูลจำนวน {sample_size} รายการในการคำนวณค่าเฉลี่ย</footer>
</body>
</html>"""

    output_file.write_text(html_template, encoding="utf-8")
    return output_file
