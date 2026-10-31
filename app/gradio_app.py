#!/usr/bin/env python3
"""Standalone Interactive Gradio Web Interface for IoT-23 Stacking Inference."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DROP_IDENTIFIER_COLS, MODELS_DIR
from src.preprocessing import categorize_iana_port, consolidate_iot23_label


def prepare_inference_dataframe(
    df_input: pd.DataFrame,
    bundle: dict,
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """Applies the exact leakage-free preprocessing pipeline to an uploaded CSV."""
    df_work = df_input.copy()
    unnamed = [c for c in df_work.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df_work = df_work.drop(columns=unnamed)

    y_true_idx = None
    class_names = bundle["class_names"]
    label_candidates = [
        c for c in df_work.columns if c.lower() in ("label", "labels", "target", "y")
    ]
    if label_candidates:
        lbl_col = label_candidates[0]
        raw_lbl = df_work[lbl_col].map(consolidate_iot23_label)
        mapping = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        mapped = raw_lbl.map(lambda v: mapping.get(str(v), -1)).to_numpy()
        if np.all(mapped >= 0):
            y_true_idx = mapped
        df_work = df_work.drop(columns=[lbl_col])

    drop_cols = [c for c in DROP_IDENTIFIER_COLS if c in df_work.columns]
    if drop_cols:
        df_work = df_work.drop(columns=drop_cols)

    df_work = df_work.replace(["-", "(empty)", "", "nan", "None"], np.nan)
    df_work = df_work.replace([np.inf, -np.inf], np.nan)

    if "id.orig_p" in df_work.columns and "orig_port_cat" not in df_work.columns:
        df_work["orig_port_cat"] = categorize_iana_port(df_work["id.orig_p"])
    if "id.resp_p" in df_work.columns and "resp_port_cat" not in df_work.columns:
        df_work["resp_port_cat"] = categorize_iana_port(df_work["id.resp_p"])

    num_cols = bundle["numeric_features"]
    cat_cols = bundle["categorical_features"]

    for col in num_cols:
        if col not in df_work.columns:
            df_work[col] = np.nan
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
        df_work[col] = np.log1p(np.maximum(df_work[col], 0))

    for col in cat_cols:
        if col not in df_work.columns:
            df_work[col] = "missing"
        df_work[col] = df_work[col].astype(object)

    X_ready = df_work[num_cols + cat_cols].copy()
    X_encoded = bundle["preprocessor"].transform(X_ready)
    return X_encoded, y_true_idx, df_work


def run_gradio_iot23_inference(csv_file, top_n: int = 20):
    if csv_file is None:
        return pd.DataFrame(), "No CSV file uploaded.", None, ""

    bundle_path = MODELS_DIR / "best_iot23_stacking_pipeline.joblib"
    if not bundle_path.exists():
        return (
            pd.DataFrame(),
            f"Model bundle not found at `{bundle_path}`. Please run `python run_pipeline.py` first.",
            None,
            "",
        )

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    class_names = bundle["class_names"]
    benign_idx = class_names.index("Benign") if "Benign" in class_names else 0

    file_path = csv_file.name if hasattr(csv_file, "name") else str(csv_file)
    df_uploaded = pd.read_csv(file_path, low_memory=False)
    X_enc, y_true_idx, _ = prepare_inference_dataframe(df_uploaded, bundle)

    y_pred_idx = model.predict(X_enc)
    y_proba = model.predict_proba(X_enc)

    malicious_prob = 1.0 - y_proba[:, benign_idx]
    pred_labels = [class_names[i] for i in y_pred_idx]

    df_out = pd.DataFrame({
        "Flow_Index": np.arange(len(df_uploaded)),
        "Predicted_Class": pred_labels,
        "Malicious_Confidence": np.round(malicious_prob, 4),
        "Top_Class_Probability": np.round(np.max(y_proba, axis=1), 4),
    })

    mal_count = int(np.sum(np.array(pred_labels) != "Benign"))
    summary_md = (
        f"**Active Model**: `{bundle['best_model_name']}` | "
        f"**Total Flows Analyzed**: `{len(df_uploaded):,}` | "
        f"**Malicious Flows Detected**: `{mal_count:,}` "
        f"(`{mal_count / max(1, len(df_uploaded)) * 100:.2f}%`)"
    )

    cm_fig = None
    report_str = "Ground-truth `label` column not present in uploaded file."
    if y_true_idx is not None:
        cm_norm = confusion_matrix(
            y_true_idx,
            y_pred_idx,
            labels=list(range(len(class_names))),
            normalize="true",
        )
        cm_fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm, display_labels=class_names
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format=".2f",
            xticks_rotation=30,
        )
        ax.set_title(
            f"Normalized Confusion Matrix ({bundle['best_model_code']})",
            fontweight="bold",
        )
        ax.grid(False)
        plt.tight_layout()

        report_str = classification_report(
            y_true_idx,
            y_pred_idx,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

    top_suspicious = (
        df_out.sort_values(by="Malicious_Confidence", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )
    return top_suspicious, summary_md, cm_fig, report_str


def launch_gradio_app(share: bool = False):
    import gradio as gr

    with gr.Blocks(
        title="IoT-23 Botnet Detection — Meta-Engineered Stacking"
    ) as demo:
        gr.Markdown(
            "## IoT-23 Multiclass Botnet Detection — Meta-Engineered Stacking Ensemble"
        )
        gr.Markdown(
            "Upload an IoT-23 network flow CSV (such as `reports/test_sample_holdout.csv` exported from the pure hold-out test split) "
            "to inspect flow predictions, malicious confidence scores, and evaluation metrics."
        )
        with gr.Row():
            csv_input = gr.File(
                label="Upload Network Flow CSV", file_types=[".csv"]
            )
            top_n_slider = gr.Slider(
                5, 100, value=20, step=5, label="Top-N Highest-Risk Flows"
            )
        run_button = gr.Button("Execute Stacking Inference")
        summary_box = gr.Markdown()
        table_output = gr.Dataframe(
            label="Top-N Suspicious Network Flows", interactive=False
        )
        with gr.Row():
            cm_output = gr.Plot(label="Normalized Confusion Matrix")
            report_output = gr.Textbox(
                label="Per-Class Classification Report", lines=12
            )
        run_button.click(
            fn=run_gradio_iot23_inference,
            inputs=[csv_input, top_n_slider],
            outputs=[table_output, summary_box, cm_output, report_output],
        )
    demo.launch(share=share)


if __name__ == "__main__":
    launch_gradio_app()
