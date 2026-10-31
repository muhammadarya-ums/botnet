#!/usr/bin/env python3
"""Command-line entry point to execute the complete IoT-23 Thesis Pipeline (RQ1, RQ2, RQ3)."""

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    SEED,
)
from src.evaluation import (
    compute_feature_importance_table,
    run_rq1_benchmark,
    run_rq2_ablation_study,
    run_rq3_cross_validation,
    run_rq3_traffic_scenarios,
)
from src.preprocessing import (
    load_and_sanitize_dataset,
    split_and_preprocess_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the complete IoT-23 Stacking Ensemble & Meta-FE Thesis Pipeline."
    )
    parser.add_argument(
        "--data-path",
        default=DATASET_PATH,
        help="Path to raw IoT-23 dataset.csv (default: dataset.csv)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run with reduced estimator counts (30 trees) for rapid verification",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rf_est = 30 if args.fast else 120
    xgb_est = 30 if args.fast else 120
    meta_xgb_est = 25 if args.fast else 100

    print("=" * 78)
    print("STEP 1: LOADING, SANITIZING, DEDUPLICATING & SUBSAMPLING DATASET")
    print("=" * 78)
    df_raw, df_clean, df_sample, df_raw_dist, df_missing_audit = (
        load_and_sanitize_dataset(dataset_path=args.data_path, seed=SEED)
    )
    print(f"Raw Rows: {len(df_raw):,} | Unique Deduplicated Rows: {len(df_clean):,} | Sampled Rows: {len(df_sample):,}")
    df_raw_dist.to_csv(REPORTS_DIR / "01_raw_class_distribution.csv")
    df_missing_audit.to_csv(REPORTS_DIR / "02_missing_value_audit.csv")

    print("\n" + "=" * 78)
    print("STEP 2: LOG1P TRANSFORM, 80/20 SPLIT, COLUMNTRANSFORMER & CALIBRATED SMOTE")
    print("=" * 78)
    prep_bundle = split_and_preprocess_data(
        df_sample=df_sample, seed=SEED, export_holdout_csv=True
    )
    X_train_raw = prep_bundle["X_train_raw"]
    y_train = prep_bundle["y_train"]
    y_test = prep_bundle["y_test"]
    X_test_proc = prep_bundle["X_test_proc"]
    X_train_bal = prep_bundle["X_train_bal"]
    y_train_bal = prep_bundle["y_train_bal"]
    class_names = prep_bundle["class_names"]
    num_classes = prep_bundle["num_classes"]
    all_feature_names = prep_bundle["all_feature_names"]
    prep_bundle["df_partition_summary"].to_csv(
        REPORTS_DIR / "03_partition_and_smote_summary.csv"
    )
    print(f"Encoded Features: {len(all_feature_names)} | X_train_bal: {X_train_bal.shape} | X_test_proc: {X_test_proc.shape}")

    print("\n" + "=" * 78)
    print("STEP 3 (RQ1): BENCHMARKING 5 BASE LEARNERS & 20 STACKING CONFIGURATIONS (STC-01..20)")
    print("=" * 78)
    rq1_out = run_rq1_benchmark(
        X_train_bal=X_train_bal,
        y_train_bal=y_train_bal,
        X_test_proc=X_test_proc,
        y_test=y_test,
        class_names=class_names,
        rf_estimators=rf_est,
        xgb_estimators=xgb_est,
        meta_xgb_estimators=meta_xgb_est,
        seed=SEED,
    )
    df_rq1_leaderboard = rq1_out["df_rq1_leaderboard"]
    trained_models = rq1_out["trained_models"]
    test_predictions = rq1_out["test_predictions"]
    test_probabilities = rq1_out["test_probabilities"]
    best_stacking_code = rq1_out["best_stacking_code"]
    best_single_code = rq1_out["best_single_code"]
    y_test_bin = rq1_out["y_test_bin"]

    df_rq1_leaderboard.to_csv(REPORTS_DIR / "rq1_stc20_leaderboard.csv", index=False)
    print(f"Best Stacking Configuration : {best_stacking_code}")
    print(f"Best Single Base Learner    : {best_single_code}")
    print(df_rq1_leaderboard.head(10).to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 4 (RQ2): CONTROLLED 4-STAGE META-LEVEL FEATURE ENGINEERING ABLATION STUDY")
    print("=" * 78)
    top_stc_row = df_rq1_leaderboard[
        df_rq1_leaderboard["Code"].str.startswith("STC-")
    ].iloc[0]
    top_triplet_keys = [
        k.strip().lower() for k in top_stc_row["Base_Combination"].split(",")
    ]
    df_rq2_ablation = run_rq2_ablation_study(
        top_triplet_keys=top_triplet_keys,
        X_train_bal=X_train_bal,
        y_train_bal=y_train_bal,
        X_test_proc=X_test_proc,
        y_test=y_test,
        y_test_bin=y_test_bin,
        num_classes=num_classes,
        rf_estimators=rf_est,
        xgb_estimators=xgb_est,
        meta_xgb_estimators=meta_xgb_est,
        seed=SEED,
    )
    df_rq2_ablation.to_csv(REPORTS_DIR / "rq2_meta_fe_ablation.csv", index=False)
    print(df_rq2_ablation.to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 5 (RQ3 PART 1): LEAKAGE-FREE 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 78)
    top4_stc_codes = (
        df_rq1_leaderboard[df_rq1_leaderboard["Code"].str.startswith("STC-")]
        .head(4)["Code"]
        .tolist()
    )
    df_cv_summary = run_rq3_cross_validation(
        X_train_raw=X_train_raw,
        y_train=y_train,
        top_stc_codes=top4_stc_codes,
        num_classes=num_classes,
        rf_estimators=rf_est,
        xgb_estimators=xgb_est,
        meta_xgb_estimators=meta_xgb_est,
        seed=SEED,
    )
    df_cv_summary.to_csv(REPORTS_DIR / "rq3_cv5fold_summary.csv", index=False)
    print(df_cv_summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 6 (RQ3 PART 2): MULTI-SCENARIO IOT NETWORK TRAFFIC ROBUSTNESS EVALUATION")
    print("=" * 78)
    models_for_rq3 = list(
        dict.fromkeys(
            [best_stacking_code]
            + [c for c in top4_stc_codes if c != best_stacking_code][:1]
            + [best_single_code, "Single-A (RF)", "Single-B (LR)"]
        )
    )
    df_rq3_scenarios = run_rq3_traffic_scenarios(
        X_test_proc=X_test_proc,
        y_test=y_test,
        class_names=class_names,
        trained_models=trained_models,
        models_to_evaluate=models_for_rq3,
        seed=SEED,
    )
    df_rq3_scenarios.to_csv(REPORTS_DIR / "rq3_traffic_scenarios.csv", index=False)
    print(df_rq3_scenarios.to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 7: FEATURE IMPORTANCE & SERIALIZING PRODUCTION BUNDLE")
    print("=" * 78)
    df_feature_importance = compute_feature_importance_table(
        trained_models=trained_models, all_feature_names=all_feature_names
    )
    df_feature_importance.to_csv(
        REPORTS_DIR / "feature_importance_78features.csv", index=False
    )

    artifact_bundle = {
        "best_model_code": best_stacking_code,
        "best_model_name": df_rq1_leaderboard[
            df_rq1_leaderboard["Code"] == best_stacking_code
        ].iloc[0]["Model_Name"],
        "model": trained_models[best_stacking_code],
        "preprocessor": prep_bundle["preprocessor"],
        "target_encoder": prep_bundle["target_encoder"],
        "class_names": class_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_encoded_features": all_feature_names,
        "rq1_leaderboard": df_rq1_leaderboard,
        "rq2_ablation": df_rq2_ablation,
        "rq3_cv_summary": df_cv_summary,
        "rq3_scenarios": df_rq3_scenarios,
    }
    bundle_path = MODELS_DIR / "best_iot23_stacking_pipeline.joblib"
    joblib.dump(artifact_bundle, bundle_path)
    print(f"Saved model bundle to : {bundle_path.resolve()}")
    print(f"Saved CSV reports to  : {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
