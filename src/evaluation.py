"""Evaluation routines for RQ1 (20 STC Benchmark), RQ2 (Ablation), and RQ3 (CV & Scenarios)."""

import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    SEED,
    SMOTE_MIN_CV_SAMPLES,
    STACKING_CV_FOLDS,
)
from src.meta_feature_engineering import build_meta_learner_pipeline
from src.models_stc20 import build_stc20_configurations, get_base_learner_catalog
from src.preprocessing import build_input_preprocessor


def run_rq1_benchmark(
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: pd.Series,
    class_names: List[str],
    rf_estimators: int = 120,
    xgb_estimators: int = 120,
    meta_xgb_estimators: int = 100,
    seed: int = SEED,
) -> Dict[str, object]:
    """Trains and evaluates all 5 Single Base Learners and all 20 STC configurations."""
    num_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
    cc_class_idx = class_names.index("C&C") if "C&C" in class_names else 0

    df_stc_spec, stc20_catalog = build_stc20_configurations(
        n_classes=num_classes,
        rf_estimators=rf_estimators,
        xgb_estimators=xgb_estimators,
        meta_xgb_estimators=meta_xgb_estimators,
        seed=seed,
    )
    base_catalog = get_base_learner_catalog(
        seed=seed, rf_estimators=rf_estimators, xgb_estimators=xgb_estimators
    )

    trained_models: Dict[str, BaseEstimator] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    test_probabilities: Dict[str, np.ndarray] = {}
    rq1_records = []

    for key, (letter, estimator) in base_catalog.items():
        model_id = f"Single-{letter} ({key.upper()})"
        clf = clone(estimator)
        t0 = time.time()
        clf.fit(X_train_bal, y_train_bal)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test_proc)
        y_prob = clf.predict_proba(X_test_proc)

        per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
        roc_auc = roc_auc_score(
            y_test_bin, y_prob, average="macro", multi_class="ovr"
        )
        pr_auc = average_precision_score(y_test_bin, y_prob, average="macro")

        trained_models[model_id] = clf
        test_predictions[model_id] = y_pred
        test_probabilities[model_id] = y_prob

        rq1_records.append({
            "Code": f"Single-{letter}",
            "Model_Name": model_id,
            "Architecture": "Single Base Learner",
            "Base_Combination": key.upper(),
            "Meta_Learner": "-",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision_Macro": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall_Macro": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1_Macro": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1_Weighted": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "ROC_AUC_Macro": roc_auc,
            "PR_AUC_Macro": pr_auc,
            "CC_Recall": float(per_class_rec[cc_class_idx]),
            "Train_Time_s": round(train_time, 3),
        })

    for _, row_meta in df_stc_spec.iterrows():
        stc_code = row_meta["STC_Code"]
        full_label = row_meta["Full_Label"]
        clf = stc20_catalog[stc_code]

        t0 = time.time()
        clf.fit(X_train_bal, y_train_bal)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test_proc)
        y_prob = clf.predict_proba(X_test_proc)

        per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
        roc_auc = roc_auc_score(
            y_test_bin, y_prob, average="macro", multi_class="ovr"
        )
        pr_auc = average_precision_score(y_test_bin, y_prob, average="macro")

        trained_models[stc_code] = clf
        test_predictions[stc_code] = y_pred
        test_probabilities[stc_code] = y_prob

        rq1_records.append({
            "Code": stc_code,
            "Model_Name": full_label,
            "Architecture": f"Stacking ({row_meta['Meta_Learner']} Meta)",
            "Base_Combination": row_meta["Base_Learners"],
            "Meta_Learner": row_meta["Meta_Learner"],
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision_Macro": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall_Macro": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1_Macro": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1_Weighted": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "ROC_AUC_Macro": roc_auc,
            "PR_AUC_Macro": pr_auc,
            "CC_Recall": float(per_class_rec[cc_class_idx]),
            "Train_Time_s": round(train_time, 3),
        })

    df_rq1_leaderboard = (
        pd.DataFrame(rq1_records)
        .sort_values(by=["F1_Macro", "PR_AUC_Macro", "Accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    best_stacking_code = (
        df_rq1_leaderboard[df_rq1_leaderboard["Code"].str.startswith("STC-")]
        .iloc[0]["Code"]
    )
    best_single_code = (
        df_rq1_leaderboard[df_rq1_leaderboard["Code"].str.startswith("Single-")]
        .iloc[0]["Model_Name"]
    )

    return {
        "df_stc_spec": df_stc_spec,
        "df_rq1_leaderboard": df_rq1_leaderboard,
        "trained_models": trained_models,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "best_stacking_code": best_stacking_code,
        "best_single_code": best_single_code,
        "y_test_bin": y_test_bin,
    }


def run_rq2_ablation_study(
    top_triplet_keys: List[str],
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: pd.Series,
    y_test_bin: np.ndarray,
    num_classes: int = 6,
    rf_estimators: int = 120,
    xgb_estimators: int = 120,
    meta_xgb_estimators: int = 100,
    seed: int = SEED,
) -> pd.DataFrame:
    """Executes the 4-stage controlled ablation study for MetaFeatureEngineer (RQ2)."""
    ablation_regimes = [
        ("Stage 0: Raw Probabilities (No Meta-FE)", False, False, False),
        ("Stage 1: + Z-Score Normalization (StandardScaler)", True, False, False),
        ("Stage 2: + Meta-Interactions (Entropy, Margin, Consensus)", True, True, False),
        ("Stage 3: Full Proposed Meta-FE (Norm + Interactions + PCA)", True, True, True),
    ]

    ablation_records = []
    base_cat = get_base_learner_catalog(
        seed=seed, rf_estimators=rf_estimators, xgb_estimators=xgb_estimators
    )

    for ml_code in ["LR", "XGB"]:
        baseline_f1 = None
        for stage_name, sc_flag, int_flag, pca_flag in ablation_regimes:
            estimators_triplet = [(k, clone(base_cat[k][1])) for k in top_triplet_keys]
            meta_pipe = build_meta_learner_pipeline(
                meta_code=ml_code,
                n_classes=num_classes,
                use_scaling=sc_flag,
                use_interactions=int_flag,
                use_pca=pca_flag,
                xgb_estimators=meta_xgb_estimators,
                seed=seed,
            )
            stack_ablation = StackingClassifier(
                estimators=estimators_triplet,
                final_estimator=meta_pipe,
                stack_method="predict_proba",
                cv=STACKING_CV_FOLDS,
                n_jobs=1,
            )
            t0 = time.time()
            stack_ablation.fit(X_train_bal, y_train_bal)
            elapsed = time.time() - t0

            y_pred_ab = stack_ablation.predict(X_test_proc)
            y_prob_ab = stack_ablation.predict_proba(X_test_proc)

            f1_mac = f1_score(y_test, y_pred_ab, average="macro", zero_division=0)
            if baseline_f1 is None:
                baseline_f1 = f1_mac

            fitted_fe = stack_ablation.final_estimator_.named_steps["meta_fe"]
            n_meta_in = getattr(
                fitted_fe, "n_raw_meta_features_", num_classes * len(top_triplet_keys)
            )
            n_pca_out = getattr(fitted_fe, "n_pca_components_", n_meta_in)
            final_dim = (num_classes + n_pca_out) if pca_flag else n_meta_in

            ablation_records.append({
                "Meta_Learner": ml_code,
                "Ablation_Stage": stage_name,
                "Meta_Feature_Dim": int(final_dim),
                "Accuracy": accuracy_score(y_test, y_pred_ab),
                "Precision_Macro": precision_score(
                    y_test, y_pred_ab, average="macro", zero_division=0
                ),
                "Recall_Macro": recall_score(
                    y_test, y_pred_ab, average="macro", zero_division=0
                ),
                "F1_Macro": f1_mac,
                "Delta_F1_vs_Stage0": f1_mac - baseline_f1,
                "ROC_AUC_Macro": roc_auc_score(
                    y_test_bin, y_prob_ab, average="macro", multi_class="ovr"
                ),
                "PR_AUC_Macro": average_precision_score(
                    y_test_bin, y_prob_ab, average="macro"
                ),
                "Fit_Time_s": round(elapsed, 3),
            })

    return pd.DataFrame(ablation_records)


def run_rq3_cross_validation(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    top_stc_codes: List[str],
    num_classes: int = 6,
    cv_folds: int = CV_FOLDS,
    smote_min_cv: int = SMOTE_MIN_CV_SAMPLES,
    rf_estimators: int = 120,
    xgb_estimators: int = 120,
    meta_xgb_estimators: int = 100,
    seed: int = SEED,
) -> pd.DataFrame:
    """Executes leakage-free 5-Fold Stratified Cross-Validation (RQ3 Part 1)."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_records = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
        skf.split(X_train_raw, y_train), start=1
    ):
        X_f_tr, X_f_val = X_train_raw.iloc[tr_idx], X_train_raw.iloc[val_idx]
        y_f_tr, y_f_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        fold_prep = build_input_preprocessor()
        X_f_tr_proc = fold_prep.fit_transform(X_f_tr)
        X_f_val_proc = fold_prep.transform(X_f_val)

        f_counts = y_f_tr.value_counts()
        f_strategy = {k: max(int(v), int(smote_min_cv)) for k, v in f_counts.items()}
        k_nn = max(1, min(5, int(f_counts.min()) - 1))
        f_smote = SMOTE(
            sampling_strategy=f_strategy, random_state=seed, k_neighbors=k_nn
        )
        X_f_tr_bal, y_f_tr_bal = f_smote.fit_resample(
            X_f_tr_proc, y_f_tr.reset_index(drop=True)
        )

        base_cat = get_base_learner_catalog(
            seed=seed, rf_estimators=rf_estimators, xgb_estimators=xgb_estimators
        )
        _, stc_dict = build_stc20_configurations(
            n_classes=num_classes,
            rf_estimators=rf_estimators,
            xgb_estimators=xgb_estimators,
            meta_xgb_estimators=meta_xgb_estimators,
            seed=seed,
        )

        fold_candidates: Dict[str, BaseEstimator] = {}
        for k, (letter, est) in base_cat.items():
            fold_candidates[f"Single-{letter} ({k.upper()})"] = clone(est)
        for code in top_stc_codes:
            fold_candidates[code] = clone(stc_dict[code])

        for model_name, est in fold_candidates.items():
            t0 = time.time()
            est.fit(X_f_tr_bal, y_f_tr_bal)
            preds = est.predict(X_f_val_proc)
            elapsed = time.time() - t0

            cv_records.append({
                "Model": model_name,
                "Fold": fold_idx,
                "Accuracy": accuracy_score(y_f_val, preds),
                "Precision_Macro": precision_score(
                    y_f_val, preds, average="macro", zero_division=0
                ),
                "Recall_Macro": recall_score(
                    y_f_val, preds, average="macro", zero_division=0
                ),
                "F1_Macro": f1_score(
                    y_f_val, preds, average="macro", zero_division=0
                ),
                "Time_s": elapsed,
            })

    df_cv_raw = pd.DataFrame(cv_records)
    return (
        df_cv_raw.groupby("Model")
        .agg(
            CV_Acc_Mean=("Accuracy", "mean"),
            CV_Acc_Std=("Accuracy", "std"),
            CV_Prec_Mean=("Precision_Macro", "mean"),
            CV_Rec_Mean=("Recall_Macro", "mean"),
            CV_F1_Mean=("F1_Macro", "mean"),
            CV_F1_Std=("F1_Macro", "std"),
            CV_Time_Mean_s=("Time_s", "mean"),
        )
        .sort_values(by="CV_F1_Mean", ascending=False)
        .reset_index()
    )


def run_rq3_traffic_scenarios(
    X_test_proc: np.ndarray,
    y_test: pd.Series,
    class_names: List[str],
    trained_models: Dict[str, BaseEstimator],
    models_to_evaluate: List[str],
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluates candidate models across 4 operational IoT traffic scenarios (RQ3 Part 2)."""
    y_test_arr = y_test.to_numpy()
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    mask_scen_a = np.isin(
        y_test_arr,
        [
            class_to_idx[c]
            for c in ["Benign", "DDoS", "Scanning"]
            if c in class_to_idx
        ],
    )
    mask_scen_b = np.isin(
        y_test_arr,
        [
            class_to_idx[c]
            for c in ["Benign", "C&C", "Botnet", "Attack"]
            if c in class_to_idx
        ],
    )

    rng_scen = np.random.default_rng(seed)
    benign_idx = np.where(y_test_arr == class_to_idx["Benign"])[0]
    malicious_idx = np.where(y_test_arr != class_to_idx["Benign"])[0]
    n_rare_mal = max(len(class_names) - 1, int(round(len(benign_idx) * 0.02)))
    rare_mal_idx = rng_scen.choice(
        malicious_idx, size=min(n_rare_mal, len(malicious_idx)), replace=False
    )
    idx_scen_c = np.concatenate([benign_idx, rare_mal_idx])

    scenarios_spec = [
        (
            "Scenario A: Volumetric & Scanning (Benign + DDoS + Scanning)",
            np.where(mask_scen_a)[0],
        ),
        (
            "Scenario B: Stealthy C&C & Exploitation (Benign + C&C + Botnet + Attack)",
            np.where(mask_scen_b)[0],
        ),
        (
            "Scenario C: Rare Intrusion Monitoring (High Benign-to-Malicious Ratio)",
            idx_scen_c,
        ),
        (
            "Scenario D: Full Heterogeneous Multi-Vector Traffic (All 6 Classes)",
            np.arange(len(y_test_arr)),
        ),
    ]

    scenario_records = []
    for scen_name, scen_indices in scenarios_spec:
        X_scen = X_test_proc[scen_indices]
        y_scen = y_test_arr[scen_indices]
        present_classes = np.unique(y_scen)

        for m_key in models_to_evaluate:
            clf = trained_models[m_key]
            y_p = clf.predict(X_scen)

            ben_mask = y_scen == class_to_idx["Benign"]
            ben_fp_rate = (
                float(np.mean(y_p[ben_mask] != class_to_idx["Benign"]))
                if np.any(ben_mask)
                else 0.0
            )

            scenario_records.append({
                "Traffic_Scenario": scen_name,
                "Sample_Count": len(scen_indices),
                "Model": m_key,
                "Accuracy": accuracy_score(y_scen, y_p),
                "Precision_Macro": precision_score(
                    y_scen,
                    y_p,
                    labels=present_classes,
                    average="macro",
                    zero_division=0,
                ),
                "Recall_Macro": recall_score(
                    y_scen,
                    y_p,
                    labels=present_classes,
                    average="macro",
                    zero_division=0,
                ),
                "F1_Macro": f1_score(
                    y_scen,
                    y_p,
                    labels=present_classes,
                    average="macro",
                    zero_division=0,
                ),
                "Benign_FPR": ben_fp_rate,
            })

    return pd.DataFrame(scenario_records)


def compute_feature_importance_table(
    trained_models: Dict[str, BaseEstimator],
    all_feature_names: List[str],
) -> pd.DataFrame:
    """Computes combined normalized feature importance across RF, XGB, and LR."""
    rf_fitted = trained_models["Single-A (RF)"]
    lr_fitted = trained_models["Single-B (LR)"]
    xgb_fitted = trained_models["Single-E (XGB)"]

    imp_rf = rf_fitted.feature_importances_
    imp_xgb = xgb_fitted.feature_importances_
    imp_lr_raw = np.mean(np.abs(lr_fitted.coef_), axis=0)
    imp_lr = imp_lr_raw / (np.sum(imp_lr_raw) + 1e-12)

    imp_combined = (imp_rf + imp_xgb + imp_lr) / 3.0

    return (
        pd.DataFrame({
            "Feature": all_feature_names,
            "RF_Gini_Importance": imp_rf,
            "XGB_Gain_Importance": imp_xgb,
            "LR_Coef_Importance": imp_lr,
            "Combined_Importance": imp_combined,
        })
        .sort_values(by="Combined_Importance", ascending=False)
        .reset_index(drop=True)
    )
