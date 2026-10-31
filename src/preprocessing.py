"""Data sanitization, Zeek deduplication, IANA port categorization, and preprocessing."""

import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DROP_IDENTIFIER_COLS,
    LABEL_MAPPING,
    MIN_CLASS_SAMPLES,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    SAMPLE_SIZE,
    SEED,
    SMOTE_MIN_TRAIN_SAMPLES,
    TARGET_COL,
    TEST_SIZE,
)


def consolidate_iot23_label(raw_value: object) -> str:
    """Maps granular IoT-23 flow labels into the 6 canonical traffic classes."""
    tokens = str(raw_value).split()
    base_token = tokens[-1] if len(tokens) > 1 else str(raw_value)
    return LABEL_MAPPING.get(base_token, base_token)


def categorize_iana_port(port_series: pd.Series) -> np.ndarray:
    """Groups numerical TCP/UDP port numbers into standardized IANA port regimes."""
    p = pd.to_numeric(port_series, errors="coerce")
    return np.where(
        p.isna(),
        "missing",
        np.where(
            p < 1024,
            "well_known",
            np.where(
                p < 10000,
                "registered_low",
                np.where(p < 49152, "registered_high", "dynamic_ephemeral"),
            ),
        ),
    )


def build_input_preprocessor() -> ColumnTransformer:
    """Creates a fresh ColumnTransformer for numeric and categorical flow features."""
    ohe_params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_params["sparse_output"] = False
    else:
        ohe_params["sparse"] = False

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(**ohe_params)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def load_and_sanitize_dataset(
    dataset_path: str,
    sample_size: int = SAMPLE_SIZE,
    min_class_samples: int = MIN_CLASS_SAMPLES,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw IoT-23 CSV, consolidates labels, removes identifiers/timestamps,
    deduplicates identical flows, bins ports into IANA categories, and performs
    proportional stratified subsampling.
    """
    df_raw = pd.read_csv(dataset_path, low_memory=False)
    unnamed_cols = [c for c in df_raw.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df_raw = df_raw.drop(columns=unnamed_cols)

    df_raw[TARGET_COL] = df_raw[TARGET_COL].map(consolidate_iot23_label)

    raw_counts = df_raw[TARGET_COL].value_counts()
    df_raw_dist = pd.DataFrame({
        "Raw_Flow_Count": raw_counts,
        "Percentage (%)": (raw_counts / len(df_raw) * 100).round(4),
    })

    # 1. Remove session/host identifiers and capture timestamp (ts)
    df_clean = df_raw.drop(
        columns=[c for c in DROP_IDENTIFIER_COLS if c in df_raw.columns]
    ).copy()

    # 2. Convert Zeek missing tokens to NaN
    df_clean = df_clean.replace(["-", "(empty)", "", "nan", "None"], np.nan)
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # 3. Exact deduplication prior to sampling and splitting
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    unique_counts = df_clean[TARGET_COL].value_counts()

    # 4. IANA port categorization
    df_clean["orig_port_cat"] = categorize_iana_port(df_clean["id.orig_p"])
    df_clean["resp_port_cat"] = categorize_iana_port(df_clean["id.resp_p"])
    df_clean = df_clean.drop(columns=["id.orig_p", "id.resp_p"])

    # 5. Proportional stratified subsampling
    sampled_subsets = []
    total_unique = len(df_clean)
    for cls_name, cls_count in unique_counts.items():
        proportion = cls_count / total_unique
        target_n = max(min_class_samples, int(round(sample_size * proportion)))
        cls_df = df_clean[df_clean[TARGET_COL] == cls_name]
        sampled_subsets.append(
            cls_df.sample(n=min(target_n, len(cls_df)), random_state=seed)
        )

    df_sample = (
        pd.concat(sampled_subsets, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    for col in NUMERIC_FEATURES:
        df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df_sample[col] = df_sample[col].astype(object)

    X_raw_sample = df_sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    df_missing_audit = pd.DataFrame(
        {
            "Feature_Type": [
                "Numeric" if c in NUMERIC_FEATURES else "Categorical"
                for c in X_raw_sample.columns
            ],
            "Valid_Count": X_raw_sample.notna().sum().values,
            "Missing_Count": X_raw_sample.isna().sum().values,
            "Missing_Percentage (%)": (
                X_raw_sample.isna().mean().values * 100
            ).round(2),
        },
        index=X_raw_sample.columns,
    )

    return df_raw, df_clean, df_sample, df_raw_dist, df_missing_audit


def split_and_preprocess_data(
    df_sample: pd.DataFrame,
    test_size: float = TEST_SIZE,
    smote_min_samples: int = SMOTE_MIN_TRAIN_SAMPLES,
    seed: int = SEED,
    export_holdout_csv: bool = True,
) -> Dict[str, object]:
    """Applies log1p transformation, performs stratified 80/20 train/hold-out split,
    exports a leakage-free hold-out test sample CSV for Gradio, fits ColumnTransformer
    strictly on X_train, and applies calibrated SMOTE strictly on X_train.
    """
    X_raw_sample = df_sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y_raw_sample = df_sample[TARGET_COL].astype(str).copy()

    X_transformed = X_raw_sample.copy()
    for col in NUMERIC_FEATURES:
        X_transformed[col] = np.log1p(np.maximum(X_transformed[col], 0))

    target_encoder = LabelEncoder()
    y_encoded = pd.Series(
        target_encoder.fit_transform(y_raw_sample),
        index=y_raw_sample.index,
        name=TARGET_COL,
    )
    class_names = list(target_encoder.classes_)
    num_classes = len(class_names)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_transformed,
        y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded,
    )

    if export_holdout_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df_holdout_export = df_sample.loc[X_test_raw.index].copy()
        df_holdout_demo = (
            df_holdout_export.sample(
                n=min(2500, len(df_holdout_export)), random_state=seed
            )
            .reset_index(drop=True)
        )
        df_holdout_demo.to_csv(
            REPORTS_DIR / "test_sample_holdout.csv", index=False
        )

    preprocessor = build_input_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    ohe_feature_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    all_feature_names = NUMERIC_FEATURES + ohe_feature_names

    train_counts = y_train.value_counts()
    smote_strategy = {
        cls_idx: max(int(cnt), int(smote_min_samples))
        for cls_idx, cnt in train_counts.items()
    }
    k_nn = max(1, min(5, int(train_counts.min()) - 1))
    smote_sampler = SMOTE(
        sampling_strategy=smote_strategy,
        random_state=seed,
        k_neighbors=k_nn,
    )
    X_train_bal, y_train_bal = smote_sampler.fit_resample(
        X_train_proc, y_train.reset_index(drop=True)
    )

    df_partition_summary = pd.DataFrame({
        "Class_Name": class_names,
        "Train_Before_SMOTE": [
            int((y_train == i).sum()) for i in range(num_classes)
        ],
        "Train_After_SMOTE": [
            int((y_train_bal == i).sum()) for i in range(num_classes)
        ],
        "Holdout_Test_Pure": [
            int((y_test == i).sum()) for i in range(num_classes)
        ],
    }).set_index("Class_Name")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "class_names": class_names,
        "num_classes": num_classes,
        "all_feature_names": all_feature_names,
        "df_partition_summary": df_partition_summary,
    }
