"""RQ1 Core: Catalog of 5 Base Learners, 2 Meta-Learners, and 20 STC Configurations."""

import itertools
from typing import Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.config import SEED, STACKING_CV_FOLDS
from src.meta_feature_engineering import build_meta_learner_pipeline


def get_base_learner_catalog(
    seed: int = SEED,
    rf_estimators: int = 120,
    xgb_estimators: int = 120,
) -> Dict[str, Tuple[str, BaseEstimator]]:
    """Returns the 5 canonical base learners (Codes A-E) per Thesis Proposal Table 3.2."""
    return {
        "rf": (
            "A",
            RandomForestClassifier(
                n_estimators=rf_estimators,
                max_depth=12,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        "lr": (
            "B",
            LogisticRegression(
                C=0.5,
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        "nb": (
            "C",
            GaussianNB(var_smoothing=1e-8),
        ),
        "knn": (
            "D",
            KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
                n_jobs=-1,
            ),
        ),
        "xgb": (
            "E",
            XGBClassifier(
                n_estimators=xgb_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    }


def build_stc20_configurations(
    n_classes: int = 6,
    use_scaling: bool = True,
    use_interactions: bool = True,
    use_pca: bool = True,
    cv_folds: int = STACKING_CV_FOLDS,
    rf_estimators: int = 120,
    xgb_estimators: int = 120,
    meta_xgb_estimators: int = 100,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, Dict[str, StackingClassifier]]:
    """Constructs all 20 Stacking Configurations (STC-01 to STC-20) per Thesis Table 3.2.

    Note: Uses n_jobs=1 on StackingClassifier while keeping n_jobs=-1 on base learners
    to avoid nested OpenMP/loky thread oversubscription.
    """
    catalog = get_base_learner_catalog(
        seed=seed, rf_estimators=rf_estimators, xgb_estimators=xgb_estimators
    )
    base_keys = ["rf", "lr", "nb", "knn", "xgb"]
    triplets = list(itertools.combinations(base_keys, 3))

    stc_models: Dict[str, StackingClassifier] = {}
    stc_metadata = []

    stc_counter = 1
    for ml_idx, ml_code in [(1, "LR"), (2, "XGB")]:
        for triplet in triplets:
            stc_id = f"STC-{stc_counter:02d}"
            letter_codes = [catalog[k][0] for k in triplet]
            estimator_list = [(k, clone(catalog[k][1])) for k in triplet]

            meta_pipe = build_meta_learner_pipeline(
                meta_code=ml_code,
                n_classes=n_classes,
                use_scaling=use_scaling,
                use_interactions=use_interactions,
                use_pca=use_pca,
                xgb_estimators=meta_xgb_estimators,
                seed=seed,
            )

            stack_clf = StackingClassifier(
                estimators=estimator_list,
                final_estimator=meta_pipe,
                stack_method="predict_proba",
                cv=cv_folds,
                n_jobs=1,
            )

            label_str = f"{stc_id} ({'+'.join(k.upper() for k in triplet)} -> {ml_code})"
            stc_models[stc_id] = stack_clf
            stc_metadata.append({
                "STC_Code": stc_id,
                "Proposal_Notation": f"BL: {','.join(letter_codes)} | ML: {ml_idx}",
                "Base_Learners": ", ".join(k.upper() for k in triplet),
                "Meta_Learner": ml_code,
                "Full_Label": label_str,
            })
            stc_counter += 1

    return pd.DataFrame(stc_metadata), stc_models
