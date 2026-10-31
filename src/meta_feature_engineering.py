"""RQ2 Core: Custom Meta-Level Feature Engineering Transformer for Stacking Ensembles."""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import SEED


class MetaFeatureEngineer(BaseEstimator, TransformerMixin):
    """Meta-level feature engineering transformer for stacking ensembles (RQ2).

    Enriches the concatenated out-of-fold base-learner probability matrix P of shape
    (n_samples, n_base_learners * n_classes) by synthesizing:
      1. Class-wise arithmetic consensus mean across base learners
      2. Class-wise multiplicative geometric consensus across base learners
      3. Class-wise inter-learner standard deviation (epistemic disagreement)
      4. Per-learner Shannon entropy
      5. Per-learner Top-2 confidence margin
      6. Pairwise base-learner probability dot-product agreement
    Followed by optional StandardScaler normalization and orthogonal PCA projection.
    """

    def __init__(
        self,
        n_classes: int = 6,
        use_scaling: bool = True,
        use_interactions: bool = True,
        use_pca: bool = True,
        pca_variance: float = 0.98,
        random_state: int = SEED,
    ):
        self.n_classes = n_classes
        self.use_scaling = use_scaling
        self.use_interactions = use_interactions
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.random_state = random_state

    def _synthesize_features(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.nan_to_num(
            np.asarray(X, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0
        )
        n_samples, total_cols = X_arr.shape
        n_learners = total_cols // self.n_classes

        if not self.use_interactions or n_learners < 2:
            return X_arr

        eps = 1e-12
        prob_cube = np.clip(
            X_arr[:, : n_learners * self.n_classes].reshape(
                n_samples, n_learners, self.n_classes
            ),
            eps,
            1.0,
        )

        # 1. Class-wise consensus mean: (n_samples, n_classes)
        consensus_mean = np.mean(prob_cube, axis=1)

        # 2. Class-wise multiplicative geometric consensus: (n_samples, n_classes)
        consensus_geom = np.exp(np.mean(np.log(prob_cube), axis=1))

        # 3. Class-wise inter-learner standard deviation: (n_samples, n_classes)
        consensus_std = np.std(prob_cube, axis=1)

        # 4. Per-learner Shannon entropy: (n_samples, n_learners)
        shannon_entropy = -np.sum(prob_cube * np.log(prob_cube), axis=2)

        # 5. Per-learner Top-2 probability margin: (n_samples, n_learners)
        sorted_probs = np.sort(prob_cube, axis=2)
        top2_margin = sorted_probs[:, :, -1] - sorted_probs[:, :, -2]

        # 6. Pairwise base-learner probability dot-product agreement: (n_samples, C(M,2))
        pairwise_products = []
        for i in range(n_learners):
            for j in range(i + 1, n_learners):
                sim_ij = np.sum(
                    prob_cube[:, i, :] * prob_cube[:, j, :],
                    axis=1,
                    keepdims=True,
                )
                pairwise_products.append(sim_ij)
        pairwise_mat = (
            np.hstack(pairwise_products)
            if pairwise_products
            else np.empty((n_samples, 0))
        )

        F = np.hstack([
            X_arr,
            consensus_mean,
            consensus_geom,
            consensus_std,
            shannon_entropy,
            top2_margin,
            pairwise_mat,
        ])
        return np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=0.0)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        F = self._synthesize_features(X)
        self.n_raw_meta_features_ = int(F.shape[1])

        if self.use_scaling:
            self.scaler_ = StandardScaler()
            F_scaled = np.nan_to_num(
                self.scaler_.fit_transform(F), nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            self.scaler_ = None
            F_scaled = F

        if self.use_pca and F_scaled.shape[1] > 2:
            self.pca_ = PCA(
                n_components=self.pca_variance,
                svd_solver="full",
                random_state=self.random_state,
            )
            self.pca_.fit(F_scaled)
            self.n_pca_components_ = int(self.pca_.n_components_)
        else:
            self.pca_ = None
            self.n_pca_components_ = int(F_scaled.shape[1])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        F = self._synthesize_features(X)
        if self.scaler_ is not None:
            F_scaled = np.nan_to_num(
                self.scaler_.transform(F), nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            F_scaled = F

        if self.pca_ is not None:
            F_pca = self.pca_.transform(F_scaled)
            X_arr = np.nan_to_num(
                np.asarray(X, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0
            )
            n_samples, total_cols = X_arr.shape
            n_learners = total_cols // self.n_classes
            prob_cube = X_arr[:, : n_learners * self.n_classes].reshape(
                n_samples, n_learners, self.n_classes
            )
            consensus_mean = np.mean(prob_cube, axis=1)
            return np.hstack([consensus_mean, F_pca])

        return F_scaled


def build_meta_learner_pipeline(
    meta_code: str,
    n_classes: int = 6,
    use_scaling: bool = True,
    use_interactions: bool = True,
    use_pca: bool = True,
    pca_variance: float = 0.98,
    xgb_estimators: int = 100,
    seed: int = SEED,
) -> Pipeline:
    """Constructs the meta-learner Pipeline combining MetaFeatureEngineer and classifier."""
    if meta_code.upper() == "LR":
        clf = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=seed
        )
    elif meta_code.upper() == "XGB":
        clf = XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported meta_code: {meta_code}")

    return Pipeline([
        (
            "meta_fe",
            MetaFeatureEngineer(
                n_classes=n_classes,
                use_scaling=use_scaling,
                use_interactions=use_interactions,
                use_pca=use_pca,
                pca_variance=pca_variance,
                random_state=seed,
            ),
        ),
        ("meta_clf", clf),
    ])
