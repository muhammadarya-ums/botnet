# Multiclass IoT Botnet Detection Using Stacking Ensemble and Meta-Level Feature Engineering (IoT-23)

This repository provides a complete, leakage-free experimental framework for multiclass IoT botnet detection on the Stratosphere IPS **IoT-23** (`conn.log`) dataset. It is provided in **two synchronized formats** so you can run it either interactively in Jupyter/Colab or as a modular Python software package from the terminal:

1. **Interactive Jupyter Notebook (`IoT23_Stacking_Ensemble_Thesis_Complete.ipynb`)**:
   - Divided into **31 structured cells** (16 Markdown documentation cells + 15 executable code cells).
   - Ready to open and run directly in JupyterLab, Jupyter Notebook, VS Code, or Google Colab.
2. **Modular Python Package (`src/`, `app/`, `scripts/`, `run_pipeline.py`)**:
   - Clean, production-grade software architecture separating configuration (`src/config.py`), data sanitization & preprocessing (`src/preprocessing.py`), RQ2 meta-feature engineering (`src/meta_feature_engineering.py`), RQ1 20-configuration stacking catalog (`src/models_stc20.py`), RQ3 evaluation & plotting (`src/evaluation.py`), and Gradio deployment (`app/gradio_app.py`).

## Project Directory Structure

```text
IoT23_Stacking_Ensemble_Thesis/
├── IoT23_Stacking_Ensemble_Thesis_Complete.ipynb  # Complete 31-cell Jupyter Notebook (RQ1, RQ2, RQ3)
├── run_pipeline.py                                # One-command CLI runner for the full experimental pipeline
├── requirements.txt                               # Python package dependencies
├── README.md                                      # Documentation and execution guide
├── src/
│   ├── __init__.py
│   ├── config.py                                  # Global constants, seed (42), feature schemas, label mappings
│   ├── preprocessing.py                           # Identifier/timestamp sanitization, Zeek NaN cleaning, deduplication, IANA port binning, log1p, ColumnTransformer, SMOTE
│   ├── meta_feature_engineering.py                # RQ2 Core: MetaFeatureEngineer (Consensus, Entropy, Margin, Dispersion, StandardScaler, PCA)
│   ├── models_stc20.py                            # RQ1 Core: 5 Base Learners (A-E), 2 Meta-Learners (1-2), and 20 Stacking Configurations (STC-01 to STC-20)
│   └── evaluation.py                              # RQ1 Benchmark, RQ2 4-Stage Ablation, RQ3 5-Fold CV & 4 Traffic Regimes, Per-Class metrics, 300-DPI figures
├── app/
│   └── gradio_app.py                              # Interactive Gradio web interface loading models/best_iot23_stacking_pipeline.joblib
├── scripts/
│   └── generate_synthetic_demo_data.py            # Generates a realistic 21-column synthetic dataset.csv for rapid smoke testing
├── models/                                        # Output directory for serialized .joblib model bundles
└── reports/                                       # Output directory for CSV leaderboards, test_sample_holdout.csv, and 300-DPI figures
    └── figures/
```

## Research Questions Addressed

- **RQ1 (Optimal Base & Meta-Learner Combinations)**:
  Evaluates all **5 Base Learners** (`A: Random Forest`, `B: Logistic Regression`, `C: Gaussian Naive Bayes`, `D: K-Nearest Neighbors`, `E: XGBoost`) and **2 Meta-Learners** (`ML 1: Logistic Regression`, `ML 2: XGBoost`) across all **20 Stacking Configurations (`STC-01` to `STC-20`)** defined in Table 3.2 of the thesis proposal using strict out-of-fold probability estimation (`cv=3`, `stack_method='predict_proba'`) and fresh estimator cloning (`sklearn.base.clone`).
- **RQ2 (Meta-Level Feature Engineering)**:
  Implements `MetaFeatureEngineer(BaseEstimator, TransformerMixin)` to enrich the out-of-fold probability matrix $P \in \mathbb{R}^{N \times 18}$ into 45 meta-features (class-wise arithmetic consensus, multiplicative geometric consensus, inter-learner standard deviation, per-learner Shannon entropy, top-2 probability margin, and pairwise cosine/dot-product agreement), followed by `StandardScaler` normalization and orthogonal `PCA` projection, validated via a controlled **4-Stage Ablation Study**.
- **RQ3 (Robustness Across Diverse IoT Network Traffic Scenarios)**:
  Executes a leakage-free **5-Fold Stratified Cross-Validation** protocol (fitting `ColumnTransformer` and `SMOTE` strictly inside each fold) and evaluates model resilience across **4 operational IoT traffic scenarios** on the unaugmented hold-out test set:
  1. *Scenario A*: Volumetric & Reconnaissance Traffic (`Benign`, `DDoS`, `Scanning`)
  2. *Scenario B*: Stealthy Command-and-Control & Exploitation Traffic (`Benign`, `C&C`, `Botnet`, `Attack`)
  3. *Scenario C*: Rare Intrusion Monitoring (`100:1` Benign-to-Malicious Ratio)
  4. *Scenario D*: Full Multi-Vector Heterogeneous IoT Traffic (All 6 Classes)

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place `dataset.csv` in the Root Folder (or Generate Demo Data for Verification)
Place your raw IoT-23 `dataset.csv` file in the project root directory (`IoT23_Stacking_Ensemble_Thesis/dataset.csv`).
If you want to run a 30-second verification test before running on the full 6.04M-row dataset:
```bash
python scripts/generate_synthetic_demo_data.py
```

### 3. Option A — Run via Jupyter Notebook
Open `IoT23_Stacking_Ensemble_Thesis_Complete.ipynb` in JupyterLab, Jupyter Notebook, VS Code, or Google Colab, and click **Run All**.

### 4. Option B — Run via Modular Python CLI
```bash
# Run full thesis pipeline on dataset.csv
python run_pipeline.py --data-path dataset.csv

# Or run a fast smoke test with reduced estimator counts
python run_pipeline.py --data-path dataset.csv --fast
```

### 5. Launch the Interactive Gradio Web Interface
After running the pipeline (which saves `models/best_iot23_stacking_pipeline.joblib` and exports `reports/test_sample_holdout.csv` from the unseen hold-out test split):
```bash
python app/gradio_app.py
```
