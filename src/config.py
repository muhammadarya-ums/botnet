"""Global configuration, constants, schema definitions, and label mappings."""

from pathlib import Path

SEED: int = 42
DATASET_PATH: str = "dataset.csv"
TARGET_COL: str = "label"

SAMPLE_SIZE: int = 60000
MIN_CLASS_SAMPLES: int = 250
TEST_SIZE: float = 0.20

CV_FOLDS: int = 5
STACKING_CV_FOLDS: int = 3
SMOTE_MIN_TRAIN_SAMPLES: int = 2500
SMOTE_MIN_CV_SAMPLES: int = 1800

MODELS_DIR: Path = Path("models")
REPORTS_DIR: Path = Path("reports")
FIGURES_DIR: Path = REPORTS_DIR / "figures"

DROP_IDENTIFIER_COLS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "local_orig",
    "local_resp",
    "detailed-label",
]

NUMERIC_FEATURES = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "conn_state",
    "history",
    "orig_port_cat",
    "resp_port_cat",
]

LABEL_MAPPING = {
    "PartOfAHorizontalPortScan": "Scanning",
    "Okiru-Attack": "Botnet",
    "Okiru": "Botnet",
    "-": "Benign",
    "DDoS": "DDoS",
    "Attack": "Attack",
    "FileDownload": "Attack",
    "C&C-HeartBeat": "C&C",
    "C&C-Torii": "Botnet",
    "C&C-Mirai": "Botnet",
    "C&C-FileDownload": "Attack",
    "C&C-HeartBeat-FileDownload": "C&C",
}
