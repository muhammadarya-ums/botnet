#!/usr/bin/env python3
"""Generates a realistic 21-column synthetic IoT-23 dataset.csv for rapid local smoke testing."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_iot23_csv(output_path: str = "dataset.csv", n_rows: int = 12000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    raw_labels = [
        "PartOfAHorizontalPortScan",
        "Okiru",
        "-",
        "DDoS",
        "C&C-HeartBeat",
        "Attack",
    ]
    probs = [0.42, 0.06, 0.30, 0.12, 0.06, 0.04]
    y = rng.choice(raw_labels, size=n_rows, p=probs)

    # Create class-conditional network flow patterns so models learn meaningful decision boundaries
    orig_pkts = np.where(
        y == "DDoS",
        rng.integers(200, 5000, size=n_rows),
        np.where(
            y == "PartOfAHorizontalPortScan",
            rng.integers(1, 4, size=n_rows),
            rng.integers(2, 80, size=n_rows),
        ),
    )
    orig_ip_bytes = orig_pkts * np.where(
        y == "DDoS", rng.integers(512, 1500, size=n_rows), rng.integers(40, 400, size=n_rows)
    )
    resp_port = np.where(
        y == "PartOfAHorizontalPortScan",
        rng.choice([22, 23, 80, 8080], size=n_rows),
        np.where(
            y == "C&C-HeartBeat",
            rng.choice([15000, 25000, 55000], size=n_rows),
            rng.choice([53, 80, 443, 8081, 49155], size=n_rows),
        ),
    )
    conn_state = np.where(
        y == "PartOfAHorizontalPortScan",
        "S0",
        np.where(y == "DDoS", "OTH", np.where(y == "-", "SF", "REJ")),
    )
    history = np.where(
        y == "PartOfAHorizontalPortScan",
        "S",
        np.where(y == "DDoS", "C", np.where(y == "-", "ShAdDaf", "Sr")),
    )

    df = pd.DataFrame({
        "ts": rng.uniform(1540000000.0, 1550000000.0, size=n_rows),
        "uid": [f"C{i:07d}" for i in range(n_rows)],
        "id.orig_h": [f"192.168.1.{rng.integers(2, 254)}" for _ in range(n_rows)],
        "id.orig_p": rng.integers(1024, 65535, size=n_rows),
        "id.resp_h": [f"10.0.0.{rng.integers(2, 254)}" for _ in range(n_rows)],
        "id.resp_p": resp_port,
        "proto": np.where(y == "DDoS", "udp", "tcp"),
        "service": np.where(y == "-", rng.choice(["-", "http", "dns"], size=n_rows), "-"),
        "duration": np.where(y == "PartOfAHorizontalPortScan", "-", rng.uniform(0.001, 30.0, size=n_rows).round(4).astype(str)),
        "orig_bytes": np.where(y == "PartOfAHorizontalPortScan", "-", (orig_ip_bytes * 0.8).astype(int).astype(str)),
        "resp_bytes": np.where(y == "-", rng.integers(64, 8192, size=n_rows).astype(str), "0"),
        "conn_state": conn_state,
        "local_orig": ["-"] * n_rows,
        "local_resp": ["-"] * n_rows,
        "missed_bytes": np.where(y == "DDoS", rng.integers(0, 500, size=n_rows), 0),
        "history": history,
        "orig_pkts": orig_pkts,
        "orig_ip_bytes": orig_ip_bytes,
        "resp_pkts": np.where(y == "-", rng.integers(1, 60, size=n_rows), rng.integers(0, 5, size=n_rows)),
        "resp_ip_bytes": np.where(y == "-", rng.integers(60, 12000, size=n_rows), rng.integers(0, 200, size=n_rows)),
        "detailed-label": y,
        "label": y,
    })

    out = Path(output_path)
    df.to_csv(out, index=False)
    print(f"Generated synthetic IoT-23 dataset ({len(df):,} rows x {df.shape[1]} cols) at: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic IoT-23 dataset.csv for smoke testing.")
    parser.add_argument("--output", default="dataset.csv", help="Output CSV path (default: dataset.csv)")
    parser.add_argument("--rows", type=int, default=12000, help="Number of rows to generate (default: 12000)")
    args = parser.parse_args()
    generate_synthetic_iot23_csv(output_path=args.output, n_rows=args.rows)
