"""Summarize Tables S2–S4 from fold-level CSVs; no model inference.

Usage:
  python summarize_tables_s2_s4.py --input-dir fold_metrics --output-dir summary

Input layout (15 CSV files):
  fold_metrics/resnet50/{original,he,clahe,gamma,gaussian}.csv
  fold_metrics/vit/{original,he,clahe,gamma,gaussian}.csv
  fold_metrics/multimodal_vit/{original,he,clahe,gamma,gaussian}.csv

Each CSV must contain fold and the six metric columns listed below.
Metrics must be proportions (0–1), not percentages. Youden index can be negative.
SD is sample standard deviation across ten folds (ddof=1).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"resnet50": "S2", "vit": "S3", "multimodal_vit": "S4"}
SETTINGS = {"original": "Original", "he": "HE", "clahe": "CLAHE",
            "gamma": "Gamma correction", "gaussian": "Gaussian noise"}
METRICS = ["accuracy", "sensitivity", "specificity", "youden_index", "f1_score", "auc"]


def read_metrics(path):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode: {path}")
    df.columns = df.columns.str.strip()
    required = ["fold"] + METRICS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df[required].apply(pd.to_numeric, errors="raise")
    if len(df) != 10 or sorted(df["fold"].tolist()) != list(range(1, 11)):
        raise ValueError(f"{path}: expected exactly one row per fold, numbered 1–10")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"{path}: missing or nonfinite values")
    for metric in METRICS:
        lower = -1 if metric == "youden_index" else 0
        if not df[metric].between(lower, 1).all():
            raise ValueError(f"{path}: {metric} values outside [{lower}, 1]")
    return df.sort_values("fold")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Validate all inputs before writing results. No imputation or skipped folds.
    frames, provenance = {}, []
    for model in MODELS:
        for setting in SETTINGS:
            path = args.input_dir / model / f"{setting}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Required file missing: {path}")
            frames[model, setting] = read_metrics(path)
            provenance.append({"model": model, "preprocessing": setting,
                "file": f"{model}/{setting}.csv",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    numeric, source = [], []
    for model, table in MODELS.items():
        display = []
        for setting, label in SETTINGS.items():
            df = frames[model, setting]
            row = {"Preprocessing": label}
            source.append(df.assign(model=model, preprocessing=setting))
            for metric in METRICS:
                mean = float(df[metric].mean())
                sd = float(df[metric].std(ddof=1))
                row[metric] = f"{mean:.3f} ± {sd:.3f}"
                numeric.append({"table": table, "model": model,
                    "preprocessing": setting, "metric": metric,
                    "n_folds": 10, "mean": mean, "sd": sd})
            display.append(row)
        pd.DataFrame(display).to_csv(args.output_dir / f"Table_{table}_mean_sd.csv",
                                    index=False, encoding="utf-8-sig")

    pd.DataFrame(numeric).to_csv(args.output_dir / "all_metrics_numeric.csv",
                                index=False, encoding="utf-8-sig")
    pd.concat(source, ignore_index=True).to_csv(args.output_dir / "all_fold_metrics.csv",
                                               index=False, encoding="utf-8-sig")
    metadata = {"method": "Unweighted arithmetic mean and sample SD across ten folds",
                "ddof": 1, "display_decimal_places": 3,
                "numpy_version": np.__version__, "pandas_version": pd.__version__,
                "inputs": provenance}
    (args.output_dir / "summary_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tables S2–S4 saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
