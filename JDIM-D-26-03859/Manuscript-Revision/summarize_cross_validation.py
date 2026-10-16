import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd



EXPERIMENTS = [
    (
        "Unimodal ResNet50",
        "Original",
        "resnet50_original.csv",
    ),
    (
        "Unimodal ResNet50",
        "Gamma correction",
        "resnet50_gamma.csv",
    ),
    (
        "Unimodal ViT-B/16",
        "Original",
        "vit_original.csv",
    ),
    (
        "Unimodal ViT-B/16",
        "CLAHE",
        "vit_clahe.csv",
    ),
    (
        "Multimodal ViT-B/16",
        "Original",
        "multimodal_vit_original.csv",
    ),
    (
        "Multimodal ViT-B/16",
        "HE",
        "multimodal_vit_he.csv",
    ),
]

METRICS = {
    "accuracy": "ACC",
    "sensitivity": "SEN",
    "specificity": "SPE",
    "f1_score": "F1 score",
    "auc": "AUC",
}

N_FOLDS = 10
DECIMALS = 3
SD_DDOF = 1  # Sample SD: denominator = n - 1


def read_and_validate(path):
    """Read metrics without dropping or imputing any observations."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = ["fold"] + list(METRICS)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns: {missing}")

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: missing or non-finite values")

    if (
        len(df) != N_FOLDS
        or df["fold"].duplicated().any()
        or set(df["fold"]) != set(range(1, N_FOLDS + 1))
    ):
        raise ValueError(
            f"{path.name}: expected one row for each fold from 1 to 10"
        )

    for metric in METRICS:
        if not df[metric].between(0, 1).all():
            raise ValueError(
                f"{path.name}: {metric} must be between 0 and 1"
            )

    df["fold"] = df["fold"].astype(int)
    return df.sort_values("fold").reset_index(drop=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_dir / "results" / "fold_metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "table2_summary",
    )
    args = parser.parse_args()

    # Validate all inputs before writing results.
    datasets = []
    for model, preprocessing, filename in EXPERIMENTS:
        path = args.input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        datasets.append(
            (model, preprocessing, path, read_and_validate(path))
        )

    formatted_rows = []
    numeric_rows = []
    source_frames = []
    input_records = []

    for model, preprocessing, path, df in datasets:
        formatted = {
            "Model": model,
            "Image preprocessing": preprocessing,
        }

        for metric, display_name in METRICS.items():
            values = df[metric].to_numpy(dtype=float)

            # Equal weight for each validation fold.
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=SD_DDOF))

            formatted[display_name] = (
                f"{mean:.{DECIMALS}f} ± {sd:.{DECIMALS}f}"
            )

            numeric_rows.append({
                "model": model,
                "preprocessing": preprocessing,
                "metric": metric,
                "n_folds": len(values),
                "mean": mean,
                "sample_sd": sd,
            })

        formatted_rows.append(formatted)

        source = df.copy()
        source.insert(0, "preprocessing", preprocessing)
        source.insert(0, "model", model)
        source_frames.append(source)

        input_records.append({
            "model": model,
            "preprocessing": preprocessing,
            "filename": path.name,
            "sha256": sha256(path),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(formatted_rows)
    table.to_csv(
        args.output_dir / "table2_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(numeric_rows).to_csv(
        args.output_dir / "table2_numeric.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )

    pd.concat(source_frames, ignore_index=True).to_csv(
        args.output_dir / "table2_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )

    metadata = {
        "summary": "Unweighted arithmetic mean across validation folds",
        "variability": "Sample standard deviation across validation folds",
        "n_folds": N_FOLDS,
        "sd_ddof": SD_DDOF,
        "display_decimal_places": DECIMALS,
        "bootstrap_performed": False,
        "note": (
            "SD describes between-fold variability and is not a "
            "confidence interval. This script does not calculate "
            "AUC from pooled patient-level predictions."
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "inputs": input_records,
    }

    with open(
        args.output_dir / "table2_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(table.to_string(index=False))
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()