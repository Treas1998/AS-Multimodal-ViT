
from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import sklearn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve
)


DEV_CSV = Path(
    r""
)
EXT_CSV = Path(
    r""
)
OUT_DIR = DEV_CSV.parent / "clinical_only_comparison"

N_BOOT = 2000
SEED = 42
DPI = 600


DEV_SEX_MAP = {1: 1, 2: 0}

CONT = ["age", "esr", "crp"]
CAT = ["sex", "b27"]
FEATURES = CONT + CAT


def read_csv_auto(path):
    for enc in ["utf-8-sig", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码：{path}")


def check_data(df, name):
    for col in FEATURES + ["label"]:
        if col not in df:
            raise ValueError(f"{name}缺少字段：{col}")
        df[col] = pd.to_numeric(df[col], errors="raise")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f"{name}的{col}存在无穷大。")

    if df["label"].isna().any() or not df["label"].isin([0, 1]).all():
        raise ValueError(f"{name}标签必须为0或1，且不能缺失。")

    for col in CAT:
        if not df[col].dropna().isin([0, 1]).all():
            raise ValueError(f"{name}的{col}编码不正确。")

    df["label"] = df["label"].astype(int)
    return df


def make_model():
    preprocessing = ColumnTransformer([
        ("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), CONT),
        ("categorical", SimpleImputer(strategy="most_frequent"), CAT)
    ])

    version = tuple(
        int(x) for x in re.match(r"(\d+)\.(\d+)", sklearn.__version__).groups()
    )
    if version >= (1, 8):
        no_penalty = {"C": np.inf}
    elif version >= (1, 2):
        no_penalty = {"penalty": None}
    else:
        no_penalty = {"penalty": "none"}

    regression = LogisticRegression(
        solver="lbfgs",
        max_iter=10000,
        tol=1e-8,
        class_weight=None,
        **no_penalty
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", regression)
    ])


def metrics(y, p):
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p))
    }


def save_figure(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(
            OUT_DIR / f"{name}.{ext}",
            dpi=DPI, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)


def grouped_calibration(y, p, model_name):
    data = pd.DataFrame({"label": y, "prob": p})
    group = pd.qcut(p, q=5, labels=False, duplicates="drop")
    if pd.isna(group).all():
        group = np.zeros(len(p), dtype=int)
    data["group"] = group

    table = data.groupby("group", sort=True).agg(
        N=("label", "size"),
        AS_count=("label", "sum"),
        mean_probability=("prob", "mean"),
        observed_proportion=("label", "mean"),
        min_probability=("prob", "min"),
        max_probability=("prob", "max")
    ).reset_index()
    table["group"] = table["group"].astype(int) + 1
    table["model"] = model_name
    return table


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dev = read_csv_auto(DEV_CSV).rename(columns={
        "年龄": "age", "性别": "sex"
    })
    ext = read_csv_auto(EXT_CSV).rename(columns={
        "sex_bin": "sex", "ESR": "esr", "CRP": "crp", "B27": "b27"
    })


    dev["sex"] = pd.to_numeric(dev["sex"], errors="raise")
    if not dev["sex"].dropna().isin(DEV_SEX_MAP).all():
        raise ValueError("开发队列性别编码与DEV_SEX_MAP不符。")
    dev["sex"] = dev["sex"].map(DEV_SEX_MAP)

    dev = check_data(dev, "开发队列")
    ext = check_data(ext, "外部队列")

    if len(dev) != 1178 or dev["label"].sum() != 691:
        raise ValueError("开发队列人数与预期1178人、691例AS不符。")
    if len(ext) != 255 or ext["label"].sum() != 66:
        raise ValueError("外部队列人数与预期255人、66例AS不符。")

    if ext["patient_id"].isna().any():
        raise ValueError("外部患者标识缺失。")
    ext["patient_id"] = ext["patient_id"].astype(str).str.strip()
    if (ext["patient_id"] == "").any() or ext["patient_id"].duplicated().any():
        raise ValueError("外部患者标识为空或重复。")

    p_multi = pd.to_numeric(ext["prob_AS"], errors="raise").to_numpy()
    if not np.isfinite(p_multi).all() or not ((p_multi >= 0) & (p_multi <= 1)).all():
        raise ValueError("多模态预测概率缺失或超出[0,1]。")


    if dev.duplicated(["image_path", "label"]).any():
        raise ValueError("开发表存在相同文件名和标签的重复记录，请核查。")

    dev[FEATURES].isna().sum().rename("missing_count").to_csv(
        OUT_DIR / "development_missing_counts.csv",
        encoding="utf-8-sig"
    )


    model = make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(dev[FEATURES], dev["label"])

    p_clinical = model.predict_proba(ext[FEATURES])[:, 1]
    y = ext["label"].to_numpy()

    predictions = ext[["patient_id", "label"]].copy()
    predictions["prob_clinical"] = p_clinical
    predictions["prob_multimodal"] = p_multi
    predictions.to_csv(
        OUT_DIR / "external_paired_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    joblib.dump(model, OUT_DIR / "clinical_model.joblib")


    prep = model.named_steps["preprocessing"]
    reg = model.named_steps["regression"]
    pd.DataFrame({
        "term": ["intercept"] + FEATURES,
        "coefficient": np.r_[reg.intercept_, reg.coef_[0]]
    }).to_csv(
        OUT_DIR / "clinical_coefficients.csv",
        index=False, encoding="utf-8-sig"
    )

    continuous = prep.named_transformers_["continuous"]
    pd.DataFrame({
        "variable": CONT,
        "imputation_median": continuous.named_steps["imputer"].statistics_,
        "mean_after_imputation": continuous.named_steps["scaler"].mean_,
        "std_after_imputation": continuous.named_steps["scaler"].scale_
    }).to_csv(
        OUT_DIR / "continuous_preprocessing.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "variable": CAT,
        "imputation_mode": prep.named_transformers_["categorical"].statistics_
    }).to_csv(
        OUT_DIR / "categorical_preprocessing.csv",
        index=False, encoding="utf-8-sig"
    )

    probs = {"Clinical LR": p_clinical, "Multimodal ViT": p_multi}
    point = {name: metrics(y, p) for name, p in probs.items()}
    draws = {
        name: {metric: [] for metric in ["AUC", "AP", "Brier"]}
        for name in probs
    }
    delta_auc = []
    rng = np.random.default_rng(SEED)
    skipped = 0

    # bootstrap
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue

        sampled = {}
        for name, p in probs.items():
            sampled[name] = metrics(yy, p[idx])
            for metric, value in sampled[name].items():
                draws[name][metric].append(value)
        delta_auc.append(
            sampled["Multimodal ViT"]["AUC"] - sampled["Clinical LR"]["AUC"]
        )

    if len(delta_auc) < 100:
        raise RuntimeError("有效bootstrap次数不足。")

    rows = []
    for name in probs:
        for metric in ["AUC", "AP", "Brier"]:
            low, high = np.percentile(draws[name][metric], [2.5, 97.5])
            rows.append({
                "model": name,
                "metric": metric,
                "estimate": point[name][metric],
                "CI95_low": low,
                "CI95_high": high
            })
    results = pd.DataFrame(rows)
    results.to_csv(
        OUT_DIR / "external_metrics_with_ci.csv",
        index=False, encoding="utf-8-sig"
    )

    low, high = np.percentile(delta_auc, [2.5, 97.5])
    difference = pd.DataFrame([{
        "comparison": "Multimodal ViT minus Clinical LR",
        "delta_AUC": point["Multimodal ViT"]["AUC"] - point["Clinical LR"]["AUC"],
        "CI95_low": low,
        "CI95_high": high,
        "valid_bootstrap": len(delta_auc),
        "skipped_bootstrap": skipped,
        "seed": SEED
    }])
    difference.to_csv(
        OUT_DIR / "paired_auc_difference.csv",
        index=False, encoding="utf-8-sig"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "pdf.fonttype": 42
    })
    colors = {"Clinical LR": "#D6604D", "Multimodal ViT": "#2166AC"}

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(
            fpr, tpr, color=colors[name], lw=1.8,
            label=f"{name} (AUC = {point[name]['AUC']:.3f})"
        )
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.set(
        xlabel="False positive rate", ylabel="True positive rate",
        xlim=(0, 1), ylim=(0, 1)
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "external_ROC_comparison")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    tables = []
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="Ideal calibration")
    for name, p in probs.items():
        table = grouped_calibration(y, p, name)
        tables.append(table)
        ax.plot(
            table["mean_probability"], table["observed_proportion"],
            "-o", color=colors[name], label=name
        )
    ax.set(
        xlabel="Mean predicted AS probability",
        ylabel="Observed AS proportion",
        xlim=(-0.02, 1.02), ylim=(-0.02, 1.02)
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "external_calibration_comparison")

    pd.concat(tables, ignore_index=True).to_csv(
        OUT_DIR / "calibration_groups_comparison.csv",
        index=False, encoding="utf-8-sig"
    )
    (OUT_DIR / "analysis_settings.txt").write_text(
        f"scikit-learn={sklearn.__version__}\n"
        f"Development N={len(dev)}; external N={len(ext)}\n"
        "Unpenalized logistic regression; five prespecified main effects.\n"
        "No class weighting; no external fitting or tuning.\n"
        "Continuous variables: development median imputation and scaling.\n"
        "Categorical variables: development mode imputation.\n"
        f"Paired patient bootstrap={N_BOOT}; seed={SEED}.\n"
        "Intervals conditional on fitted models; models not refitted in bootstrap.\n",
        encoding="utf-8"
    )

    print("\n外部性能：")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n配对AUC差值：")
    print(difference.to_string(index=False))
    print(f"\n完成，文件保存至：{OUT_DIR}")


if __name__ == "__main__":
    main()