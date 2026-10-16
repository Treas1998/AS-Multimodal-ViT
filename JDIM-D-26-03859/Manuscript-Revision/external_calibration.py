from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


IN_CSV = Path(
    r""
)
OUT_DIR = IN_CSV.parent / "calibration_results"

ID_COL = "patient_id"
LABEL_COL = "label"
PROB_COL = "prob_AS"

N_GROUPS = 5
N_BOOT = 2000
SEED = 42
DPI = 600


def read_csv_auto(path):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(
                path, encoding=encoding, dtype={ID_COL: str}
            )
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别CSV编码，请另存为UTF-8格式。")


def validate_data(df):
    required = [ID_COL, LABEL_COL, PROB_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少字段：{missing}")

    df = df[required].copy()
    if df[required].isna().any().any():
        raise ValueError("存在缺失值，请检查原始文件；脚本不会自动删除患者。")

    df[ID_COL] = df[ID_COL].str.strip()
    if (df[ID_COL] == "").any():
        raise ValueError("存在空患者标识。")
    if df[ID_COL].duplicated().any():
        raise ValueError("存在重复患者，请先核对，不能重复纳入。")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise")
    df[PROB_COL] = pd.to_numeric(df[PROB_COL], errors="raise")

    if not df[LABEL_COL].isin([0, 1]).all():
        raise ValueError("label必须为0或1。")
    if not np.isfinite(df[PROB_COL]).all():
        raise ValueError("预测概率包含无穷大等非法值。")
    if not df[PROB_COL].between(0, 1).all():
        raise ValueError("预测概率不在[0,1]内，请核对；不会自动截断。")

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    n = len(df)
    n_as = int(df[LABEL_COL].sum())
    print(f"患者数：{n}；AS：{n_as}；非AS：{n - n_as}")


    if n != 255 or n_as != 66:
        raise ValueError(
            f"当前数据为N={n}、AS={n_as}，与预期255人、66例AS不符。"
        )
    return df


# ================= Brier score及bootstrap CI =================
def bootstrap_brier(y, p, n_boot, seed):
    squared_errors = (p - y) ** 2
    estimate = float(squared_errors.mean())

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        values[i] = squared_errors[idx].mean()

    low, high = np.percentile(values, [2.5, 97.5])
    return estimate, float(low), float(high), values



def calibration_groups(df, n_groups):

    groups, edges = pd.qcut(
        df[PROB_COL],
        q=n_groups,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    if groups.isna().all():

        groups = pd.Series(0, index=df.index)

    work = df.assign(calibration_group=groups.astype(int) + 1)
    records = []

    for group, sub in work.groupby("calibration_group", sort=True):
        records.append({
            "group": int(group),
            "N": int(len(sub)),
            "AS_count": int(sub[LABEL_COL].sum()),
            "mean_predicted_probability": float(sub[PROB_COL].mean()),
            "observed_AS_proportion": float(sub[LABEL_COL].mean()),
            "min_predicted_probability": float(sub[PROB_COL].min()),
            "max_predicted_probability": float(sub[PROB_COL].max()),
        })

    return pd.DataFrame(records), edges



def plot_calibration(table, brier, low, high, out_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="0.55", linewidth=1.5,
        label="Ideal calibration",
    )

    x = table["mean_predicted_probability"].to_numpy()
    y = table["observed_AS_proportion"].to_numpy()

    ax.plot(
        x, y, "-o",
        color="#2166AC", linewidth=1.8, markersize=7,
        label="Grouped observed proportions",
        zorder=3,
    )

    for xi, yi, n in zip(x, y, table["N"]):
        offset = -16 if yi > 0.90 else 10
        ax.annotate(
            f"n = {n}",
            xy=(xi, yi),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=9,
        )

    ax.text(
        0.04, 0.96,
        f"Brier score = {brier:.3f}\n"
        f"95% CI: {low:.3f}–{high:.3f}",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="0.85", pad=5),
    )

    ax.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="Mean predicted AS probability",
        ylabel="Observed AS proportion",
    )
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        out_dir / "external_calibration.png",
        dpi=DPI, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        out_dir / "external_calibration.tiff",
        dpi=DPI, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        out_dir / "external_calibration.pdf",
        bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = validate_data(read_csv_auto(IN_CSV))

    y = df[LABEL_COL].to_numpy(dtype=float)
    p = df[PROB_COL].to_numpy(dtype=float)

    brier, low, high, boot_values = bootstrap_brier(
        y, p, N_BOOT, SEED
    )
    table, edges = calibration_groups(df, N_GROUPS)

    summary = {
        "N": len(df),
        "AS_count": int(y.sum()),
        "Non_AS_count": int(len(y) - y.sum()),
        "Observed_AS_proportion": float(y.mean()),
        "Mean_predicted_probability": float(p.mean()),
        "Brier_score": brier,
        "Brier_CI95_low": low,
        "Brier_CI95_high": high,
        "Bootstrap_repetitions": N_BOOT,
        "Bootstrap_seed": SEED,
        "Bootstrap_method": "Patient-level non-stratified percentile",
        "Requested_calibration_groups": N_GROUPS,
        "Actual_calibration_groups": len(table),
        "Quantile_edges": [float(v) for v in edges],
    }

    table.to_csv(
        OUT_DIR / "calibration_groups.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).drop(columns="Quantile_edges").to_csv(
        OUT_DIR / "calibration_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"Brier_score": boot_values}).to_csv(
        OUT_DIR / "bootstrap_brier_values.csv",
        index=False, encoding="utf-8-sig"
    )
    with open(OUT_DIR / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_calibration(table, brier, low, high, OUT_DIR)

    print("\n校准分组结果：")
    print(table.to_string(index=False))
    print(f"\nBrier score = {brier:.6f}")
    print(f"95% CI = [{low:.6f}, {high:.6f}]")
    print(f"\n文件已保存到：{OUT_DIR}")


if __name__ == "__main__":
    main()