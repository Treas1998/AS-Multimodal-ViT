import os
import glob
import torch
import numpy as np
from scipy.stats import bootstrap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
from dataloader import DatasetLoader


def evaluate_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    youden = sensitivity + specificity - 1
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    return acc, sensitivity, specificity, youden, f1, auc, cm


def plot_roc(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)


    fig, ax = plt.subplots(figsize=(3.2, 3.0), layout="constrained")

    ax.plot(
        fpr, tpr,
        color="#0072B2",
        linewidth=1.5,
        label=f"AUC = {auc_score:.4f}"
    )
    ax.plot(
        [0, 1], [0, 1],
        color="#E69F00",
        linestyle="--",
        linewidth=1.0
    )

    ax.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    stem = os.path.splitext(save_path)[0]
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"{os.path.basename(stem)}: AUC = {auc_score:.8f}")


def plot_confusion(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()



model_root = r""
val_csv_root = r""
save_root = r""
os.makedirs(save_root, exist_ok=True)

results = []
all_labels, all_probs, all_preds = [], [], []

for fold in range(1, 11):
    print(f"Evaluating Fold {fold}...")

    model_path = glob.glob(os.path.join(model_root, f"fold_{fold}", "best_model_*.pth"))[0]
    val_csv = os.path.join(val_csv_root, f"val_fold{fold}.csv")

    dataset = DatasetLoader(val_csv, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for x, label in loader:
            out = model(x)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = torch.argmax(out, dim=1)
            y_true.append(label.item())
            y_pred.append(pred.item())
            y_probs.append(prob.item())

    acc, sens, spec, youden, f1, auc, cm = evaluate_metrics(y_true, y_pred, y_probs)

    fold_dir = os.path.join(save_root, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    plot_roc(y_true, y_probs, os.path.join(fold_dir, "roc.png"))
    plot_confusion(cm, os.path.join(fold_dir, "confusion_matrix.png"))

    results.append({
        "fold": fold,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "youden_index": youden,
        "f1_score": f1,
        "auc": auc
    })

    all_labels.extend(y_true)
    all_probs.extend(y_probs)
    all_preds.extend(y_pred)


df = pd.DataFrame(results)
df.to_csv(os.path.join(save_root, "val_summary_metrics.csv"), index=False)


def compute_ci(data, func=np.mean):
    res = bootstrap((np.array(data),), func, confidence_level=0.95, n_resamples=1000, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high

summary_with_ci = []
metrics = ['accuracy', 'sensitivity', 'specificity', 'youden_index', 'f1_score', 'auc']

for metric in metrics:
    values = df[metric].values
    mean_val = np.mean(values)
    ci_low, ci_high = compute_ci(values)
    summary_with_ci.append({
        "metric": metric,
        "mean": round(mean_val, 4),
        "95% CI": f"[{round(ci_low, 4)}, {round(ci_high, 4)}]"
    })

summary_df = pd.DataFrame(summary_with_ci)
summary_df.to_csv(os.path.join(save_root, "val_summary_with_ci.csv"), index=False)


plot_roc(all_labels, all_probs, os.path.join(save_root, "fusion_roc.png"))
plot_confusion(confusion_matrix(all_labels, all_preds), os.path.join(save_root, "fusion_confusion_matrix.png"))

print("完成：已保存所有指标、ROC 和混淆矩阵图像。")
