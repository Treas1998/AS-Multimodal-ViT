import os, glob, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, f1_score
from scipy.stats import bootstrap
from dataloader_multimodal import MultiModalDataset
from multimodal_vit import MultiModalViT_timm

def evaluate(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    youden = sens + spec - 1
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    return acc, sens, spec, youden, f1, auc, cm

def plot_roc(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR"), plt.ylabel("TPR")
    plt.title("ROC Curve"), plt.legend()
    plt.savefig(save_path), plt.close()

def plot_cm(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"), plt.ylabel("Actual"), plt.title("Confusion Matrix")
    plt.savefig(save_path), plt.close()


model_root = r""
val_csv_root = r""
image_root = r""
save_root = r""
os.makedirs(save_root, exist_ok=True)

results, all_labels, all_probs, all_preds = [], [], [], []

for fold in range(1, 11):
    print(f"Fold {fold} evaluating...")
    model_path = glob.glob(os.path.join(model_root, f"fold_{fold}", "best_model_*.pth"))[0]
    train_csv = os.path.join(val_csv_root, f"train_fold{fold}.csv")
    val_csv = os.path.join(val_csv_root, f"val_fold{fold}.csv")

    train_set = MultiModalDataset(train_csv, image_root, train=True)
    stats = train_set.get_stats()
    val_set = MultiModalDataset(val_csv, image_root, train=False, stats=stats)
    loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = MultiModalViT_timm(num_tabular_features=val_set.tabular_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for img, tab, label in loader:
            out = model(img, tab)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = torch.argmax(out, dim=1)
            y_true.append(label.item())
            y_pred.append(pred.item())
            y_prob.append(prob.item())

    acc, sens, spec, youden, f1, auc, cm = evaluate(y_true, y_pred, y_prob)
    fold_dir = os.path.join(save_root, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    plot_roc(y_true, y_prob, os.path.join(fold_dir, "roc.png"))
    plot_cm(cm, os.path.join(fold_dir, "confusion_matrix.png"))

    results.append({
        "fold": fold,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "youden_index": youden,
        "f1_score": f1,
        "auc": auc
    })
    all_labels += y_true
    all_preds += y_pred
    all_probs += y_prob


df = pd.DataFrame(results)
df.to_csv(os.path.join(save_root, "val_summary_metrics.csv"), index=False)


def compute_ci(data):
    res = bootstrap((np.array(data),), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high

summary_ci = []
for metric in ["accuracy", "sensitivity", "specificity", "youden_index", "f1_score", "auc"]:
    vals = df[metric].values
    mean = np.mean(vals)
    low, high = compute_ci(vals)
    summary_ci.append({
        "metric": metric,
        "mean": round(mean, 4),
        "95% CI": f"[{round(low, 4)}, {round(high, 4)}]"
    })

pd.DataFrame(summary_ci).to_csv(os.path.join(save_root, "val_summary_with_ci.csv"), index=False)


plot_roc(all_labels, all_probs, os.path.join(save_root, "fusion_roc.png"))
plot_cm(confusion_matrix(all_labels, all_preds), os.path.join(save_root, "fusion_confusion_matrix.png"))

print("done")
