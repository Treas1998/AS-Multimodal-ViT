import os, json, torch, random
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multimodal_vit import MultiModalViT_timm
from dataloader_multimodal import MultiModalDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def train(model, dataloader, optimizer, criterion, device, log_path, epoch):
    model.train()
    for i, (img, tab, label) in enumerate(dataloader):
        img, tab, label = img.to(device), tab.to(device), label.to(device)
        out = model(img, tab)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with open(log_path, "a") as f:
            f.write(f"{epoch},{i},{loss.item():.4f}\n")

def validate(model, dataloader, device, epoch, fold_dir):
    model.eval()
    all_labels, all_probs, preds = [], [], []
    with torch.no_grad():
        for img, tab, label in dataloader:
            img, tab = img.to(device), tab.to(device)
            out = model(img, tab)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = torch.argmax(out, dim=1)
            all_labels.append(label.item())
            all_probs.append(prob.item())
            preds.append(pred.item())

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc_score = auc(*roc_curve(all_labels, all_probs)[:2])

    with open(os.path.join(fold_dir, "val_log.csv"), "a") as f:
        f.write(f"{epoch},{acc*100:.2f}\n")

    # ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR"), plt.ylabel("TPR")
    plt.title(f"ROC - Epoch {epoch}")
    plt.legend()
    plt.savefig(os.path.join(fold_dir, f"roc_epoch{epoch}.png"))
    plt.close()

    # Confusion
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted"), plt.ylabel("Actual")
    plt.savefig(os.path.join(fold_dir, f"conf_matrix_epoch{epoch}.png"))
    plt.close()

    return acc

def plot_logs(file, save_path, title, ylabel):
    if not os.path.exists(file): return
    df = pd.read_csv(file, header=None)
    plt.figure()
    if 'loss' in save_path:
        df.columns = ['epoch', 'iter', 'loss']
        df.groupby('epoch')['loss'].mean().plot()
    else:
        df.columns = ['epoch', 'acc']
        plt.plot(df['epoch'], df['acc'])
    plt.title(title), plt.xlabel("Epoch"), plt.ylabel(ylabel), plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    num_epochs, lr, batch_size = 300, 1e-4, 32
    patience = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = r""
    image_root = r""

    for fold in range(2, 11):
        print(f"\n=== Fold {fold} ===")
        fold_dir = f"fold_{fold}"
        os.makedirs(fold_dir, exist_ok=True)

        train_csv = os.path.join(root, f"train_fold{fold}.csv")
        val_csv = os.path.join(root, f"val_fold{fold}.csv")
        train_set = MultiModalDataset(train_csv, image_root, train=True)
        stats = train_set.get_stats()
        val_set = MultiModalDataset(val_csv, image_root, train=False, stats=stats)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)


        model = MultiModalViT_timm(
            num_tabular_features=train_set.tabular_dim,
            local_vit_weight_path=r""
        ).to(device)

        params = count_parameters(model)
        json.dump({'params': params}, open(os.path.join(fold_dir, 'model_complexity.json'), 'w'))

        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        best_acc, epochs_no_improve, start_epoch = 0.0, 0, 1
        resume_path = os.path.join(fold_dir, "resume.pth")
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            best_acc = checkpoint["best_acc"]
            start_epoch = checkpoint["epoch"] + 1
            epochs_no_improve = checkpoint["epochs_no_improve"]

        early_stop = False
        for epoch in range(start_epoch, num_epochs + 1):
            if early_stop: break
            train(model, train_loader, optimizer, criterion, device, os.path.join(fold_dir, "train_log.csv"), epoch)
            if epoch % 4 == 0:
                acc = validate(model, val_loader, device, epoch, fold_dir)
                scheduler.step(acc)
                if acc > best_acc:
                    for f in os.listdir(fold_dir):
                        if f.startswith("best_model") and f.endswith(".pth"):
                            os.remove(os.path.join(fold_dir, f))
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join(fold_dir, f"best_model_epoch{epoch}_acc{acc*100:.2f}.pth"))
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "epochs_no_improve": epochs_no_improve
                }, resume_path)
                if epochs_no_improve >= patience:
                    early_stop = True

        plot_logs(os.path.join(fold_dir, "train_log.csv"), os.path.join(fold_dir, "train_loss.png"), "Train Loss", "Loss")
        plot_logs(os.path.join(fold_dir, "val_log.csv"), os.path.join(fold_dir, "val_acc.png"), "Val Accuracy", "Accuracy")
