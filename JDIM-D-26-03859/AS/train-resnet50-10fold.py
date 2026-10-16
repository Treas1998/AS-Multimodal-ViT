import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import DatasetLoader
from ptflops import get_model_complexity_info


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return torch.mean(F_loss) if self.reduction == 'mean' else F_loss


def train(model, device, train_dataloader, optimizer, criterion, epoch, log_file):
    model.train()
    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with open(log_file, "a") as f:
            f.write(f"{epoch},{iter},{loss.item():.4f}\n")


def val(model, device, val_dataloader, epoch, fold_dir):
    model.eval()
    test_probs, test_labels, pred_labels = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)

            test_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total * 100
    roc_auc = auc(*roc_curve(test_labels, test_probs)[:2])
    f1 = f1_score(test_labels, pred_labels)

    with open(os.path.join(fold_dir, "val_log.csv"), "a") as f:
        f.write(f"{epoch},{acc:.2f}\n")


    plt.figure()
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC - Epoch {epoch}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(os.path.join(fold_dir, f'roc_epoch{epoch}.png'))
    plt.close()

    cm = confusion_matrix(test_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(fold_dir, f'conf_matrix_epoch{epoch}.png'))
    plt.close()

    print(f'Val Epoch {epoch} - Acc: {acc:.2f}% - AUC: {roc_auc:.4f} - F1: {f1:.4f}')
    return acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    num_epochs = 300
    lr = 1e-4
    batch_size = 64
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_csv_path = os.path.join(os.getcwd(), "datasets-Gaussian Noise-10fold")

    for fold in range(1, 11):
        print(f"\n===== Fold {fold} =====")

        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        train_csv = os.path.join(root_csv_path, f"train_fold{fold}.csv")
        val_csv = os.path.join(root_csv_path, f"val_fold{fold}.csv")
        train_dataset = DatasetLoader(train_csv, train=True)
        val_dataset = DatasetLoader(val_csv, train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


        fold_dir = f"fold_{fold}"
        os.makedirs(fold_dir, exist_ok=True)
        params = count_parameters(model)
        flops, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
        with open(os.path.join(fold_dir, 'model_complexity.json'), 'w') as f:
            json.dump({'params': params, 'flops': flops}, f)

        # Optimizer + Scheduler
        criterion = FocalLoss(alpha=0.1, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        # Early stopping
        resume_path = os.path.join(fold_dir, "resume.pth")
        best_acc, patience, epochs_no_improve = 0.0, 20, 0
        start_epoch = 1

        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            best_acc = checkpoint["best_acc"]
            start_epoch = checkpoint["epoch"] + 1
            epochs_no_improve = checkpoint["epochs_no_improve"]
            print(f"Resumed from Epoch {start_epoch-1}, Best Acc: {best_acc:.2f}%")

        early_stop = False
        for epoch in range(start_epoch, num_epochs + 1):
            if early_stop:
                break

            train(model, device, train_loader, optimizer, criterion, epoch, os.path.join(fold_dir, "train_log.csv"))

            if epoch % 4 == 0:
                acc = val(model, device, val_loader, epoch, fold_dir)
                scheduler.step(acc)

                if acc > best_acc:

                    for f in os.listdir(fold_dir):
                        if f.startswith("best_model") and f.endswith(".pth"):
                            os.remove(os.path.join(fold_dir, f))

                    best_acc = acc
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), os.path.join(fold_dir, f"best_model_epoch{epoch}_acc{acc:.2f}.pth"))
                    print(f"Saved Best Model - Fold {fold} Epoch {epoch}")
                else:
                    epochs_no_improve += 1
                    print(f"No Improvement for {epochs_no_improve} evaluations")


                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "epochs_no_improve": epochs_no_improve
                }, resume_path)

                if epochs_no_improve >= patience:
                    early_stop = True


        def plot_curve(file, save_path, title, ylabel):
            if not os.path.exists(file):
                return
            df = pd.read_csv(file, header=None)
            plt.figure()
            if 'loss' in save_path:
                df.columns = ['epoch', 'iter', 'loss']
                df.groupby('epoch')['loss'].mean().plot()
            else:
                df.columns = ['epoch', 'acc']
                plt.plot(df['epoch'], df['acc'])
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.savefig(save_path)
            plt.close()

        plot_curve(os.path.join(fold_dir, "train_log.csv"), os.path.join(fold_dir, "train_loss.png"), "Train Loss", "Loss")
        plot_curve(os.path.join(fold_dir, "val_log.csv"), os.path.join(fold_dir, "val_acc.png"), "Val Accuracy", "Accuracy")
