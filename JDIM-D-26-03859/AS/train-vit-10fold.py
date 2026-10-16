import os
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import DatasetLoader
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return torch.mean(loss)


def train(model, device, dataloader, optimizer, criterion, epoch, log_path):
    model.train()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{i},{loss.item():.4f}\n')


def val(model, device, dataloader, epoch, log_path):
    model.eval()
    correct, total, probs, labels_all = 0, 0, [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, 1)[:, 1]
            _, pred = torch.max(outputs, 1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)
            probs.extend(prob.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = correct / total * 100
    auc = roc_auc_score(labels_all, probs)

    with open(log_path, 'a') as f:
        f.write(f'{epoch},{acc:.2f}\n')
    print(f'Val Epoch {epoch} - Acc: {acc:.2f}% - AUC: {auc:.4f}')
    return acc


def plot_curve(csv_file, save_path, title, ylabel):
    df = pd.read_csv(csv_file, header=None)
    if 'loss' in save_path:
        df.columns = ['epoch', 'iter', 'loss']
        data = df.groupby('epoch')['loss'].mean()
    else:
        df.columns = ['epoch', 'acc']
        data = df.set_index('epoch')['acc']

    plt.figure()
    data.plot()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(4, 11):
    print(f'\n===== Fold {fold} =====')
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads[-1] = nn.Linear(model.heads[-1].in_features, 2)
    model.to(device)

    # 路径适配
    csv_root = r''
    train_csv = os.path.join(csv_root, f'train_fold{fold}.csv')
    val_csv = os.path.join(csv_root, f'val_fold{fold}.csv')

    train_dataset = DatasetLoader(train_csv, train=True)
    val_dataset = DatasetLoader(val_csv, train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = FocalLoss()


    checkpoint_dir = f'checkpoint_gamma_fold{fold}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_path = os.path.join(checkpoint_dir, 'resume.pth')
    train_log = os.path.join(checkpoint_dir, 'train_log.csv')
    val_log = os.path.join(checkpoint_dir, 'val_log.csv')

    start_epoch, best_acc, epochs_no_improve, patience = 1, 0.0, 0, 20

    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f'Resumed from Epoch {start_epoch-1} Best Acc: {best_acc:.2f}%')

    early_stop = False
    for epoch in range(start_epoch, 1):
        if early_stop:
            print(' Early stopping')
            break

        train(model, device, train_loader, optimizer, criterion, epoch, train_log)

        if epoch % 4 == 0:
            acc = val(model, device, val_loader, epoch, val_log)
            scheduler.step(acc)

            if acc > best_acc:
                best_acc, epochs_no_improve = acc, 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model_fold{fold}_epoch{epoch}_acc{acc:.2f}.pth'))
                print(f'Saved Best Model - Fold {fold} Epoch {epoch}')
            else:
                epochs_no_improve += 1
                print(f'No Improvement for {epochs_no_improve} evaluations')

            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc,
                'epochs_no_improve': epochs_no_improve
            }, resume_path)

            if epochs_no_improve >= patience:
                early_stop = True

    plot_curve(train_log, os.path.join(checkpoint_dir, 'train_loss_curve.png'), 'Training Loss', 'Loss')
    plot_curve(val_log, os.path.join(checkpoint_dir, 'val_acc_curve.png'), 'Validation Accuracy', 'Accuracy')
