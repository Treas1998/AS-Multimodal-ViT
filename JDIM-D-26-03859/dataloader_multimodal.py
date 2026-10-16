import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class MultiModalDataset(Dataset):
    def __init__(self, csv_path, image_root, train=True, stats=None):

        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.train = train

        self.continuous_cols = ['年龄', 'esr', 'crp']
        self.categorical_cols = ['性别', 'b27']


        self.data[self.continuous_cols] = self.data[self.continuous_cols].astype(float)
        self.data[self.categorical_cols] = self.data[self.categorical_cols].astype(float)

        if train:
            self.means = self.data[self.continuous_cols].mean()
            self.stds = self.data[self.continuous_cols].std().replace(0, 1e-6)
        else:
            self.means = stats['mean']
            self.stds = stats['std']

        self.image_transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.image_transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


        self.tabular_dim = len(self.continuous_cols) + len(self.categorical_cols)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path = os.path.join(self.image_root, row['image_path'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        image = Image.open(img_path).convert('RGB')
        transform = self.image_transform_train if self.train else self.image_transform_test
        image = transform(image)


        cont_feats = torch.tensor(
            ((row[self.continuous_cols].astype(float) - self.means) / self.stds).values,
            dtype=torch.float32
        )
        cat_feats = torch.tensor(row[self.categorical_cols].astype(float).values, dtype=torch.float32)
        tabular = torch.cat([cont_feats, cat_feats], dim=0)

        label = int(row['label'])
        return image, tabular, label

    def get_stats(self):

        return {'mean': self.means, 'std': self.stds}
