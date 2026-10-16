import csv
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DatasetLoader(Dataset):
    def __init__(self, csv_path, train=True):
        with open(csv_path, 'r') as file:
            self.data = list(csv.reader(file))

        self.train = train
        self.root_dir = os.path.dirname(csv_path)

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image_path, label = self.data[index]
        full_path = os.path.join(r"", image_path)
        image = Image.open(full_path).convert("RGB")

        if self.train:
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)

        return image, int(label)

    def __len__(self):
        return len(self.data)
