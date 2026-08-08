"""pytorchexample: A Flower / PyTorch app."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



class CSVFeatureDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, synthetic_size: int = 100):
        self.csv_path = csv_path
        self.transform = transform
        self.synthetic = False
        self.synthetic_size = synthetic_size

        path = Path(csv_path)
        if not path.exists():
            self.synthetic = True
            self._generate_synthetic_data()
            return

        self.df = pd.read_csv(csv_path)
        self.has_image_path = "img_path" in self.df.columns
        if not self.has_image_path:
            self.feature_columns = [c for c in self.df.columns if c != "label"]
        if self.df.empty:
            self.synthetic = True
            self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        self.synthetic_images = np.random.rand(self.synthetic_size, 3, 32, 32).astype(np.float32)
        self.synthetic_labels = np.random.randint(0, 10, size=self.synthetic_size, dtype=np.int64)

    def __len__(self):
        return self.synthetic_size if self.synthetic else len(self.df)

    def __getitem__(self, idx):
        if self.synthetic:
            image = torch.from_numpy(self.synthetic_images[idx])
            label = int(self.synthetic_labels[idx])
            return {"img": image, "label": label}

        row = self.df.iloc[idx]
        label = int(row["label"])

        if self.has_image_path:
            image_path = row["img_path"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(self.csv_path), image_path)
            if not os.path.exists(image_path):
                image = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
            else:
                image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            values = row[self.feature_columns].to_numpy(dtype=np.float32)
            if values.size == 0:
                values = np.random.rand(3 * 32 * 32).astype(np.float32)
            image = torch.from_numpy(values)
            if image.numel() == 3 * 32 * 32:
                image = image.view(3, 32, 32)

        return {"img": image, "label": label}


def load_client_data(data_dir: str, batch_size: int):
    """Load local CSV client data from the mounted data directory."""
    train_csv = Path(data_dir) / "train.csv"
    val_csv = Path(data_dir) / "val.csv"

    train_dataset = CSVFeatureDataset(str(train_csv), transform=pytorch_transforms)
    val_dataset = CSVFeatureDataset(str(val_csv), transform=pytorch_transforms)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size)
    return trainloader, testloader


def load_server_data(data_dir: str, batch_size: int):
    """Load test set and return dataloader."""
    val_csv = Path(data_dir) / "val.csv"
    val_dataset = CSVFeatureDataset(str(val_csv), transform=pytorch_transforms)

    testloader = DataLoader(val_dataset, batch_size=batch_size)
    return testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


