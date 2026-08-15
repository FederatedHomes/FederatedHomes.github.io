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


TRAIN_DATA_FILE = os.environ.get("TRAIN_DATA_FILE", "train.csv")
VAL_DATA_FILE = os.environ.get("VAL_DATA_FILE", "val.csv")


def load_client_data(data_dir: str, batch_size: int):
    """Load local CSV client data from the mounted data directory."""
    train_csv = Path(data_dir) / TRAIN_DATA_FILE
    val_csv = Path(data_dir) / VAL_DATA_FILE

    train_dataset = CSVFeatureDataset(str(train_csv), transform=pytorch_transforms)
    val_dataset = CSVFeatureDataset(str(val_csv), transform=pytorch_transforms)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size)
    return trainloader, testloader


def load_server_data(data_dir: str, batch_size: int):
    """Load test set and return dataloader."""
    val_csv = Path(data_dir) / VAL_DATA_FILE
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




from torch.utils.data import DataLoader, IterableDataset
class StreamingDataset(IterableDataset):
    def __init__(self, csv_path: str, segment_length: int, overlap_fraction: float, num_scales: int, label_columns: list):
        """
        Initialize the StreamingDataset with the buffer parameters.
        Does not load the CSV into memory, but reads it in chunks during iteration.
        """
        self.csv_path = csv_path
        self.segment_length = segment_length
        self.overlap_fraction = overlap_fraction
        self.num_scales = num_scales
        self.label_columns = label_columns

        # calculate how many rows to shift forward  for each segment based on the overlap fraction
        # e.g. if segment_length=100 and overlap_fraction=0.5, then shift_length=50
        self.step_size = int(self.segment_length * (1 - self.overlap_fraction))

        # Guard against zero or negative step sizes (prevent infinite loops)
        if self.step_size <= 0:
            self.step_size = 1

    def create_feature_and_label(self, chunk):
        """
        Processes a pandas Dataframe chunk of length segment_length.
        Handles missing values and returns a numpy array of shape (segment_length, num_features) and the corresponding label.
        """
        # identify the columns that are not label columns
        feature_columns = [c for c in chunk.columns if c not in self.label_columns]
        self.num_features = len(feature_columns)
        # create dict of column indices to feature names for consistent ordering
        self.dict_idx_feature = {i: col for i, col in enumerate(feature_columns)}

        # forward fill, backward fill, and then fill remaining NaNs with 0 in the feature columns
        chunk[feature_columns] = chunk[feature_columns].ffill().bfill().fillna(0)

        # create a single label by taking the mode of the label column in the segment
        label = chunk[self.label_columns].mode().iloc[0].to_numpy()

        # convert the chunk to a numpy array
        segment_array = chunk.to_numpy()

        segment_cwt_matrix = self.create_cwt_matrix(segment_array, wavelet_name='morl', rescale_size=self.num_scales, log_scale=True)
        # dummy_feature_array = np.random.rand(self.num_scales, self.num_scales, self.num_features)  # Placeholder: replace with actual feature extraction logic

        return segment_cwt_matrix, label

    def create_cwt_matrix(self, segment_array, wavelet_name='morl', rescale_size=128, log_scale=False):
        import pywt
        from skimage.transform import resize

        # range of scales for CWT
        scales = np.arange(1, self.num_scales + 1)

        if log_scale:
            scales = np.logspace(np.log(1), np.log(self.num_scales+1), num=self.num_scales, base=np.e, dtype=np.float32)

        # preallocate matrix for CWT images
        cwt_matrix = np.ndarray(shape=(rescale_size, rescale_size, self.num_features), dtype=np.float32)

        for signal_idx,signal_name in self.dict_idx_feature.items():
            signal_data = segment_array[:, signal_idx]
            coeffs, freqs = pywt.cwt(signal_data, scales, wavelet_name)
            coeffs_resized = resize(coeffs, (rescale_size, rescale_size), mode='constant', anti_aliasing=True)
            cwt_matrix[:, :, signal_idx] = coeffs_resized

        return cwt_matrix


    def __iter__(self):
        """
        The core streaming logic: 
        - reads the CSV in chunks, 
        - processes each chunk into overlapping segments, and 
        - yields processed tensors as feature-label pairs one at a time.
        """
        csv_stream = pd.read_csv(self.csv_path, chunksize=self.segment_length, index_col="timestamp")
        for chunk in csv_stream:
            # Append the newly read data to the sliding buffer
            buffer = pd.concat([buffer, chunk], ignore_index=True) if 'buffer' in locals() else chunk

            # While the buffer has enough data to create a segment, process it
            while len(buffer) >= self.segment_length:
                # Process the first segment_length rows sorted by column index to ensure consistent feature ordering
                buffer_segment = buffer.iloc[:self.segment_length].sort_index(axis=1)
                feature, label = self.create_feature_and_label(buffer_segment)

                # Convert to pytorch tensors moving channels to the front
                feature_tensor = torch.tensor(feature, dtype=torch.float32).permute(2, 0, 1)
                label_tensor = torch.tensor(label, dtype=torch.long)

                # Yield the feature-label pair as a dictionary
                yield {
                    "data": buffer_segment[list(self.dict_idx_feature.values())].to_numpy(), 
                    "dict_idx_feature": self.dict_idx_feature, 
                    "feature_tensor": feature_tensor, 
                    "label_tensor": label_tensor
                }

                # Slide the buffer forward by step_size
                buffer = buffer.iloc[self.step_size:].reset_index(drop=True)


class Utilities:
    @staticmethod
    def split_indices_per_label(y):

        """
        Splits the indices of the input array y into separate lists based on unique labels.
        Returns a list of arrays, where each array contains the indices corresponding to a unique label.
        """
        unique_labels = np.unique(y)
        label_indices = {str(label): np.where(y == label)[0] for label in unique_labels}
        print(f"Split indices into {len(unique_labels)} unique labels: {list(label_indices.keys())}")
        return label_indices

    @staticmethod
    def plot_cwt_coeffs_per_label(X, y, signal, sample, scales, wavelet):
        import matplotlib.pyplot as plt
        import pywt

        unique_labels = np.unique(y)
        label_indices = {str(label): np.where(y == label)[0] for label in unique_labels}
        print(f"Split indices into {len(unique_labels)} unique labels: {list(label_indices.keys())}")

        n_labels = len(label_indices)
        fig, axs = plt.subplots(
            nrows=2,
            ncols=n_labels,
            sharex=True,
            sharey="row",
            figsize=(8, 5 * n_labels)
        )
        vmin_val = None
        vmax_val = None

        # display the scalogram for each label in the first row of subplots
        for ax, indices, name in zip(axs[0], label_indices.values(), label_indices.keys()):
            coeffs, freqs = pywt.cwt(X[indices[sample], :, signal], scales, wavelet)
            vmin_val = coeffs.min() if vmin_val is None else vmin_val
            vmax_val = coeffs.max() if vmax_val is None else vmax_val

            im = ax.imshow(coeffs, cmap='coolwarm', aspect='auto', vmin=vmin_val, vmax=vmax_val)
            ax.set_title(name)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.03, location='top')
            cbar_ticks = [coeffs.min(), coeffs.mean(), coeffs.max()]
            cbar.set_ticks(cbar_ticks)
            cbar.ax.set_xticklabels([f"{tick:.0f}" for tick in cbar_ticks])
            cbar.ax.set_xlabel('Signal Intensity', rotation=0, va='top', labelpad=15)
        axs[0, 0].set_ylabel('Scale')

        # display the original timeseries signal below the scalograms
        for ax, indices, name in zip(axs[1], label_indices.values(), label_indices.keys()):
            ax.plot(X[indices[sample], :, signal])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time')
        axs[1, 0].set_ylabel('Signal Value')

        fig.tight_layout()

        path_filename = os.path.join(os.getenv("DATA_DIR"), f"cwt_coeffs_signal_{signal}_sample_{sample}.png")
        plt.savefig(path_filename, dpi=300)
        


if __name__ == "__main__":
    # Example usage of StreamingDataset
    CSV_FILE_PATH = os.path.join(os.getenv("DATA_DIR"), "train.csv")  # Path to your CSV file
    SEGMENT_LENGTH = 10
    OVERLAP_FRACTION = 0.5
    NUM_SCALES = 32
    LABEL_COLUMNS = ["Sensor_5"]  # Adjust based on your CSV structure
    BATCH_SIZE = 5

    dataset = StreamingDataset(
        csv_path=CSV_FILE_PATH,
        segment_length=SEGMENT_LENGTH,
        overlap_fraction=OVERLAP_FRACTION,
        num_scales=NUM_SCALES,
        label_columns=LABEL_COLUMNS
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    utilities = Utilities()
    plotting_enabled = True  # Set to False to disable plotting
    for batch in dataloader:
        print(batch["data"].shape, batch["feature_tensor"].shape, batch["label_tensor"].shape)
        print(f"Feature names: {batch['dict_idx_feature']}")
        if plotting_enabled:
            utilities.plot_cwt_coeffs_per_label(
                X=batch["data"].numpy(),
                y=batch["label_tensor"].numpy(),
                signal=0,  # Change this to visualize different signals
                sample=0,  # Change this to visualize different samples
                scales=np.arange(1, NUM_SCALES + 1),
                wavelet='morl'
            )
            plotting_enabled = False  # Disable further plotting after the first batch


    
