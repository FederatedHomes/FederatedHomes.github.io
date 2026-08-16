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


TRAIN_DATA_FILE = os.environ.get("TRAIN_DATA_FILE", "train.csv")
VAL_DATA_FILE = os.environ.get("VAL_DATA_FILE", "val.csv")
SEGMENT_LENGTH = int(os.environ.get("SEGMENT_LENGTH", 5))
OVERLAP_FRACTION = float(os.environ.get("OVERLAP_FRACTION", 0.5))
NUM_SCALES = int(os.environ.get("NUM_SCALES", 32))
LABEL_COLUMN = os.environ.get("LABEL_COLUMN", "Sensor_5")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 30))

class CustomNet(nn.Module):
    """CNN tuned for the StreamingDataset CWT feature tensors."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes, )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# fds = None  # Cache FederatedDataset
# pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_client_data(data_dir: str):
    """Load local CSV client data from the mounted data directory."""
    train_csv = Path(data_dir) / TRAIN_DATA_FILE
    val_csv = Path(data_dir) / VAL_DATA_FILE


    train_dataset = StreamingDataset(
        csv_path=str(train_csv),
        segment_length=SEGMENT_LENGTH,
        overlap_fraction=OVERLAP_FRACTION,
        num_scales=NUM_SCALES,
        label_column=LABEL_COLUMN
    )

    val_dataset = StreamingDataset(
            csv_path=str(val_csv),
            segment_length=SEGMENT_LENGTH,
            overlap_fraction=OVERLAP_FRACTION,
            num_scales=NUM_SCALES,
            label_column=LABEL_COLUMN
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader


def load_server_data(data_dir: str):
    """Load test set and return dataloader."""
    val_csv = Path(data_dir) / VAL_DATA_FILE
    val_dataset = StreamingDataset(
        csv_path=str(val_csv),
        segment_length=SEGMENT_LENGTH,
        overlap_fraction=OVERLAP_FRACTION,
        num_scales=NUM_SCALES,
        label_column=LABEL_COLUMN
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return val_loader


def train_model(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        batches = list(trainloader)
        for batch_index, batch in enumerate(batches):
            features = batch["feature_tensor"].to(device)
            labels = batch["label_tensor"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Visualize CWT coefficients for the first and last batch of the first epoch
            is_first_batch = batch_index == 0
            is_last_batch = batch_index == len(batches) - 1
            if is_first_batch or is_last_batch and epoch == 0:
                signal_dict = {i: name[0] for i, name in batch["dict_idx_feature"].items()}

                Utilities().plot_cwt_coeffs_per_label(
                    X=batch["data"].numpy(),
                    y=batch["label_tensor"].numpy(),
                    signal=np.random.choice(list(signal_dict.keys())),  # Choose a random signal to visualize
                    scales=np.arange(1, NUM_SCALES + 1),
                    wavelet='morl',
                    filename_prefix=f"batch_{batch_index}",
                    signal_dict=signal_dict
                )
    avg_trainloss = running_loss / (epochs * len(batches))
    return avg_trainloss


def test_model(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        batches = list(testloader)
        for batch_index, batch in enumerate(batches):
            features = batch["feature_tensor"].to(device)
            labels = batch["label_tensor"].to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            # Visualize CWT coefficients for the first and last batch
            is_first_batch = batch_index == 0
            is_last_batch = batch_index == len(batches) - 1
            if is_first_batch or is_last_batch:
                signal_dict = {i: name[0] for i, name in batch["dict_idx_feature"].items()}

                Utilities().plot_cwt_coeffs_per_label(
                    X=batch["data"].numpy(),
                    y=batch["label_tensor"].numpy(),
                    signal=np.random.choice(list(signal_dict.keys())),  # Choose a random signal to visualize
                    scales=np.arange(1, NUM_SCALES + 1),
                    wavelet='morl',
                    filename_prefix=f"batch_{batch_index}",
                    signal_dict=signal_dict
                )

    accuracy = correct / len(batches)
    loss = loss / len(batches)
    return loss, accuracy


from torch.utils.data import DataLoader, IterableDataset
class StreamingDataset(IterableDataset):
    def __init__(self, csv_path: str, segment_length: int, overlap_fraction: float, num_scales: int, label_column: str):
        """
        Initialize the StreamingDataset with the buffer parameters.
        Does not load the CSV into memory, but reads it in chunks during iteration.
        """
        self.csv_path = csv_path
        self.segment_length = segment_length
        self.overlap_fraction = overlap_fraction
        self.num_scales = num_scales
        self.label_column = label_column
        self.synthetic = False

        if not os.path.exists(csv_path):
            self.synthetic = True
            self.synthetic_size = BATCH_SIZE
            self.num_synthetic_features = 4
            self.num_synthetic_classes = 5
            self._generate_synthetic_data()
            return

        # calculate how many rows to shift forward  for each segment based on the overlap fraction
        # e.g. if segment_length=100 and overlap_fraction=0.5, then shift_length=50
        self.step_size = int(self.segment_length * (1 - self.overlap_fraction))

        # Guard against zero or negative step sizes (prevent infinite loops)
        if self.step_size <= 0:
            self.step_size = 1

    def _generate_synthetic_data(self):
        self.synthetic_segments = np.random.rand(self.synthetic_size, self.segment_length, self.num_synthetic_features)
        self.synthetic_arrays = np.random.rand(self.synthetic_size, self.num_synthetic_features, self.num_scales, self.num_scales).astype(np.float32)
        self.synthetic_labels = np.random.randint(0, self.num_synthetic_classes, size=self.synthetic_size, dtype=np.int64)
        self.synthetic_dict_idx_feature = {i: f"Sensor_{i+1}" for i in range(self.num_synthetic_features)}

    def create_feature_and_label(self, segment_chunk):
        """
        Processes a pandas Dataframe chunk of length segment_length.
        Handles missing values and returns a numpy array of shape (segment_length, num_features) and the corresponding label.
        """
        # identify the columns that are not label columns
        feature_columns = [c for c in segment_chunk.columns if c != self.label_column]
        self.num_features = len(feature_columns)
        # create dict of column indices to feature names for consistent ordering
        self.dict_idx_feature = {i: col for i, col in enumerate(feature_columns)}

        # forward fill, backward fill, and then fill remaining NaNs with 0 in the feature columns
        segment_chunk[feature_columns] = segment_chunk[feature_columns].ffill().bfill().fillna(0)

        # create a single label by taking the mode of the label column in the segment
        label = int(segment_chunk[self.label_column].mode().iloc[0])

        # convert the segment to a numpy array
        segment_array = segment_chunk.to_numpy()

        segment_cwt_matrix = self.create_cwt_matrix(segment_array, wavelet_name='morl', rescale_size=self.num_scales, log_scale=True)

        return segment_cwt_matrix, label

    def create_cwt_matrix(self, segment_array, wavelet_name='morl', rescale_size=128, log_scale=False):
        import pywt
        from skimage.transform import resize

        # range of scales for CWT
        scales = np.arange(1, self.num_scales + 1)

        if log_scale:
            scales = np.logspace(np.log(1), np.log(self.num_scales+1), num=self.num_scales, base=np.e, dtype=np.float32)

        # preallocate matrix for CWT images
        cwt_matrix = np.ndarray(shape=(self.num_features, rescale_size, rescale_size), dtype=np.float32)

        for signal_idx,signal_name in self.dict_idx_feature.items():
            signal_data = segment_array[:, signal_idx]
            coeffs, freqs = pywt.cwt(signal_data, scales, wavelet_name)
            coeffs_resized = resize(coeffs, (rescale_size, rescale_size), mode='constant', anti_aliasing=True)
            cwt_matrix[signal_idx,:, :] = coeffs_resized

        return cwt_matrix


    def __iter__(self):
        """
        The core streaming logic: 
        - reads the CSV in chunks, 
        - processes each chunk into overlapping segments, and 
        - yields processed tensors as feature-label pairs one at a time.
        """
        if self.synthetic:
            for i in range(self.synthetic_size):
                yield {
                    "data": self.synthetic_segments[i], 
                    "dict_idx_feature": self.synthetic_dict_idx_feature, 
                    "feature_tensor": torch.tensor(self.synthetic_arrays[i], dtype=torch.float32),
                    "label_tensor": torch.tensor(self.synthetic_labels[i]).long()
                }
            return
        
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
                feature_tensor = torch.tensor(feature, dtype=torch.float32)
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
    def plot_cwt_coeffs_per_label(X, y, signal, scales, wavelet, sample=0, filename_prefix="batch_0", signal_dict=None):
        """
        Plots the CWT coefficients for a specific signal across different labels in the dataset.
        Parameters:
        - X: 3D numpy array of shape (num_samples, num_scales, num_features)
        - y: 1D numpy array of labels corresponding to each sample in X
        - signal: index of the signal (feature) to visualize
        - scales: array of scales used for the CWT
        - wavelet: name of the wavelet used for the CWT
        - sample: index of the sample to visualize (default is 0)
        - filename_prefix: prefix for the saved plot filename
        - signal_dict: optional dictionary mapping signal indices to human-readable names
        """
        import matplotlib.pyplot as plt
        import pywt

        signal_name = signal_dict.get(signal, f"Signal_id{signal}") if signal_dict else f"Signal_id{signal}"

        unique_labels = np.unique(y)
        label_indices = {str(label): np.where(y == label)[0] for label in unique_labels}
        print(f"Found {len(unique_labels)} unique labels: {list(label_indices.keys())}")

        n_labels = len(label_indices)
        fig, axs = plt.subplots(
            nrows=2,
            ncols=n_labels,
            sharex=True,
            sharey="row",
            figsize=(4 * n_labels, 8)
        )
        vmin_val = None
        vmax_val = None

        # display the scalogram for each label in the first row of subplots
        for ax, indices, name in zip(axs[0], label_indices.values(), label_indices.keys()):
            coeffs, freqs = pywt.cwt(X[indices[sample], :, signal], scales, wavelet)
            vmin_val = coeffs.min() if vmin_val is None else vmin_val
            vmax_val = coeffs.max() if vmax_val is None else vmax_val

            im = ax.imshow(coeffs, cmap='coolwarm', aspect='auto', vmin=vmin_val, vmax=vmax_val)
            ax.set_title(name, fontsize=8, pad=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, location='top')
            cbar_ticks = [coeffs.min(), coeffs.mean(), coeffs.max()]
            cbar.set_ticks(cbar_ticks)
            cbar.ax.set_xticklabels([f"{tick:.0f}" for tick in cbar_ticks], fontsize=6)
            cbar.ax.set_xlabel(f'{signal_name} Intensity', rotation=0, va='top', labelpad=15, fontsize=10)
        axs[0, 0].set_ylabel('Scale')

        # display the original timeseries signal below the scalograms
        for ax, indices, name in zip(axs[1], label_indices.values(), label_indices.keys()):
            ax.plot(X[indices[sample], :, signal])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time', fontsize=10)
        axs[1, 0].set_ylabel('Signal Value')

        fig.tight_layout()

        path_filename = os.path.join(os.getenv("DATA_DIR"), f"{filename_prefix}_cwt_coeffs_signal_{signal_name}_sample_{sample}.png")
        plt.savefig(path_filename, dpi=300)
        print(f"Saved file to {path_filename}")
        plt.close(fig)
        


if __name__ == "__main__":
    # Example usage of StreamingDataset
    CSV_FILE_PATH = os.path.join(os.getenv("DATA_DIR"), "train.csv")  # Path to your CSV file
    SEGMENT_LENGTH = 5
    OVERLAP_FRACTION = 0.5
    NUM_SCALES = 32
    LABEL_COLUMN = "Sensor_5"  # Adjust based on your CSV structure
    BATCH_SIZE = 30

    dataset = StreamingDataset(
        csv_path=CSV_FILE_PATH,
        segment_length=SEGMENT_LENGTH,
        overlap_fraction=OVERLAP_FRACTION,
        num_scales=NUM_SCALES,
        label_column=LABEL_COLUMN
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    utilities = Utilities()
    batches = list(dataloader)

    for batch_index, batch in enumerate(batches):
        is_first_batch = batch_index == 0
        is_last_batch = batch_index == len(batches) - 1

        if is_first_batch or is_last_batch:
            print(f"Batch {batch_index}: data shape {batch['data'].shape}, feature tensor shape {batch['feature_tensor'].shape}, label tensor shape {batch['label_tensor'].shape}")
            print(f"Feature names: {batch['dict_idx_feature']}")
            utilities = Utilities()
            signal_dict = {i: name[0] for i, name in batch["dict_idx_feature"].items()}
            
            utilities.plot_cwt_coeffs_per_label(
                X=batch["data"].numpy(),
                y=batch["label_tensor"].numpy(),
                signal=np.random.choice(list(signal_dict.keys())),  # Choose a random signal to visualize
                scales=np.arange(1, NUM_SCALES + 1),
                wavelet='morl',
                filename_prefix=f"batch_{batch_index}",
                signal_dict=signal_dict
            )


    
