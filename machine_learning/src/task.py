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

from src.data_contract import CONTRACT

# define a mapping from string representation of dtypes to PyTorch dtypes
TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}

TRAIN_DATA_FILE = os.environ.get("TRAIN_DATA_FILE", "train.csv")
VAL_DATA_FILE = os.environ.get("VAL_DATA_FILE", "val.csv")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 30))

class CustomNet(nn.Module):
    """CNN tuned for the StreamingDataset CWT feature tensors."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(CONTRACT.num_features, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, CONTRACT.num_classes)

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
        contract=CONTRACT,
    )

    val_dataset = StreamingDataset(
        csv_path=str(val_csv),
        contract=CONTRACT,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader


def load_server_data(data_dir: str):
    """Load test set and return dataloader."""
    val_csv = Path(data_dir) / VAL_DATA_FILE
    val_dataset = StreamingDataset(
        csv_path=str(val_csv),
        contract=CONTRACT,
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
    total_examples = 0

    for epoch in range(epochs):
        it = iter(trainloader)
        batch_index = 1

        try:
            current_batch = next(it)
        except StopIteration:
            continue

        while True:
            is_first_batch = (batch_index==1)

            try:
                next_batch = next(it)
                is_last_batch = False
            except StopIteration:
                is_last_batch = True

            features = current_batch["feature_tensor"].to(device)
            labels = current_batch["label_tensor"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total_examples += features.shape[0]

            # Visualize CWT coefficients for the first and last batch of the first epoch
            if is_first_batch or is_last_batch and epoch == 0:
                signal_dict = {i: name[0] for i, name in current_batch["dict_idx_feature"].items()}
                Utilities().plot_cwt_coeffs_per_label(
                    X=current_batch["data"].numpy(),
                    y=current_batch["label_tensor"].numpy(),
                    signal=np.random.choice(list(signal_dict.keys())),  # Choose a random signal to visualize
                    scales=CONTRACT.cwt_scales,
                    wavelet=CONTRACT.wavelet,
                    filename_prefix=f"batch_{batch_index}",
                    signal_dict=signal_dict
                )

            if is_last_batch:
                break

            current_batch = next_batch
            batch_index += 1
            
    avg_trainloss = running_loss / (epochs * batch_index)
    return avg_trainloss, total_examples


def test_model(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total_examples = 0

    with torch.no_grad():
        it = iter(testloader)
        batch_index = 1

        try:
            current_batch = next(it)
        except StopIteration:
            print("ERROR: Could not iterate over the test batch.")

        while True:
            is_first_batch = (batch_index==1)

            try:
                next_batch = next(it)
                is_last_batch = False
            except StopIteration:
                is_last_batch = True

            features = current_batch["feature_tensor"].to(device)
            labels = current_batch["label_tensor"].to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            total_examples += features.shape[0]

            # Visualize CWT coefficients for the first and last batch
            if is_first_batch or is_last_batch:
                signal_dict = {i: name[0] for i, name in current_batch["dict_idx_feature"].items()}
                Utilities().plot_cwt_coeffs_per_label(
                    X=current_batch["data"].numpy(),
                    y=current_batch["label_tensor"].numpy(),
                    signal=np.random.choice(list(signal_dict.keys())),  # Choose a random signal to visualize
                    scales=CONTRACT.cwt_scales,
                    wavelet=CONTRACT.wavelet,
                    filename_prefix=f"batch_{batch_index}",
                    signal_dict=signal_dict
                )

            if is_last_batch:
                break

            current_batch = next_batch
            batch_index += 1

    accuracy = correct / batch_index
    loss = loss / batch_index
    return loss, accuracy, total_examples


from torch.utils.data import DataLoader, IterableDataset


class StreamingDataset(IterableDataset):
    def __init__(self, csv_path: str, contract=CONTRACT):
        """
        Initialize the StreamingDataset from the shared DataContract.

        The DataContract is the source of truth for:
        - raw feature/label schema
        - segmentation
        - missing-value preprocessing
        - CWT configuration
        - model-facing tensor shape and dtype
        """
        self.csv_path = csv_path
        self.contract = contract
        self.synthetic = False

        if not os.path.exists(csv_path):
            self.synthetic = True
            self.synthetic_size = BATCH_SIZE
            self.num_synthetic_features = contract.num_features
            self.num_synthetic_classes = contract.num_classes
            self._generate_synthetic_data()
            return

        self.step_size = contract.segmentation_step_size

    def _generate_synthetic_data(self):
        self.synthetic_segments = np.random.rand(
            self.synthetic_size,
            self.contract.segment_length,
            self.num_synthetic_features,
        ).astype(self.contract.feature_dtype)

        output_height, output_width = self.contract.output_size
        self.synthetic_arrays = np.random.rand(
            self.synthetic_size,
            self.num_synthetic_features,
            output_height,
            output_width,
        ).astype(self.contract.tensor_dtype)

        self.synthetic_labels = np.random.randint(
            0,
            self.num_synthetic_classes,
            size=self.synthetic_size,
            dtype=np.int64,
        )

        self.synthetic_dict_idx_feature = {
            i: name for i, name in enumerate(self.contract.feature_names)
        }

    def _validate_raw_schema(self, segment_chunk):
        """Ensure the local CSV conforms to the contractual column schema."""
        required_columns = set(self.contract.feature_names) | {
            self.contract.label_column
        }
        missing_columns = required_columns - set(segment_chunk.columns)

        if missing_columns:
            raise ValueError(
                f"CSV is missing required contract columns: "
                f"{sorted(missing_columns)}"
            )

    def validate_segment(self, segment_chunk):
        """Validate a contract-sized segment before feature transformation."""

        if len(segment_chunk) != self.contract.segment_length:
            raise ValueError(
                f"Invalid segment length: expected "
                f"{self.contract.segment_length}, got {len(segment_chunk)}"
            )

        label_series = pd.to_numeric(
            segment_chunk[self.contract.label_column],
            errors="coerce",
        )

        if label_series.isna().any():
            raise ValueError(
                f"Non-numeric or missing values found in label column "
                f"'{self.contract.label_column}'"
            )

        if not np.equal(
            label_series.to_numpy(),
            label_series.astype(np.int64).to_numpy(),
        ).all():
            raise ValueError(
                f"Non-integer values found in label column "
                f"'{self.contract.label_column}'"
            )

        valid_labels = {
            int(key) for key in self.contract.label_info.keys()
        }

        observed_labels = set(
            label_series.astype(np.int64).tolist()
        )

        bad_labels = observed_labels - valid_labels

        if bad_labels:
            raise ValueError(
                f"Unknown label values: {sorted(bad_labels)}. "
                f"Valid labels: {sorted(valid_labels)}"
            )

    def create_feature_and_label(self, segment_chunk):
        """
        Process one contract-sized DataFrame segment.

        Returns:
            CWT feature matrix with shape (channels, height, width)
            and the corresponding integer class label.
        """

        self.validate_segment(segment_chunk)

        # Explicitly select and order only the contract-defined features.
        feature_columns = list(self.contract.feature_names)
        segment_features = segment_chunk.loc[:, feature_columns].copy()

        # Contract-defined missing-value strategy:
        # forward fill, backward fill, then replace any remaining NaNs with zero.
        if self.contract.missing_value_strategy == "forward_fill_backward_fill_zero":
            segment_features = (
                segment_features.ffill().bfill().fillna(0)
            )
        else:
            raise ValueError(
                f"Unsupported missing-value strategy: "
                f"{self.contract.missing_value_strategy}"
            )

        # Convert feature values to the contract-defined source dtype.
        segment_features = segment_features.astype(self.contract.feature_dtype)

        # Create a single segment label using the mode of the contractual label column.
        label = int(segment_chunk[self.contract.label_column].mode().iloc[0])

        self.num_features = self.contract.num_features
        self.dict_idx_feature = {
            i: name for i, name in enumerate(feature_columns)
        }

        segment_array = segment_features.to_numpy(dtype=self.contract.feature_dtype)

        segment_cwt_matrix = self.create_cwt_matrix(segment_array)

        return segment_cwt_matrix, label

    def create_cwt_matrix(self, segment_array):
        """Create the contract-defined CWT tensor for every feature channel."""
        import pywt
        from skimage.transform import resize

        if self.contract.transform_type != "cwt":
            raise ValueError(
                f"Unsupported transform type: {self.contract.transform_type}"
            )

        scales = self.contract.cwt_scales
        output_height, output_width = self.contract.output_size

        cwt_matrix = np.empty(
            shape=(
                self.contract.num_features,
                output_height,
                output_width,
            ),
            dtype=self.contract.tensor_dtype,
        )

        for signal_idx, signal_name in self.dict_idx_feature.items():
            signal_data = segment_array[:, signal_idx]
            coeffs, _ = pywt.cwt(
                signal_data,
                scales,
                self.contract.wavelet,
            )

            coeffs_resized = resize(
                coeffs,
                (output_height, output_width),
                mode="constant",
                anti_aliasing=True,
            )

            cwt_matrix[signal_idx, :, :] = coeffs_resized.astype(
                self.contract.tensor_dtype
            )

        return cwt_matrix

    def __iter__(self):
        """
        Stream local CSV data as overlapping, contract-compliant segments.
        """
        if self.synthetic:
            for i in range(self.synthetic_size):
                yield {
                    "data": self.synthetic_segments[i],
                    "dict_idx_feature": self.synthetic_dict_idx_feature,
                    "feature_tensor": torch.tensor(
                        self.synthetic_arrays[i],
                        dtype=TORCH_DTYPES[self.contract.tensor_dtype],
                    ),
                    "label_tensor": torch.tensor(
                        self.synthetic_labels[i],
                        dtype=TORCH_DTYPES[self.contract.label_dtype],
                    ),
                }
            return

        csv_stream = pd.read_csv(
            self.csv_path,
            chunksize=self.contract.segment_length,
            index_col="timestamp",
        )

        buffer = pd.DataFrame()

        for chunk in csv_stream:
            self._validate_raw_schema(chunk)

            # Keep only contract-defined feature and label columns.
            chunk = chunk.loc[
                :,
                list(self.contract.feature_names)
                + [self.contract.label_column],
            ]

            buffer = pd.concat([buffer, chunk], ignore_index=True)

            while len(buffer) >= self.contract.segment_length:
                buffer_segment = buffer.iloc[
                    : self.contract.segment_length
                ].copy()

                feature, label = self.create_feature_and_label(buffer_segment)

                feature_tensor = torch.tensor(
                    feature,
                    dtype=TORCH_DTYPES[self.contract.tensor_dtype],
                )
                label_tensor = torch.tensor(
                    label,
                    dtype=TORCH_DTYPES[self.contract.label_dtype],
                )

                yield {
                    "data": buffer_segment[
                        list(self.contract.feature_names)
                    ].to_numpy(dtype=self.contract.feature_dtype),
                    "dict_idx_feature": self.dict_idx_feature,
                    "feature_tensor": feature_tensor,
                    "label_tensor": label_tensor,
                }

                buffer = buffer.iloc[
                    self.step_size:
                ].reset_index(drop=True)


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
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, location='top')
            cbar_ticks = [coeffs.min(), coeffs.mean(), coeffs.max()]
            cbar.set_ticks(cbar_ticks)
            cbar.ax.set_xticklabels([f"{tick:.1f}" for tick in cbar_ticks], fontsize=6)
            cbar.ax.set_xlabel(f'{signal_name} Intensity', rotation=0, va='top', labelpad=15, fontsize=10)
        axs[0, 0].set_ylabel('Scale')

        # display the original timeseries signal below the scalograms
        for ax, indices, name in zip(axs[1], label_indices.values(), label_indices.keys()):
            ax.plot(X[indices[sample], :, signal])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time', fontsize=10)
            ax.set_title(name, fontsize=10, pad=5)
        axs[1, 0].set_ylabel('Signal Value')

        fig.tight_layout()

        path_filename = os.path.join(os.getenv("DATA_DIR"), f"{filename_prefix}_cwt_coeffs_signal_{signal_name}_sample_{sample}.png")
        plt.savefig(path_filename, dpi=300)
        print(f"Saved file to {path_filename}")
        plt.close(fig)

    @staticmethod
    def create_confusion_matrix(y_true=None, y_pred=None, label_names:list=[]):
            import seaborn as sns
            from sklearn import metrics
            import matplotlib.pyplot as plt

            # determine the total accuracy
            accuracy = f"{metrics.accuracy_score(y_true,y_pred):.2f}"
            precision = f"{metrics.precision_score(y_true,y_pred):.2f}"
            recall = f"{metrics.recall_score(y_true,y_pred):.2f}"

            # calculate the confusion matrix
            conf_matrix = metrics.confusion_matrix(y_true=y_true,y_pred=y_pred)

            fig,ax = plt.subplots(figsize=(6,6))
            ax = sns.heatmap(
                conf_matrix,
                fmt='g',
                cmap=plt.cm.Blues,
                cbar=False,
                xticklabels=label_names,
                yticklabels=label_names
            )

            ax.set_title(f"Confusion Matrix | Accuracy : {accuracy} | Precision: {precision} | Recall: {recall}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            fig.tight_layout()

            path_filename = os.path.join(os.getenv("DATA_DIR"), "ConfusionMatrix.png")
            plt.savefig(path_filename, dpi=300)
            print(f"Saved file to {path_filename}")
            plt.close(fig)

    @staticmethod
    def pca_of_cwt_coeffs(X, n_scales, wavelet_name="morl"):
        from sklearn.decomposition import PCA
        # apply PCA for just a single component to get the most significant coefficient per scale
        pca =  PCA(n_components=1)
        # create a range of scales
        scales = np.arange(1,n_scales+1)

        X_pca = np.array([])
        for signal in range(X.shape[2]):
            pca_components = np.empty((0,n_scales), dtype='float32')
            for sample in range(X.shape[0]):
                coeffs, freqs = pywt.cwt(X[sample, :, signal], scales, wavelet_name)
                pca_components = np.vstack([pca_components, pca.fit_transform(coeffs).flatten()])

            if signal==0:
                X_pca = pca_components
            else:
                X_pca = np.concatenate((X_pca, pca_components), axis=1)
        return X_pca

    @staticmethod
    def build_and_fit_xgb_model(X_train, y_train, X_val, y_val, n_depth, subsample, n_estimators):
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            max_depth = n_depth,
            objective = 'multi:softmax',
            num_classes = 2,
            subsample = subsample,
            n_estimators = n_estimators,
            eval_metric = ["merror"]
        )

        eval_set = [(X_val,y_val)]
        history = xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        return xgb_model, history

def validate_model_compatibility(model: nn.Module, state_dict: dict) -> None:
    """Validate that received Flower model weights match the contract-defined model."""

    expected_state_dict = model.state_dict()

    expected_keys = set(expected_state_dict.keys())
    received_keys = set(state_dict.keys())

    missing_keys = expected_keys - received_keys
    unexpected_keys = received_keys - expected_keys

    if missing_keys or unexpected_keys:
        raise ValueError(
            "Incompatible federated model state_dict. "
            f"Missing keys: {sorted(missing_keys)}; "
            f"Unexpected keys: {sorted(unexpected_keys)}"
        )

    shape_mismatches = []

    for key in expected_state_dict:
        expected_shape = tuple(expected_state_dict[key].shape)
        received_shape = tuple(state_dict[key].shape)

        if expected_shape != received_shape:
            shape_mismatches.append(
                f"{key}: expected {expected_shape}, received {received_shape}"
            )

    if shape_mismatches:
        raise ValueError(
            "Incompatible federated model tensor shapes: "
            + "; ".join(shape_mismatches)
        )


if __name__ == "__main__":
    # Example usage of StreamingDataset
    CSV_FILE_PATH = os.path.join(os.getenv("DATA_DIR"), TRAIN_DATA_FILE)

    dataset = StreamingDataset(
        csv_path=CSV_FILE_PATH,
        contract=CONTRACT,
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    batches = list(dataloader)

    for batch_index, batch in enumerate(batches):
        is_first_batch = batch_index == 0
        is_last_batch = batch_index == len(batches) - 1

        if is_first_batch or is_last_batch:
            print(
                f"Batch {batch_index}: data shape {batch['data'].shape}, "
                f"feature tensor shape {batch['feature_tensor'].shape}, "
                f"label tensor shape {batch['label_tensor'].shape}"
            )

            signal_dict = {
                i: name[0]
                for i, name in batch["dict_idx_feature"].items()
            }
            print(f"Feature names and internal index: {signal_dict}")

            Utilities().plot_cwt_coeffs_per_label(
                X=batch["data"].numpy(),
                y=batch["label_tensor"].numpy(),
                signal=np.random.choice(list(signal_dict.keys())),
                scales=CONTRACT.cwt_scales,
                wavelet=CONTRACT.wavelet,
                filename_prefix=f"batch_{batch_index}",
                signal_dict=signal_dict,
            )

