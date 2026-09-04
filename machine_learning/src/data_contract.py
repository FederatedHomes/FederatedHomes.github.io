# data_contract.py — ships with the FAB, imported by every ClientApp
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class DataContract:
    """Authoritative description of the data presented to the PyTorch model.

    The existing ``validate`` method intentionally remains unchanged during this
    contract-expansion step. New fields and derived properties describe the
    requirements that will be consumed by the dataset/model refactor in a later
    step.
    """

    # Contract identity
    version: str

    # Raw input schema
    label_column: str
    label_info: Dict[str, Dict[str, str]]
    feature_info: Dict[str, Dict[str, str]]
    feature_dtype: str
    label_dtype: str

    # Segmentation
    segment_length: int
    overlap_fraction: float

    # Missing-value preprocessing
    missing_value_strategy: str

    # CWT transformation
    transform_type: str
    wavelet: str
    num_scales: int
    scale_mode: str
    scale_min: float
    scale_max: float
    output_size: Tuple[int, int]

    # Model-facing tensor specification
    tensor_layout: str
    tensor_dtype: str

    def validate(self, features: np.ndarray, labels: np.ndarray, feature_to_index: Dict, label_to_index: Dict) -> tuple:
        # time series batches must have dimensions: (batch_size x segment_length x num_signals)
        valid_num_signals = len(self.feature_info.keys())
        if features.shape[2] != valid_num_signals:
            raise ValueError(
                f"Expected {valid_num_signals} signals, got {features.shape[2]}"
            )
        if features.shape[1] != self.segment_length:
            raise ValueError(
                f"Expected segments with {self.segment_length} datapoints, got {features.shape[1]}"
            )
        
        # the signal name must correspond to the correct signal index
        for feature_name,feature_index in feature_to_index.items():
            valid_feature_name = self.feature_info[str(feature_index)]["name"]
            if feature_name != valid_feature_name:
                raise ValueError(
                    f"Expected feature {valid_feature_name} at index {feature_index}, got {feature_name}"
                )

        # labels must be one of the allowed values
        valid_labels = set([int(key) for key in self.label_info.keys()])
        valid_num_labels = len(valid_labels)
        bad_labels = set(labels.tolist()) - valid_labels
        if bad_labels:
            raise ValueError(f"Unknown label indices: {bad_labels}. Valid labels: {valid_labels}")

        # label name must be correspond to the correct label index
        for label_name,label_index in label_to_index.items():
            valid_label_name = self.label_info[str(label_index)]["name"]
            if label_name != valid_label_name:
                raise ValueError(
                    f"Expected label {valid_label_name} at index {label_index}, got {label_name}"
                )
            
        # return valid number of signals and number of labels
        return valid_num_signals, valid_num_labels

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Return feature names in their contract-defined index order."""
        return tuple(
            self.feature_info[str(index)]["name"]
            for index in sorted(self.feature_info.keys(), key=int)
        )

    @property
    def label_names(self) -> Tuple[str, ...]:
        """Return label names in their contract-defined index order."""
        return tuple(
            self.label_info[str(index)]["name"]
            for index in sorted(self.label_info.keys(), key=int)
        )

    @property
    def num_features(self) -> int:
        """Number of model input feature channels."""
        return len(self.feature_info)

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return len(self.label_info)

    @property
    def input_channels(self) -> int:
        """Number of channels expected by the model input tensor."""
        return self.num_features

    @property
    def tensor_shape(self) -> Tuple[int, int, int]:
        """Model input shape excluding the batch dimension (C, H, W)."""
        height, width = self.output_size
        return self.input_channels, height, width

    @property
    def model_input_shape(self) -> Tuple[int, int, int, int]:
        """Model input shape including a symbolic/variable batch dimension.

        ``-1`` represents the runtime batch size and is not a tensor dimension
        that should be materialized directly.
        """
        channels, height, width = self.tensor_shape
        return -1, channels, height, width

    @property
    def cwt_scales(self) -> np.ndarray:
        """Return the deterministic CWT scale values required by the contract."""
        if self.scale_mode == "linear":
            return np.linspace(
                self.scale_min,
                self.scale_max,
                num=self.num_scales,
                dtype=np.float32,
            )
        if self.scale_mode == "logarithmic":
            return np.logspace(
                np.log(self.scale_min),
                np.log(self.scale_max),
                num=self.num_scales,
                base=np.e,
                dtype=np.float32,
            )
        raise ValueError(f"Unsupported scale mode: {self.scale_mode}")

    @property
    def segmentation_step_size(self) -> int:
        """Return the number of rows by which the segmentation window advances."""
        return max(1, int(self.segment_length * (1 - self.overlap_fraction)))


CONTRACT = DataContract(
    version="1.1",

    # Raw input schema
    label_column="Sensor_5",
    label_info={
        "0": {
            "name": "NORMAL",
            "description": "All equipment signals are in normal operation."
        },
        "1": {
            "name": "ALERT",
            "description": "At least two signals are outside normal operation."
        }
    },
    feature_info={
        "0": {
            "name": "Sensor_1",
            "description": "Description of feature 1."
        },
        "1": {
            "name": "Sensor_2",
            "description": "Description of feature 2."
        },
        "2": {
            "name": "Sensor_3",
            "description": "Description of feature 3."
        },
        "3": {
            "name": "Sensor_4",
            "description": "Description of feature 4."
        }
    },
    feature_dtype="float32",
    label_dtype="int64",

    # Segmentation
    segment_length=5,
    overlap_fraction=0.5,

    # Missing-value preprocessing
    missing_value_strategy="forward_fill_backward_fill_zero",

    # CWT transformation
    transform_type="cwt",
    wavelet="morl",
    num_scales=32,
    scale_mode="logarithmic",
    scale_min=1.0,
    scale_max=33.0,
    output_size=(32, 32),

    # Model-facing tensor specification
    tensor_layout="NCHW",
    tensor_dtype="float32",
)
