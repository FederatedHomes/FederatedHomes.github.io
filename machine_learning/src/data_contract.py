# data_contract.py  — ships with the FAB, imported by every ClientApp
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass(frozen=True)
class DataContract:
    label_info: Dict[str,Dict[str,str]]
    feature_info: Dict[str,Dict[str,str]]
    segment_length: int
    overlap_fraction:float
    num_scales:int
    version: str = "1.0"


    def validate(self, features: np.ndarray, labels: np.ndarray, feature_to_index:Dict, label_to_index:Dict) -> tuple:
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
        

CONTRACT = DataContract(
    segment_length=5,
    overlap_fraction=0.5,
    num_scales=32,
    label_info = {
        "0": {
            "name": "NORMAL",
            "description": "All equipment signals are in normal operation."
        },
        "1": {
            "name": "ALERT",
            "description": "At least two signals are outside normal operation."
        }
    },
    feature_info = {
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
)   