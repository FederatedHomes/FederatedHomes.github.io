from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_contract import CONTRACT


@pytest.fixture
def valid_segment():
    """Return one valid contract-sized segment."""

    data = {
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=CONTRACT.segment_length,
            freq="s",
        ),
        "Sensor_1": np.arange(
            CONTRACT.segment_length,
            dtype=np.float32,
        ),
        "Sensor_2": np.arange(
            CONTRACT.segment_length,
            dtype=np.float32,
        ),
        "Sensor_3": np.arange(
            CONTRACT.segment_length,
            dtype=np.float32,
        ),
        "Sensor_4": np.arange(
            CONTRACT.segment_length,
            dtype=np.float32,
        ),
        "Sensor_5": np.zeros(
            CONTRACT.segment_length,
            dtype=np.int64,
        ),
    }

    return pd.DataFrame(data).set_index("timestamp")


@pytest.fixture
def client_data_factory(tmp_path):
    """
    Create isolated client datasets for multi-client validation tests.
    """
    def factory(client_id, train_df, val_df=None):
        client_dir = tmp_path / client_id
        client_dir.mkdir(parents=True, exist_ok=True)

        train_path = client_dir / "train.csv"
        val_path = client_dir / "val.csv"

        train_df.to_csv(train_path)

        if val_df is None:
            val_df = train_df.copy()

        val_df.to_csv(val_path)

        return client_dir

    return factory