from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_contract import CONTRACT
from src.task import StreamingDataset


def make_dataset(tmp_path, dataframe, contract=CONTRACT):
    """Create a temporary CSV-backed StreamingDataset."""

    tmp_path.mkdir(parents=True, exist_ok=True)

    csv_path = tmp_path / "client.csv"

    dataframe.to_csv(csv_path)

    return StreamingDataset(
        csv_path=str(csv_path),
        contract=contract,
    )


def test_valid_client_data_is_accepted(
    tmp_path,
    valid_segment,
):
    dataset = make_dataset(
        tmp_path,
        valid_segment,
    )

    feature, label = dataset.create_feature_and_label(
        valid_segment
    )

    assert feature.shape == CONTRACT.tensor_shape
    assert feature.dtype == np.dtype(
        CONTRACT.tensor_dtype
    )
    assert label in {
        int(key)
        for key in CONTRACT.label_info.keys()
    }


def test_multiple_valid_clients_are_independently_accepted(
    tmp_path,
    valid_segment,
):
    client_a = valid_segment.copy()
    client_b = valid_segment.copy()

    client_b["Sensor_1"] += 100
    client_b["Sensor_2"] += 200

    dataset_a = make_dataset(
        tmp_path / "client_a",
        client_a,
    )

    dataset_b = make_dataset(
        tmp_path / "client_b",
        client_b,
    )

    feature_a, label_a = (
        dataset_a.create_feature_and_label(client_a)
    )

    feature_b, label_b = (
        dataset_b.create_feature_and_label(client_b)
    )

    assert feature_a.shape == CONTRACT.tensor_shape
    assert feature_b.shape == CONTRACT.tensor_shape

    assert label_a in {0, 1}
    assert label_b in {0, 1}


def test_missing_feature_is_rejected(
    tmp_path,
    valid_segment,
):
    dataframe = valid_segment.drop(
        columns=["Sensor_3"]
    )

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with pytest.raises(
        ValueError,
        match="missing required contract columns",
    ):
        dataset._validate_raw_schema(
            dataframe
        )


def test_extra_feature_is_logged_and_ignored(
    tmp_path,
    valid_segment,
    caplog,
):
    dataframe = valid_segment.copy()

    dataframe["Extra_Sensor"] = np.arange(
        CONTRACT.segment_length
    )

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with caplog.at_level("WARNING"):
        dataset._validate_raw_schema(
            dataframe
        )

    assert "Extra_Sensor" in caplog.text
    assert "will be ignored" in caplog.text


def test_reordered_features_are_accepted(
    tmp_path,
    valid_segment,
):
    columns = [
        "Sensor_3",
        "Sensor_1",
        "Sensor_4",
        "Sensor_2",
        "Sensor_5",
    ]

    dataframe = valid_segment.loc[
        :,
        columns,
    ]

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    feature, label = (
        dataset.create_feature_and_label(
            dataframe
        )
    )

    assert feature.shape == CONTRACT.tensor_shape
    assert label in {0, 1}


def test_convertible_feature_dtype_is_aligned(
    tmp_path,
    valid_segment,
    caplog,
):
    dataframe = valid_segment.copy()

    for feature_name in CONTRACT.feature_names:
        dataframe[feature_name] = (
            dataframe[feature_name]
            .astype(np.float64)
        )

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with caplog.at_level("WARNING"):
        feature, _ = (
            dataset.create_feature_and_label(
                dataframe
            )
        )

    assert feature.dtype == np.dtype(
        CONTRACT.tensor_dtype
    )

    assert (
        "DataContract dtype" in caplog.text
    )


def test_numeric_string_features_are_converted(
    tmp_path,
    valid_segment,
    caplog,
):
    dataframe = valid_segment.copy()

    for feature_name in CONTRACT.feature_names:
        dataframe[feature_name] = (
            dataframe[feature_name]
            .map(str)
        )

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with caplog.at_level("WARNING"):
        feature, _ = (
            dataset.create_feature_and_label(
                dataframe
            )
        )

    assert feature.shape == CONTRACT.tensor_shape
    assert (
        "Converting feature" in caplog.text
    )


def test_non_convertible_feature_is_rejected(valid_segment, tmp_path):
    segment = valid_segment.copy()

    segment["Sensor_1"] = segment["Sensor_1"].astype(object)
    segment.loc[segment.index[0], "Sensor_1"] = "not-a-number"

    csv_path = tmp_path / "invalid_feature.csv"
    segment.to_csv(csv_path)

    dataset = StreamingDataset(
        csv_path=str(csv_path),
        contract=CONTRACT,
    )

    with pytest.raises(ValueError, match="convert"):
        next(iter(dataset))


def test_convertible_label_dtype_is_aligned(
    tmp_path,
    valid_segment,
    caplog,
):
    dataframe = valid_segment.copy()

    dataframe["Sensor_5"] = (
        dataframe["Sensor_5"]
        .astype(np.float64)
    )

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with caplog.at_level("WARNING"):
        _, label = (
            dataset.create_feature_and_label(
                dataframe
            )
        )

    assert isinstance(label, int)
    assert (
        "Converting label column" in caplog.text
    )


def test_numeric_string_labels_are_converted(
    tmp_path,
    valid_segment,
    caplog,
):
    dataframe = valid_segment.copy()

    dataframe["Sensor_5"] = (
        dataframe["Sensor_5"]
        .map(str)
    )

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with caplog.at_level("WARNING"):
        _, label = (
            dataset.create_feature_and_label(
                dataframe
            )
        )

    assert label in {0, 1}
    assert (
        "Converting label column" in caplog.text
    )


def test_non_convertible_label_is_rejected(valid_segment, tmp_path):
    segment = valid_segment.copy()

    segment["Sensor_5"] = segment["Sensor_5"].astype(object)
    segment.loc[segment.index[0], "Sensor_5"] = "INVALID"

    csv_path = tmp_path / "invalid_label.csv"
    segment.to_csv(csv_path)

    dataset = StreamingDataset(
        csv_path=str(csv_path),
        contract=CONTRACT,
    )

    with pytest.raises(ValueError, match="convert"):
        next(iter(dataset))


def test_fractional_label_is_rejected(valid_segment, tmp_path):
    segment = valid_segment.copy()

    segment["Sensor_5"] = segment["Sensor_5"].astype(object)
    segment.loc[segment.index[0], "Sensor_5"] = "0.5"

    csv_path = tmp_path / "fractional_label.csv"
    segment.to_csv(csv_path)

    dataset = StreamingDataset(
        csv_path=str(csv_path),
        contract=CONTRACT,
    )

    with pytest.raises(ValueError, match="integer"):
        next(iter(dataset))


def test_unknown_label_is_rejected(
    tmp_path,
    valid_segment,
):
    dataframe = valid_segment.copy()

    dataframe["Sensor_5"] = 99

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with pytest.raises(
        ValueError,
        match="Unknown label values",
    ):
        dataset.create_feature_and_label(
            dataframe
        )


def test_wrong_segmentation_length_is_rejected(
    tmp_path,
    valid_segment,
):
    dataframe = valid_segment.iloc[:-1].copy()

    dataset = make_dataset(
        tmp_path,
        dataframe,
    )

    with pytest.raises(
        ValueError,
        match="Invalid segment length",
    ):
        dataset.create_feature_and_label(
            dataframe
        )


def test_invalid_overlap_configuration_is_rejected(
    tmp_path,
    valid_segment,
):
    invalid_contract = replace(
        CONTRACT,
        overlap_fraction=1.5,
    )

    with pytest.raises(
        ValueError,
        match="overlap_fraction",
    ):
        make_dataset(
            tmp_path,
            valid_segment,
            contract=invalid_contract,
        )


def test_invalid_segment_length_configuration_is_rejected(
    tmp_path,
    valid_segment,
):
    invalid_contract = replace(
        CONTRACT,
        segment_length=0,
    )

    with pytest.raises(
        ValueError,
        match="segment_length",
    ):
        make_dataset(
            tmp_path,
            valid_segment,
            contract=invalid_contract,
        )


def test_invalid_tensor_dimensions_are_rejected(
    tmp_path,
    valid_segment,
):
    dataset = make_dataset(
        tmp_path,
        valid_segment,
    )

    invalid_tensor = np.zeros(
        (
            CONTRACT.num_features + 1,
            CONTRACT.output_size[0],
            CONTRACT.output_size[1],
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Invalid model input tensor dimensions",
    ):
        dataset._validate_tensor(
            invalid_tensor
        )


def test_invalid_tensor_dtype_is_rejected(
    tmp_path,
    valid_segment,
):
    dataset = make_dataset(
        tmp_path,
        valid_segment,
    )

    invalid_tensor = np.zeros(
        CONTRACT.tensor_shape,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Invalid model input tensor dtype",
    ):
        dataset._validate_tensor(
            invalid_tensor
        )

def test_mixed_multi_client_validation(
    tmp_path,
    valid_segment,
):
    clients = {
        "client-a": valid_segment.copy(),

        "client-b": valid_segment.copy(),

        "client-c-missing-feature": (
            valid_segment
            .drop(columns=["Sensor_2"])
        ),

        "client-d-extra-feature": (
            valid_segment.assign(
                Extra_Sensor=np.arange(
                    CONTRACT.segment_length
                )
            )
        ),

        "client-e-reordered": (
            valid_segment.loc[
                :,
                [
                    "Sensor_4",
                    "Sensor_2",
                    "Sensor_1",
                    "Sensor_3",
                    "Sensor_5",
                ],
            ]
        ),
    }

    expected = {
        "client-a": "valid",
        "client-b": "valid",
        "client-c-missing-feature": "invalid",
        "client-d-extra-feature": "valid",
        "client-e-reordered": "valid",
    }

    results = {}

    for client_id, dataframe in clients.items():
        client_dir = tmp_path / client_id
        client_dir.mkdir(parents=True)

        csv_path = client_dir / "train.csv"
        dataframe.to_csv(csv_path)

        dataset = StreamingDataset(
            csv_path=str(csv_path),
            contract=CONTRACT,
        )

        try:
            list(dataset)
            results[client_id] = "valid"
        except ValueError:
            results[client_id] = "invalid"

    assert results == expected