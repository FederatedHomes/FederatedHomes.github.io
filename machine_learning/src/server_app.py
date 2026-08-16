"""pytorchexample: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

import json

from src.task import CustomNet, load_server_data, test_model

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
MODEL_BASE_NAME = os.environ.get("MODEL_BASE_NAME", "final_model.pt")
GLOBAL_MODEL_PREFIX = os.environ.get("GLOBAL_MODEL_PREFIX", "global")
METRICS_BASE_NAME = os.environ.get("METRICS_BASE_NAME", "final_metrics.json")

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Infer the model shape from the actual data loader so the server model matches the training data.
    test_dataloader = load_server_data(DATA_DIR)
    sample_batch = next(iter(test_dataloader))
    in_channels = sample_batch["feature_tensor"].shape[1]
    num_classes = 5

    global_model = CustomNet(in_channels=in_channels, num_classes=num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final global metrics
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        final_round = max(result.evaluate_metrics_serverapp.keys())
        final_metrics = dict(result.evaluate_metrics_serverapp.get(final_round, {}))
        metrics_path = os.path.join(
            CHECKPOINT_DIR,
            f"{GLOBAL_MODEL_PREFIX}_{METRICS_BASE_NAME}",
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        model_path = os.path.join(
            CHECKPOINT_DIR,
            f"{GLOBAL_MODEL_PREFIX}_{MODEL_BASE_NAME}",
        )
        torch.save(state_dict, model_path)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    test_dataloader = load_server_data(DATA_DIR)
    sample_batch = next(iter(test_dataloader))
    in_channels = sample_batch["feature_tensor"].shape[1]
    num_classes = 5

    model = CustomNet(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the global model on the test set
    test_loss, test_acc = test_model(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
