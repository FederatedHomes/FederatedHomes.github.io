"""pytorchexample: A Flower / PyTorch app."""

import json
import os

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CLIENT_ID = os.environ.get("CLIENT_ID", "0")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
MODEL_BASE_NAME = os.environ.get("MODEL_BASE_NAME", "final_model.pt")
CLIENT_MODEL_PREFIX = os.environ.get("CLIENT_MODEL_PREFIX", "client")
METRICS_BASE_NAME = os.environ.get("METRICS_BASE_NAME", "final_metrics.json")

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.task import CustomNet, load_client_data
from src.task import test_model
from src.task import train_model


def save_client_checkpoint(model: torch.nn.Module) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{CLIENT_MODEL_PREFIX}_{CLIENT_ID}_{MODEL_BASE_NAME}",
    )
    torch.save(model.state_dict(), checkpoint_path)


def save_client_metrics(metrics: dict, key: str) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    metrics_path = os.path.join(
        CHECKPOINT_DIR,
        f"{CLIENT_MODEL_PREFIX}_{CLIENT_ID}_{METRICS_BASE_NAME}",
    )
    existing = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (ValueError, json.JSONDecodeError):
            existing = {}

    existing[key] = metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the client-local data
    trainloader, valloader = load_client_data(DATA_DIR)

    batch = next(iter(trainloader))
    in_channels = batch["feature_tensor"].shape[1]
    num_classes = 5

    # Load the model and initialize it with the received weights
    model = CustomNet(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Call the training function
    train_loss = train_model(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    save_client_checkpoint(model)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(list(trainloader)),
    }
    save_client_metrics(metrics, key="train_metrics")

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the client-local data
    _, valloader = load_client_data(DATA_DIR)

    batch = next(iter(valloader))
    in_channels = batch["feature_tensor"].shape[1]
    num_classes = 5

    # Load the model and initialize it with the received weights
    model = CustomNet(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Call the evaluation function
    eval_loss, eval_acc = test_model(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    save_client_metrics(metrics, key="eval_metrics")
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
