---
tags: [federated, machine learning, vision, fds]
framework: [torch, torchvision]
---

# Federated Learning with PyTorch and Flower

This repository provides a Docker-based Flower deployment using the Flower SuperLink / SuperNode / SuperExec topology.
Each client can mount its own local CSV dataset from `./data/<client>/` and save checkpoints to `./checkpoints/<client>/`.

## What this repo contains

- `Dockerfile.superexec` — runtime image for Flower `serverapp` and `clientapp`
- `docker-compose.yml` — SuperLink/SuperNode/SuperExec service definitions and trainer orchestration
- `setup.sh` — installer script to create required directories, copy `.env.example`, and launch Compose
- `pyproject.toml` — Python project metadata and runtime dependencies
- `src/` — Flower app source
  - `src/server_app.py` — Flower `ServerApp`
  - `src/client_app.py` — Flower `ClientApp`
  - `src/task.py` — model, training, and local CSV dataset loading

## Dependency refinement

`requirements.txt` is the source of runtime dependencies installed into the `flwr_superexec:local` image during Docker build.
The base image `flwr/superexec:1.33.0` already contains the Flower runtime, so `requirements.txt` deliberately omits `flwr` and installs only the additional application packages.

`pyproject.toml` remains the Python package metadata file used for local development, packaging, and the Flower app config, but it is no longer the build-time dependency source for the Docker image.

## Data structure

The project expects a per-client dataset layout under `machine_learning/data/`:

```text
data/client1/train.csv
data/client1/val.csv

data/client2/train.csv
data/client2/val.csv
```

Each CSV must include a `label` column. The loader supports either:

- `img_path` with relative file paths to image files, or
- numeric feature columns only

> `machine_learning/data/` and `machine_learning/checkpoints/` are not tracked in Git. These directories are created by `setup.sh` and are intended for local client datasets and checkpoint storage only.

Example CSV with image paths:

```csv
img_path,label
images/img_001.png,3
images/img_002.png,7
```

Example CSV with numeric features:

```csv
feat_0,feat_1,feat_2,label
0.12,0.55,0.31,1
0.23,0.18,0.44,0
```

## Setup and launch

From the `machine_learning/` folder:

```bash
chmod +x setup.sh
./setup.sh
```

`setup.sh` presents an interactive menu with the following options:

1. Setup required directories, compose configuration, and environment file.
2. Generate Compose configuration only
3. Start the trainer service
4. Setup and then start the trainer
5. Show configured clients
6. Exit

When you choose option `4`, `setup.sh` performs the required setup steps, builds and then launches the `trainer` service that runs a federated learning session.
If the trainer service completes successfully, the script automatically shuts down the Compose stack with `docker compose down`.

You can customize Compose settings by editing `machine_learning/.env` or `machine_learning/.env.example` before running `setup.sh`.

`setup.sh` is intended to start only the trainer service and its required dependencies, rather than manually launching every service in the stack.

## Docker Compose services

The Compose stack includes:

- `superlink` — Flower SuperLink service
- `supernode-1` and `supernode-2` — Flower SuperNodes for client app routing
- `superexec-serverapp` — custom built Flower `serverapp` container
- `superexec-clientapp-1` and `superexec-clientapp-2` — custom built Flower `clientapp` containers
- `trainer` — `flwr/superexec:1.33.0` runner that starts the federated training round

The server and client app containers share the same image built from `Dockerfile.superexec`, while the `trainer` uses the official Flower CLI image to launch the federation.

The server and client services load runtime configuration from `machine_learning/.env`, including:

- `DATA_DIR` for dataset location
- `BATCH_SIZE` for model training and evaluation batch size
- `CHECKPOINT_DIR` for client checkpoint storage
- `CLIENT_ID` per client container

Clients mount data from `./data/clientN:/app/data` and write checkpoints to `./checkpoints/clientN:/app/checkpoints`.

## Client checkpoint behavior

Each client saves checkpoints into its mounted directory under `./checkpoints/clientN/`.

## Running locally without Docker

If you prefer to run locally without Docker, install the repo and dependencies, then run Flower directly from `machine_learning/`:

```bash
pip install -e .
flwr run . local-deployment --stream
```

## Notes

- The current deployment uses Flower SuperLink / SuperNode / SuperExec rather than the older `--server` / `--client` CLI topology.
- `Dockerfile.superexec` pre-creates `/app/.flwr/apps` so Flower client apps can install without permission errors.
- The `trainer` service is responsible for starting the federated run and coordinating the loaded app definitions.
- The repo only tracks `machine_learning/.flwr/config.toml`; all other generated `.flwr/` runtime state is ignored.
