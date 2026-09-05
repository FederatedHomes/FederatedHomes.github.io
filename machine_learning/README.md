---
tags: [federated, machine learning, vision, fds]
framework: [torch, torchvision]
---

# Federated Learning with PyTorch and Flower

This repository provides a Docker-based Flower deployment using the Flower SuperLink / SuperNode / SuperExec topology.

Each client can mount its own local CSV dataset from its configured client data directory and save checkpoints to its configured checkpoint directory.

## What this repo contains

- `Dockerfile.superexec` — runtime image for Flower `serverapp` and `clientapp`, including the application and testing dependencies
- `clients.yml` — source of truth for the configured federated learning clients
- `scripts/generate_compose.py` — generates the Docker Compose configuration from `clients.yml`
- `setup.sh` — interactive setup, Docker image build, testing, and federated training launcher
- `pyproject.toml` — Python project metadata and runtime dependencies
- `src/` — Flower application source
  - `src/server_app.py` — Flower `ServerApp`
  - `src/client_app.py` — Flower `ClientApp`
  - `src/task.py` — model, training, preprocessing, and local CSV dataset loading
  - `src/data_contract.py` — shared DataContract defining the model-facing data schema
- `tests/` — application and DataContract validation tests
  - `tests/conftest.py` — pytest fixtures for test data and isolated client datasets
  - `tests/test_data_contract_validation.py` — DataContract, preprocessing, segmentation, tensor, and multi-client validation tests

## Dependency refinement

`requirements.txt` is the source of runtime dependencies installed into the `flwr_superexec:local` image during Docker build.

The base image `flwr/superexec:1.33.0` already contains the Flower runtime, so `requirements.txt` deliberately omits `flwr` and installs only the additional application packages.

The Docker image also contains `pytest`, allowing the application test suite to run inside Docker without requiring the machine running the project to have the ML Python dependencies installed locally.

`pyproject.toml` remains the Python package metadata file used for local development, packaging, and the Flower app config.

## Data structure

The project expects a per-client dataset layout under `machine_learning/data/`, as configured by `clients.yml`.

For example:

```text
data/client1/train.csv
data/client1/val.csv

data/client2/train.csv
data/client2/val.csv
```

Each CSV must conform to the shared `DataContract` defined by the federated learning application.

The DataContract defines:

- Required feature columns
- Label column
- Feature and label data types
- Segmentation length and overlap
- Missing-value handling
- Continuous Wavelet Transform (CWT) configuration
- Model input tensor shape
- Model input tensor dtype
- Tensor layout

Client data is validated against this contract before being used by the model.

### Data validation behavior

The validation layer applies the following policies:

| Validation | Behavior |
|---|---|
| Missing feature | Reject |
| Extra feature | Warn and ignore |
| Reordered features | Accept |
| Convertible feature dtype | Convert and log |
| Convertible label dtype | Convert and log |
| Non-convertible feature values | Reject |
| Non-convertible label values | Reject |
| Fractional labels | Reject |
| Unknown labels | Reject |
| Incorrect segmentation length | Reject |
| Invalid segmentation configuration | Reject |
| Invalid generated tensor shape | Reject |
| Invalid generated tensor dtype | Reject |

This allows clients to provide compatible data with minor dtype differences while ensuring that incompatible data cannot silently enter the federated training process.

> `machine_learning/data/` and `machine_learning/checkpoints/` are not tracked in Git. These directories are created by `setup.sh` and are intended for local client datasets and checkpoint storage only.

## Setup and launch

From the `machine_learning/` folder:

```bash
chmod +x setup.sh
./setup.sh
```

`setup.sh` presents an interactive menu with the following options:

1. Setup required directories and environment file
2. Generate Compose configuration only
3. Build image and start the trainer
4. Setup, build image, and start the trainer
5. Run application tests in Docker
6. Show configured clients
7. Exit

### Option 5 — Run application tests in Docker

Option `5` runs the project's pytest test suite inside a Docker container.

The test runner uses the same custom `flwr_superexec:local` image that contains the application's Python dependencies. This means the host machine does **not** need local installations of PyTorch, pandas, NumPy, PyWavelets, scikit-image, pytest, or the other application dependencies.

The process is:

```text
setup.sh
   |
   +-- Build flwr_superexec:local
   |
   +-- Start test-runner container
   |
   +-- pytest tests/ -v
   |
   +-- Remove test container
```

The test runner is isolated from the federated runtime. It does not start the SuperLink, SuperNodes, or federated training services.

To run the tests:

```bash
./setup.sh
```

Select:

```text
5) Run application tests in Docker
```

The test suite currently validates:

- Valid client data
- Independent validation of multiple clients
- Missing required features
- Extra features
- Reordered feature columns
- Feature dtype conversion
- Numeric string feature conversion
- Non-convertible feature values
- Label dtype conversion
- Numeric string label conversion
- Non-convertible labels
- Fractional labels
- Unknown labels
- Segmentation length
- Invalid DataContract configuration
- Generated tensor dimensions
- Generated tensor dtype
- Mixed multi-client validation

A successful run should report all tests as `PASSED`.

### Why tests run in Docker

The project intentionally runs the test suite using the Docker application environment rather than requiring developers to install the ML dependencies locally.

This provides a consistent test environment and ensures that the tests execute against the same dependency environment used by the federated application.

The host therefore only needs the tools required to build and run the Docker environment.

## Federated training

When option `4` is selected, `setup.sh`:

1. Creates the required client data and checkpoint directories
2. Creates `.env` from `.env.example` when required
3. Generates `docker-compose.generated.yml` from `clients.yml`
4. Checks for the configured client CSV files
5. Builds the shared `flwr_superexec:local` image
6. Starts the Flower trainer and its required dependencies
7. Runs the federated learning session
8. Shuts down the Compose stack after successful completion

The generated Compose configuration is based on the clients defined in `clients.yml`, allowing the number of local test clients to be changed without manually maintaining individual Compose service definitions.

## Docker Compose services

The generated Compose stack includes:

- `superlink` — Flower SuperLink service
- `supernode-*` — Flower SuperNode services generated for the configured clients
- `superexec-serverapp` — custom Flower `serverapp` container
- `superexec-clientapp-*` — custom Flower `clientapp` containers generated for the configured clients
- `trainer` — Flower `superexec:1.33.0` runner that starts the federated training session
- `test-runner` — temporary test container that runs the pytest application test suite

The custom application services and `test-runner` share the `flwr_superexec:local` image.

The `trainer` uses the official Flower `flwr/superexec:1.33.0` image to launch the federated run.

## Client configuration

`clients.yml` is the source of truth for the configured clients.

Each client defines its:

- Client ID
- Data directory
- Checkpoint directory

For example:

```yaml
clients:
  - id: client-a
    data_dir: ./data/client-a
    checkpoint_dir: ./checkpoints/client-a

  - id: client-b
    data_dir: ./data/client-b
    checkpoint_dir: ./checkpoints/client-b
```

At least two clients are required for the federated learning deployment.

The Compose configuration is generated automatically from this file.

## Client checkpoint behavior

Each client saves checkpoints into its configured mounted checkpoint directory.

For example:

```text
checkpoints/
├── client-a/
└── client-b/
```

This allows each client to maintain its own local model/checkpoint state.

## Running locally without Docker

The recommended development and testing workflow is Docker-based.

If you prefer to run the application locally without Docker, install the repository and its dependencies, then run Flower directly from `machine_learning/`:

```bash
pip install -e .
flwr run . local-deployment --stream
```

When running the test suite directly on the host, the required Python dependencies, including pytest and the ML dependencies, must also be installed locally.

For a consistent dependency environment, use `setup.sh` option `5` instead.

## Development workflow

A recommended development workflow is:

```text
1. Modify application or DataContract code
          |
          v
2. Run ./setup.sh
          |
          v
3. Select option 5
          |
          v
4. Run Docker-based application tests
          |
          v
5. Fix any validation/test failures
          |
          v
6. Run option 4 for federated training
          |
          v
7. Commit the validated changes
```

This keeps application validation separate from the full federated runtime while still providing an easy path to end-to-end Docker-based training.

## Notes

- The current deployment uses Flower SuperLink / SuperNode / SuperExec rather than the older `--server` / `--client` CLI topology.
- `Dockerfile.superexec` provides the shared application runtime environment used by the server/client applications and the Docker-based test runner.
- `clients.yml` is the source of truth for the number and configuration of clients.
- `scripts/generate_compose.py` generates the Compose configuration from `clients.yml`.
- `setup.sh` is the primary entry point for setup, testing, and federated training.
- Option `5` runs the pytest application test suite without requiring local ML library installation.
- The `trainer` service is responsible for starting the federated run and coordinating the loaded app definitions.
- The current local Docker deployment is intended for development and integration testing; production secure communication configuration should be completed before deploying across separate physical client machines.