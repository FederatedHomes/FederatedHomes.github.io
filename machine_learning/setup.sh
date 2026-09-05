#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"


create_directories() {
  mkdir -p data/global
  mkdir -p checkpoints/global

  if [ ! -f clients.yml ]; then
    echo "Warning: clients.yml not found."
    echo "Create clients.yml before running the federated learning stack."
    return
  fi

  python3 - <<'PY'
from pathlib import Path
import yaml

config_path = Path("clients.yml")

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

clients = config.get("clients", [])

if len(clients) < 2:
    raise SystemExit(
        "ERROR: clients.yml must define at least 2 clients."
    )

for client in clients:
    client_id = str(client.get("id", "")).strip()
    data_dir = str(client.get("data_dir", "")).strip()
    checkpoint_dir = str(client.get("checkpoint_dir", "")).strip()

    if not client_id:
        raise SystemExit(
            "ERROR: Every client must define an 'id'."
        )

    if not data_dir:
        raise SystemExit(
            f"ERROR: Client '{client_id}' is missing 'data_dir'."
        )

    if not checkpoint_dir:
        raise SystemExit(
            f"ERROR: Client '{client_id}' is missing "
            "'checkpoint_dir'."
        )

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Prepared {client_id}: "
        f"data={data_dir}, "
        f"checkpoints={checkpoint_dir}"
    )
PY
}


create_env_file() {
  if [ ! -f .env ]; then
    if [ -f .env.example ]; then
      cp .env.example .env
      echo "Created .env from .env.example"
    else
      echo "Warning: .env.example not found."
      echo "Create .env manually if needed."
    fi
  fi
}


generate_compose_file() {
  if [ ! -f clients.yml ]; then
    echo "ERROR: clients.yml not found."
    return 1
  fi

  if [ ! -f generate_compose.py ]; then
    echo "ERROR: generate_compose.py not found."
    return 1
  fi

  echo "Generating Docker Compose configuration..."

  python3 generate_compose.py \
    --config clients.yml \
    --output docker-compose.generated.yml

  if [ ! -f docker-compose.generated.yml ]; then
    echo "ERROR: Docker Compose file was not generated."
    return 1
  fi

  echo "Generated docker-compose.generated.yml"
}


warn_missing_csvs() {
  if [ ! -f clients.yml ]; then
    echo "WARNING: clients.yml not found; cannot check client CSV files."
    return
  fi

  python3 - <<'PY'
from pathlib import Path
import yaml

config_path = Path("clients.yml")

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

clients = config.get("clients", [])

missing = []

for client in clients:
    client_id = str(client.get("id", "")).strip()
    data_dir = Path(str(client.get("data_dir", "")).strip())

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    if not train_csv.is_file():
        missing.append(str(train_csv))

    if not val_csv.is_file():
        missing.append(str(val_csv))

if missing:
    cat <<EOF
WARNING: Some client CSV files are missing.

The affected clients will use synthetic mock data until
the required CSV files are provided.

Missing files:
EOF

    for file in missing:
        echo "  $file"

    cat <<'EOF'

Each client data directory should normally contain:
  train.csv
  val.csv

The CSV files should conform to the DataContract defined by
the federated learning application.
EOF
else
    echo "All configured client train/validation CSV files are present."
fi
PY
}


setup() {
  echo "=========================================="
  echo "Federated Learning Environment Setup"
  echo "=========================================="

  create_directories
  create_env_file
  generate_compose_file
  warn_missing_csvs

  echo
  echo "Setup completed."
}

run_tests() {
  echo "=========================================="
  echo "Running application tests in Docker"
  echo "=========================================="

  if [ ! -d tests ]; then
    echo "ERROR: tests directory not found."
    return 1
  fi

  if ! find tests -maxdepth 1 -name 'test_*.py' -print -quit | grep -q .; then
    echo "ERROR: No pytest test files found in tests/."
    return 1
  fi

  if [ ! -f docker-compose.generated.yml ]; then
    echo "docker-compose.generated.yml not found."
    echo "Generating it from clients.yml..."

    generate_compose_file
  fi

  echo
  echo "=========================================="
  echo "Building shared Flower SuperExec image"
  echo "=========================================="

  if ! docker build \
      -f Dockerfile.superexec \
      -t flwr_superexec:local \
      .; then

    echo "ERROR: Failed to build flwr_superexec:local."
    return 1
  fi

  echo
  echo "Shared SuperExec image built successfully."
  echo

  echo "=========================================="
  echo "Running pytest inside Docker"
  echo "=========================================="

  if docker compose \
      -f docker-compose.generated.yml \
      run --rm test-runner; then

    echo
    echo "=========================================="
    echo "All application tests passed."
    echo "=========================================="

  else

    echo
    echo "=========================================="
    echo "Application tests failed."
    echo "=========================================="

    return 1
  fi
}

start_trainer() {
  if [ ! -f docker-compose.generated.yml ]; then
    echo "docker-compose.generated.yml not found."
    echo "Generating it from clients.yml..."

    generate_compose_file
  fi

  echo "=========================================="
  echo "Building shared Flower SuperExec image"
  echo "=========================================="

  if ! docker build \
      -f Dockerfile.superexec \
      -t flwr_superexec:local \
      .; then

    echo "ERROR: Failed to build flwr_superexec:local."
    return 1
  fi

  echo
  echo "Shared SuperExec image built successfully."
  echo

  echo "=========================================="
  echo "Starting Docker Compose trainer and dependencies"
  echo "=========================================="

  if docker compose \
      -f docker-compose.generated.yml \
      up trainer; then

    echo
    echo "Trainer completed successfully."
    echo "Shutting down the Compose stack..."

    docker compose \
      -f docker-compose.generated.yml \
      down

  else

    echo
    echo "Trainer exited with an error."
    echo "Leaving the Compose stack running for inspection."

    return 1
  fi
}


print_config() {
  echo
  echo "Configured clients:"

  python3 - <<'PY'
from pathlib import Path
import yaml

config_path = Path("clients.yml")

if not config_path.exists():
    print("  clients.yml not found")
    raise SystemExit(0)

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

clients = config.get("clients", [])

for client in clients:
    client_id = str(client.get("id", "")).strip()
    data_dir = str(client.get("data_dir", "")).strip()
    checkpoint_dir = str(client.get("checkpoint_dir", "")).strip()

    print(f"  {client_id}")
    print(f"    data:        {data_dir}")
    print(f"    checkpoints: {checkpoint_dir}")
PY

  echo
}


print_menu() {
  cat <<'EOF'

Select an option:
  1) Setup required directories, compose configuration, and environment file.
  2) Generate Compose configuration only
  3) Start the trainer service
  4) Setup and then start the trainer
  5) Run application tests in Docker
  6) Show configured clients
  7) Exit
EOF
}


print_menu
read -rp "Enter choice [1-6]: " choice

case "$choice" in
  1)
    setup
    ;;
  2)
    generate_compose_file
    ;;
  3)
    start_trainer
    ;;
  4)
    setup
    start_trainer
    ;;
  5)
    run_tests
    ;;
  6)
    print_config
    ;;
  7)
    echo "Exiting."
    exit 0
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac