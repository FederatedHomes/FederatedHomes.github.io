#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

create_directories() {
  mkdir -p data/client1
  mkdir -p data/client2
  mkdir -p checkpoints/client1
  mkdir -p checkpoints/client2
}

create_env_file() {
  if [ ! -f .env ]; then
    if [ -f .env.example ]; then
      cp .env.example .env
      echo "Created .env from .env.example"
    else
      echo "Warning: .env.example not found. Create .env manually if needed."
    fi
  fi
}

warn_missing_csvs() {
  if [ ! -f data/client1/train.csv ] || [ ! -f data/client1/val.csv ] || [ ! -f data/client2/train.csv ] || [ ! -f data/client2/val.csv ]; then
    cat <<'EOF'
WARNING: Example data directories were created, but some CSV files are missing.
The clients will use synthetic mock data instead of real data until CSVs are provided.
To run with real data later, add the following files:
  data/client1/train.csv
  data/client1/val.csv
  data/client2/train.csv
  data/client2/val.csv
Each CSV should include a `label` column and either:
  - `img_path` relative to the CSV file directory, or
  - numeric features columns only.
EOF
  fi
}

setup() {
  create_directories
  create_env_file
  warn_missing_csvs
}

start_trainer() {
  echo "Starting Docker Compose trainer and dependencies..."
  if docker compose up --build trainer; then
    echo "Trainer completed successfully. Shutting down the Compose stack..."
    docker compose down
  else
    echo "Trainer exited with an error. Leaving Compose stack running for inspection."
    return 1
  fi
}

print_menu() {
  cat <<'EOF'
Select an option:
  1) Setup required directories and environment file only
  2) Start the trainer service
  3) Setup and then start the trainer
  4) Exit
EOF
}

print_menu
read -rp "Enter choice [1-4]: " choice
case "$choice" in
  1)
    setup
    ;;
  2)
    start_trainer
    ;;
  3)
    setup
    start_trainer
    ;;
  *)
    echo "No action selected. Exiting."
    exit 0
    ;;
esac
