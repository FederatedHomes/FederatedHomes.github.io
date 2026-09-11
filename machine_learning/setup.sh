#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

load_environment() {
  if [ ! -f .env ]; then
    [ -f .example.env ] || { echo "ERROR: .example.env not found." >&2; return 1; }
    cp .example.env .env
    echo "Created .env from .example.env"
  fi
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
}

read_clients() {
  [ -f clients.yml ] || { echo "ERROR: clients.yml not found." >&2; return 1; }
}

ensure_host_dependencies() {
  local role="${1:-server}" missing_commands=()
  command -v python3 >/dev/null 2>&1 || missing_commands+=("python3")
  command -v docker >/dev/null 2>&1 || missing_commands+=("docker")
  if command -v docker >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
    missing_commands+=("docker compose")
  fi
  if [ "$role" = server ] && ! command -v openssl >/dev/null 2>&1; then
    missing_commands+=("openssl")
  elif [ "$role" = client ] && ! command -v ssh-keygen >/dev/null 2>&1; then
    missing_commands+=("ssh-keygen")
  fi
  if ((${#missing_commands[@]})); then
    echo "Missing required host tools: ${missing_commands[*]}" >&2
    return 1
  fi
  if ! python3 -c 'import yaml' >/dev/null 2>&1; then
    echo "Missing required Python library: PyYAML" >&2
    python3 -m pip --version >/dev/null 2>&1 || {
      echo "Install pip, then rerun ./setup.sh." >&2
      return 1
    }
    read -rp "Install PyYAML for this host now? [Y/n]: " choice
    [[ "$choice" =~ ^[Nn]$ ]] && {
      echo "ERROR: PyYAML is required." >&2
      return 1
    }
    python3 -m pip install --user PyYAML
  fi
}

require_production_profile() {
  [ "${DEPLOYMENT_PROFILE:-production}" = production ] || {
    echo "ERROR: Only DEPLOYMENT_PROFILE=production is supported. Remove the profile override from .env." >&2
    return 1
  }
}

read_client_ids() {
  CLIENT_IDS=()
  while IFS= read -r line; do
    [ -n "$line" ] && CLIENT_IDS+=("$line")
  done < <(python3 - <<'PY'
from pathlib import Path
import yaml
with Path("clients.yml").open(encoding="utf-8") as handle:
    for client in (yaml.safe_load(handle) or {}).get("clients", []):
        value = str(client.get("id", "")).strip()
        if value:
            print(value)
PY
)
}

select_host_role() {
  case "${DEPLOYMENT_ROLE:-}" in
    server|client) printf '%s\n' "$DEPLOYMENT_ROLE"; return ;;
  esac
  printf '\nWhat type of host are you preparing?\n  1) Server host — runs Flower SuperLink, ServerApp, trainer, and registration\n  2) Client host — runs one SuperNode and one ClientApp\n' >&2
  read -rp "Enter choice [1-2]: " choice
  case "$choice" in
    1) echo server ;;
    2) echo client ;;
    *) echo "ERROR: Invalid host role selection." >&2; return 1 ;;
  esac
}

select_client_id() {
  [ -n "${CLIENT_ID:-}" ] && { echo "$CLIENT_ID"; return; }
  read_client_ids
  ((${#CLIENT_IDS[@]})) || { echo "ERROR: No clients are configured." >&2; return 1; }
  echo "Select the client assigned to this machine:" >&2
  local i=1 id
  for id in "${CLIENT_IDS[@]}"; do
    printf '  %s) %s\n' "$i" "$id" >&2
    i=$((i+1))
  done
  read -rp "Enter client number: " choice
  [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#CLIENT_IDS[@]}" ] || {
    echo "ERROR: Invalid client selection." >&2
    return 1
  }
  echo "${CLIENT_IDS[$((choice-1))]}"
}

require_client_ca_certificate() {
  local ca_file="${TLS_CERTIFICATE_HOST_DIR:-./certificates/prod/tls}/ca.crt"
  [ -f "$ca_file" ] || {
    echo "ERROR: Required federation CA certificate was not found: $ca_file" >&2
    return 1
  }
  chmod 644 "$ca_file"
}

create_starter_tls_material() {
  [ "$1" = server ] || return 0
  local tls_dir="${TLS_CERTIFICATE_HOST_DIR:-./certificates/prod/tls}" starter_host="${SUPERLINK_HOST:-}"
  [ -n "$starter_host" ] || {
    echo "ERROR: SUPERLINK_HOST must be set before server preparation." >&2
    return 1
  }
  mkdir -p "$tls_dir"
  local ca_key="$tls_dir/.starter-ca.key"
  local ca_crt="$tls_dir/ca.crt"
  local superlink_key="$tls_dir/superlink.key"
  local superlink_crt="$tls_dir/superlink.crt"
  local csr="$tls_dir/.starter-superlink.csr"
  local ext="$tls_dir/.starter-superlink.ext"

  if [ -f "$ca_crt" ] && [ -f "$superlink_crt" ] && [ -f "$superlink_key" ]; then
    chmod 644 "$ca_crt" "$superlink_crt"
    chmod 600 "$superlink_key"
    echo "Existing TLS material detected; preserving it in $tls_dir."
    return 0
  fi
  if [ -f "$ca_crt" ] || [ -f "$superlink_crt" ] || [ -f "$superlink_key" ]; then
    echo "ERROR: Incomplete TLS material exists in $tls_dir. Refusing to overwrite existing credential material." >&2
    return 1
  fi

  echo "Creating starter federation TLS material for $starter_host..."
  openssl genrsa -out "$ca_key" 4096 >/dev/null 2>&1
  openssl req -x509 -new -nodes -key "$ca_key" -sha256 -days 3650 \
    -out "$ca_crt" -subj "/CN=FederatedHomes Starter CA" >/dev/null 2>&1
  openssl genrsa -out "$superlink_key" 2048 >/dev/null 2>&1
  openssl req -new -key "$superlink_key" -out "$csr" -subj "/CN=$starter_host" >/dev/null 2>&1
  if [[ "$starter_host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=IP:%s,DNS:localhost,IP:127.0.0.1\n' "$starter_host" > "$ext"
  else
    printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:%s,DNS:localhost,IP:127.0.0.1\n' "$starter_host" > "$ext"
  fi
  openssl x509 -req -in "$csr" -CA "$ca_crt" -CAkey "$ca_key" -CAcreateserial \
    -out "$superlink_crt" -days 825 -sha256 -extfile "$ext" >/dev/null 2>&1
  rm -f "$csr" "$ext" "$tls_dir/ca.srl" "$ca_key"
  chmod 644 "$ca_crt" "$superlink_crt"
  chmod 600 "$superlink_key"
  echo "Server TLS material is ready in $tls_dir."
  echo "WARNING: Starter TLS credentials are for development/testing only; replace them with federation-approved production credentials before production deployment."
  echo "Share only ca.crt with client hosts."
}

create_starter_client_auth() {
  local client_id="$1" auth_dir="${SUPERNODE_AUTH_HOST_DIR:-./certificates/prod/auth}"
  local private_key="$auth_dir/$client_id" public_key="$auth_dir/$client_id.pub"
  require_client_ca_certificate
  mkdir -p "$auth_dir"
  if [ -f "$private_key" ] && [ -f "$public_key" ]; then
    chmod 600 "$private_key"
    chmod 644 "$public_key"
    echo "Existing SuperNode authentication material detected for $client_id; preserving it."
    return 0
  fi
  if [ -f "$private_key" ] || [ -f "$public_key" ]; then
    echo "ERROR: Incomplete SuperNode authentication key pair for $client_id. Refusing to overwrite existing credential material." >&2
    return 1
  fi
  echo "Creating starter SuperNode authentication key pair for $client_id..."
  ssh-keygen -q -t ecdsa -b 384 -f "$private_key" -N "" -C "flower-supernode-$client_id"
  chmod 600 "$private_key"
  chmod 644 "$public_key"
  echo "WARNING: Starter SuperNode credentials are for development/testing only; replace them with federation-approved production credentials before production deployment."
}

prepare_flower_config() {
  python3 - <<'PY'
from pathlib import Path
import os
from src.deployment_config import load_deployment_config
config = load_deployment_config(role=os.environ.get("DEPLOYMENT_ROLE", "server"))
path = Path(".flwr/config.toml")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(f'''[superlink]\ndefault = "production"\n\n[superlink.supergrid]\naddress = "supergrid.flower.ai"\n\n[superlink.production-deployment]\n# GENERATED BY ./setup.sh FROM SUPERLINK_HOST. DO NOT EDIT THE ADDRESS HERE.\naddress = "{config.superlink_control_address}"\nroot-certificates = "/app/certificates/prod/tls/ca.crt"\n''', encoding="utf-8")
print(f"Generated {path}: Control API {config.superlink_control_address}")
PY
}

create_directories() {
  local role="$1" client_id="${2:-}"
  if [ "$role" = server ]; then
    mkdir -p "${SUPERLINK_STATE_HOST_DIR:-./state/superlink}" "./data/global" "./checkpoints/global"
    create_starter_tls_material server
  else
    create_starter_client_auth "$client_id"
    python3 - "$client_id" <<'PY'
from pathlib import Path
import sys
import yaml
requested = sys.argv[1]
with Path("clients.yml").open(encoding="utf-8") as handle:
    clients = (yaml.safe_load(handle) or {}).get("clients", [])
if len(clients) < 2:
    raise SystemExit("ERROR: clients.yml must define at least 2 clients.")
selected = [c for c in clients if str(c.get("id", "")).strip() == requested]
if not selected:
    raise SystemExit(f"ERROR: Client ID '{requested}' is not defined in clients.yml.")
for field in ("data_dir", "checkpoint_dir"):
    value = str(selected[0].get(field, "")).strip()
    if not value:
        raise SystemExit(f"ERROR: Client '{requested}' is missing '{field}'.")
    Path(value).mkdir(parents=True, exist_ok=True)
PY
  fi
}

validate_auth_environment() {
  local role="$1" client_id="${2:-}"
  python3 - "$role" "$client_id" <<'PY'
import sys
from src.deployment_config import load_deployment_config
role, client_id = sys.argv[1], sys.argv[2].strip()
config = load_deployment_config(role=role, require_files=True)
print(f"Validated SuperLink Fleet endpoint: {config.superlink_address}")
print(f"Validated SuperLink Control API: {config.superlink_control_address}")
if role == "client":
    if not client_id:
        raise SystemExit("ERROR: CLIENT_ID must be set for a client deployment.")
    private = config.supernode_auth_host_key(client_id)
    public = private.with_name(private.name + ".pub")
    for path in (private, public):
        if not path.is_file():
            raise SystemExit(f"Missing SuperNode authentication material: {path}")
PY
}

generate_server_compose() {
  python3 scripts/generate_compose.py \
    --config clients.yml \
    --output docker-compose.server.yml \
    --profile production \
    --role server
}

generate_client_compose() {
  local client_id="$1"
  python3 scripts/generate_compose.py \
    --config clients.yml \
    --output docker-compose.client.yml \
    --profile production \
    --role client \
    --client-id "$client_id"
}

prepare_host() {
  load_environment
  read_clients
  require_production_profile

  local role client_id=""
  role="$(select_host_role)"
  export DEPLOYMENT_ROLE="$role"
  ensure_host_dependencies "$role"

  if [ "$role" = client ]; then
    client_id="$(select_client_id)"
    export CLIENT_ID="$client_id"
  fi

  [ -n "${SUPERLINK_HOST:-}" ] || {
    echo "ERROR: Set SUPERLINK_HOST in .env before preparing a host." >&2
    return 1
  }

  prepare_flower_config
  create_directories "$role" "$client_id"
  validate_auth_environment "$role" "$client_id"

  if [ "$role" = server ]; then
    generate_server_compose
    echo "Generated docker-compose.server.yml"
  else
    generate_client_compose "$client_id"
    echo "Generated docker-compose.client.yml for $client_id"
  fi

  echo "Host preparation complete for role=$role${client_id:+, client=$client_id}."
}

require_prepared_compose() {
  local compose_file="$1" role="$2"
  if [ ! -f "$compose_file" ]; then
    echo "ERROR: $compose_file was not found." >&2
    echo "Run option 1 (Prepare host) for the $role host before starting infrastructure." >&2
    return 1
  fi
}

register_configured_clients() {
  local compose_file="${SERVER_COMPOSE_FILE:-docker-compose.server.yml}"
  local auth_dir="${SUPERNODE_AUTH_HOST_DIR:-./certificates/prod/auth}"
  [ -f "$compose_file" ] || {
    echo "ERROR: Server Compose file not found: $compose_file" >&2
    return 1
  }
  read_client_ids
  ((${#CLIENT_IDS[@]} >= 2)) || {
    echo "ERROR: At least 2 clients must be configured." >&2
    return 1
  }
  local id
  for id in "${CLIENT_IDS[@]}"; do
    [ -f "$auth_dir/$id.pub" ] || {
      echo "ERROR: Public key for $id was not found: $auth_dir/$id.pub" >&2
      return 1
    }
  done
  docker compose -f "$compose_file" run --rm client-registration
}

start_server_federation() {
  load_environment
  require_production_profile
  require_prepared_compose docker-compose.server.yml server
  docker compose -f docker-compose.server.yml build client-registration
  docker compose -f docker-compose.server.yml up -d --build superlink superexec-serverapp
  register_configured_clients
  echo "Server infrastructure is running. Start the trainer after the required clients are online."
}

start_client_federation() {
  load_environment
  require_production_profile
  require_prepared_compose docker-compose.client.yml client
  docker compose -f docker-compose.client.yml up --build
}

run_tests() {
  load_environment
  read_clients
  require_production_profile
  ensure_host_dependencies server
  docker build -f Dockerfile.superexec -t flwr_superexec:local .
  docker run --rm --entrypoint pytest -e PYTHONPATH=/app -v "$ROOT_DIR:/app" -w /app flwr_superexec:local tests/ -v
}

show_config() {
  load_environment
  read_clients
  echo "Deployment profile: production"
  echo "Deployment role: ${DEPLOYMENT_ROLE:-unset}"
  echo "Client ID: ${CLIENT_ID:-unset}"
  echo "SuperLink host: ${SUPERLINK_HOST:-unset}"
  python3 - <<'PY'
from src.deployment_config import load_deployment_config
try:
    c = load_deployment_config()
    print(f"Fleet API: {c.superlink_address}")
    print(f"Control API: {c.superlink_control_address}")
    print("TLS: enabled")
    print("SuperNode authentication: enabled")
except Exception as exc:
    print(f"Configuration validation failed: {exc}")
PY
  cat clients.yml
}

main_menu() {
  while true; do
    echo
    echo "FederatedHomes Flower deployment setup"
    echo "  1) Prepare host"
    echo "  2) Start server infrastructure"
    echo "  3) Start client infrastructure"
    echo "  4) Run tests"
    echo "  5) Show configuration"
    echo "  6) Exit"
    read -rp "Select an option [1-6]: " option
    case "$option" in
      1) prepare_host ;;
      2) start_server_federation ;;
      3) start_client_federation ;;
      4) run_tests ;;
      5) show_config ;;
      6) exit 0 ;;
      *) echo "ERROR: Invalid option." >&2 ;;
    esac
  done
}

main_menu
