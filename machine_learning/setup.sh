#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

ensure_host_dependencies() {
  local role="${1:-server}" missing_commands=()
  command -v python3 >/dev/null 2>&1 || missing_commands+=("python3")
  command -v docker >/dev/null 2>&1 || missing_commands+=("docker")
  if command -v docker >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then missing_commands+=("docker compose"); fi
  if [ "$role" = server ] && ! command -v openssl >/dev/null 2>&1; then missing_commands+=("openssl"); fi
  if [ "$role" = client ] && ! command -v ssh-keygen >/dev/null 2>&1; then missing_commands+=("ssh-keygen"); fi
  if ((${#missing_commands[@]})); then echo "Missing required host tools: ${missing_commands[*]}" >&2; return 1; fi
  if ! python3 -c 'import yaml' >/dev/null 2>&1; then
    echo "Missing required Python library: PyYAML" >&2
    python3 -m pip --version >/dev/null 2>&1 || { echo "Install pip, then rerun ./setup.sh." >&2; return 1; }
    read -rp "Install PyYAML for this host now? [Y/n]: " choice
    [[ "$choice" =~ ^[Nn]$ ]] && { echo "ERROR: PyYAML is required." >&2; return 1; }
    python3 -m pip install --user PyYAML || return 1
  fi
  echo "Host prerequisites verified for $role host."
}

load_environment() {
  if [ ! -f .env ]; then
    [ -f .env.example ] || { echo "ERROR: .env.example not found." >&2; return 1; }
    cp .env.example .env; echo "Created .env from .env.example"
  fi
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
}

read_clients() { [ -f clients.yml ] || { echo "ERROR: clients.yml not found." >&2; return 1; }; }

select_host_role() {
  case "${DEPLOYMENT_ROLE:-}" in server|client) printf '%s\n' "$DEPLOYMENT_ROLE"; return ;; esac
  printf '\nWhat type of host are you preparing?\n  1) Server host — runs Flower SuperLink and ServerApp\n  2) Client host — runs one SuperNode and one ClientApp\n' >&2
  read -rp "Enter choice [1-2]: " choice
  case "$choice" in 1) echo server ;; 2) echo client ;; *) echo "ERROR: Invalid host role selection." >&2; return 1 ;; esac
}

read_client_ids() {
  CLIENT_IDS=()
  while IFS= read -r line; do [ -n "$line" ] && CLIENT_IDS+=("$line"); done < <(python3 - <<'PY'
from pathlib import Path
import yaml
with Path("clients.yml").open(encoding="utf-8") as handle:
    for client in (yaml.safe_load(handle) or {}).get("clients", []):
        value = str(client.get("id", "")).strip()
        if value: print(value)
PY
)
}

select_client_id() {
  [ -n "${CLIENT_ID:-}" ] && { echo "$CLIENT_ID"; return; }
  read_client_ids; ((${#CLIENT_IDS[@]})) || { echo "ERROR: No clients are configured." >&2; return 1; }
  echo "Select the client assigned to this machine:" >&2
  local i=1 id; for id in "${CLIENT_IDS[@]}"; do printf '  %s) %s\n' "$i" "$id" >&2; i=$((i+1)); done
  read -rp "Enter client number: " choice
  [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#CLIENT_IDS[@]}" ] || { echo "ERROR: Invalid client selection." >&2; return 1; }
  echo "${CLIENT_IDS[$((choice-1))]}"
}

require_client_ca_certificate() {
  local ca_file="${TLS_CERTIFICATE_HOST_DIR:-./certificates/prod/tls}/ca.crt"
  [ -f "$ca_file" ] || { echo "ERROR: Required federation CA certificate was not found: $ca_file" >&2; return 1; }
  chmod 644 "$ca_file"
}

create_starter_tls_material() {
  [ "$1" = server ] || return 0
  local tls_dir="${TLS_CERTIFICATE_HOST_DIR:-./certificates/prod/tls}" starter_host="${SUPERLINK_HOST:-}"
  [ -n "$starter_host" ] || { echo "ERROR: SUPERLINK_HOST must be set for production server setup." >&2; return 1; }
  mkdir -p "$tls_dir"
  local ca_key="$tls_dir/.starter-ca.key" ca_crt="$tls_dir/ca.crt" superlink_key="$tls_dir/superlink.key" superlink_crt="$tls_dir/superlink.crt" csr="$tls_dir/.starter-superlink.csr" ext="$tls_dir/.starter-superlink.ext"
  if [ ! -f "$ca_crt" ] || [ ! -f "$superlink_crt" ] || [ ! -f "$superlink_key" ]; then
    echo "Creating starter federation CA and SuperLink certificate for $starter_host..."
    openssl genrsa -out "$ca_key" 4096 >/dev/null 2>&1
    openssl req -x509 -new -nodes -key "$ca_key" -sha256 -days 3650 -out "$ca_crt" -subj "/CN=FederatedHomes Starter CA" >/dev/null 2>&1
    openssl genrsa -out "$superlink_key" 2048 >/dev/null 2>&1
    openssl req -new -key "$superlink_key" -out "$csr" -subj "/CN=$starter_host" >/dev/null 2>&1
    if [[ "$starter_host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=IP:%s,DNS:localhost,IP:127.0.0.1\n' "$starter_host" > "$ext"
    else
      printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:%s,DNS:localhost,IP:127.0.0.1\n' "$starter_host" > "$ext"
    fi
    openssl x509 -req -in "$csr" -CA "$ca_crt" -CAkey "$ca_key" -CAcreateserial -out "$superlink_crt" -days 825 -sha256 -extfile "$ext" >/dev/null 2>&1
    rm -f "$csr" "$ext" "$tls_dir/ca.srl"
  fi
  rm -f "$ca_key"; chmod 644 "$ca_crt" "$superlink_crt"; chmod 600 "$superlink_key"
  echo "Server TLS material is ready in $tls_dir"
  echo "Share only ca.crt with client hosts."
}

create_starter_client_auth() {
  local client_id="$1" auth_dir="${SUPERNODE_AUTH_HOST_DIR:-./certificates/prod/auth}" private_key="$auth_dir/$client_id" public_key="$auth_dir/$client_id.pub"
  require_client_ca_certificate; mkdir -p "$auth_dir"
  if [ -f "$private_key" ] || [ -f "$public_key" ]; then
    [ -f "$private_key" ] && [ -f "$public_key" ] || { echo "ERROR: Incomplete SuperNode authentication key pair for $client_id." >&2; return 1; }
    chmod 600 "$private_key"; chmod 644 "$public_key"; return 0
  fi
  echo "Creating SuperNode authentication key pair for $client_id..."
  ssh-keygen -q -t ecdsa -b 384 -f "$private_key" -N "" -C "flower-supernode-$client_id"
  chmod 600 "$private_key"; chmod 644 "$public_key"
}

prepare_flower_config() {
  python3 - <<'PY'
from pathlib import Path
import os
from src.deployment_config import load_deployment_config
config = load_deployment_config(role=os.environ.get("DEPLOYMENT_ROLE", "server"))
path = Path(".flwr/config.toml")
path.parent.mkdir(parents=True, exist_ok=True)
existing = path.read_text(encoding="utf-8") if path.is_file() else ""
marker = "[superlink.production-deployment]"
prefix = existing.split(marker, 1)[0] if marker in existing else '''[superlink]\ndefault = "local"\n\n[superlink.supergrid]\naddress = "supergrid.flower.ai"\n\n[superlink.local]\naddress = ":local:"\n\n[superlink.local-deployment]\naddress = "superlink:9093"\ninsecure = true\n\n'''
section = f'''{marker}\n# GENERATED BY ./setup.sh FROM SUPERLINK_HOST. DO NOT EDIT THE ADDRESS HERE.\naddress = "{config.superlink_control_address}"\nroot-certificates = "/app/certificates/prod/tls/ca.crt"\n'''
path.write_text(prefix + section, encoding="utf-8")
print(f"Generated {path}: Control API {config.superlink_control_address}")
PY
}

create_directories() {
  local role="$1" client_id="${2:-}"
  if [ "$role" = server ]; then
    if [ "${DEPLOYMENT_PROFILE:-development}" = production ]; then
      [ -z "${SUPERLINK_STATE_HOST_DIR:-}" ] || mkdir -p "$SUPERLINK_STATE_HOST_DIR"
      create_starter_tls_material server
    fi
  else
    [ "${DEPLOYMENT_PROFILE:-development}" = production ] || { echo "ERROR: Physical client setup requires DEPLOYMENT_PROFILE=production." >&2; return 1; }
    create_starter_client_auth "$client_id"
    python3 - "$client_id" <<'PY'
from pathlib import Path
import sys,yaml
requested=sys.argv[1]
with Path("clients.yml").open(encoding="utf-8") as handle: clients=(yaml.safe_load(handle) or {}).get("clients",[])
if len(clients)<2: raise SystemExit("ERROR: clients.yml must define at least 2 clients.")
selected=[c for c in clients if str(c.get("id","")).strip()==requested]
if not selected: raise SystemExit(f"ERROR: Client ID '{requested}' is not defined in clients.yml.")
for field in ("data_dir","checkpoint_dir"):
    value=str(selected[0].get(field,"")).strip()
    if not value: raise SystemExit(f"ERROR: Client '{requested}' is missing '{field}'.")
    Path(value).mkdir(parents=True,exist_ok=True)
PY
  fi
}

validate_auth_environment() {
  local role="$1" client_id="${2:-}"
  [ "${DEPLOYMENT_PROFILE:-development}" = production ] || return 0
  python3 - "$role" "$client_id" <<'PY'
import sys
from src.deployment_config import load_deployment_config
role,client_id=sys.argv[1],sys.argv[2].strip()
config=load_deployment_config(role=role,require_files=True)
print(f"Validated SuperLink Fleet endpoint: {config.superlink_address}")
print(f"Validated SuperLink Control API: {config.superlink_control_address}")
if role=="client":
    if not client_id: raise SystemExit("ERROR: CLIENT_ID must be set for a client deployment.")
    private=config.supernode_auth_host_key(client_id); public=private.with_name(private.name+".pub")
    for path in (private,public):
        if not path.is_file(): raise SystemExit(f"Missing SuperNode authentication material: {path}")
else:
    if config.superlink_state_host_dir is None: raise SystemExit("ERROR: Server deployment requires SuperLink persistent state configuration.")
PY
}

prepare_host() {
  load_environment; read_clients
  local role client_id=""; role="$(select_host_role)"; export DEPLOYMENT_ROLE="$role"
  ensure_host_dependencies "$role"
  [ "$role" = client ] && { client_id="$(select_client_id)"; export CLIENT_ID="$client_id"; }
  if [ "${DEPLOYMENT_PROFILE:-development}" = production ]; then
    [ -n "${SUPERLINK_HOST:-}" ] || { echo "ERROR: Set SUPERLINK_HOST in .env before preparing a production host." >&2; return 1; }
    prepare_flower_config
  fi
  create_directories "$role" "$client_id"
  validate_auth_environment "$role" "$client_id"
  echo "Host preparation complete for role=$role${client_id:+, client=$client_id}."
}

generate_server_compose() {
  load_environment; read_clients; ensure_host_dependencies server; export DEPLOYMENT_ROLE=server
  [ "${DEPLOYMENT_PROFILE:-development}" = production ] && prepare_flower_config
  python3 scripts/generate_compose.py --config clients.yml --output docker-compose.server.yml --profile "${DEPLOYMENT_PROFILE:-development}" --role server
}

generate_client_compose() {
  load_environment; read_clients; ensure_host_dependencies client; export DEPLOYMENT_ROLE=client
  local client_id="${CLIENT_ID:-}"; [ -n "$client_id" ] || client_id="$(select_client_id)"; export CLIENT_ID="$client_id"
  [ "${DEPLOYMENT_PROFILE:-development}" = production ] && prepare_flower_config
  python3 scripts/generate_compose.py --config clients.yml --output docker-compose.client.yml --profile "${DEPLOYMENT_PROFILE:-development}" --role client --client-id "$client_id"
}

register_configured_clients() {
  [ "${DEPLOYMENT_PROFILE:-development}" = production ] || return 0
  local compose_file="${SERVER_COMPOSE_FILE:-docker-compose.server.yml}" auth_dir="${SUPERNODE_AUTH_HOST_DIR:-./certificates/prod/auth}"
  [ -f "$compose_file" ] || { echo "ERROR: Server Compose file not found: $compose_file" >&2; return 1; }
  read_client_ids; ((${#CLIENT_IDS[@]} >= 2)) || { echo "ERROR: At least 2 clients must be configured." >&2; return 1; }
  local id; for id in "${CLIENT_IDS[@]}"; do [ -f "$auth_dir/$id.pub" ] || { echo "ERROR: Public key for $id was not found: $auth_dir/$id.pub" >&2; return 1; }; done
  docker compose -f "$compose_file" run --rm client-registration
}

start_server_federation() { generate_server_compose; docker compose -f docker-compose.server.yml build --no-cache client-registration; docker compose -f docker-compose.server.yml up -d --build superlink superexec-serverapp; register_configured_clients; echo "Server infrastructure is running."; }
start_client_federation() { generate_client_compose; docker compose -f docker-compose.client.yml up --build; }

run_tests() {
  read_clients; load_environment; ensure_host_dependencies server; export DEPLOYMENT_PROFILE=development DEPLOYMENT_ROLE=all
  python3 scripts/generate_compose.py --config clients.yml --output "${DEV_COMPOSE_FILE:-docker-compose.generated.yml}" --profile development --role all
  docker compose -f "${DEV_COMPOSE_FILE:-docker-compose.generated.yml}" run --rm test-runner
}

show_config() {
  load_environment; read_clients
  echo "Deployment profile: ${DEPLOYMENT_PROFILE:-development}"; echo "Deployment role: ${DEPLOYMENT_ROLE:-unset}"; echo "Client ID: ${CLIENT_ID:-unset}"; echo "SuperLink host: ${SUPERLINK_HOST:-unset}"
  python3 - <<'PY'
from src.deployment_config import load_deployment_config
try:
    c=load_deployment_config()
    print(f"Fleet API: {c.superlink_address}")
    print(f"Control API: {c.superlink_control_address}")
except Exception as exc: print(f"Endpoint derivation unavailable: {exc}")
PY
  cat clients.yml
}

run_local_development_compose() { read_clients; load_environment; ensure_host_dependencies server; export DEPLOYMENT_PROFILE=development DEPLOYMENT_ROLE=all; python3 scripts/generate_compose.py --config clients.yml --output "${DEV_COMPOSE_FILE:-docker-compose.generated.yml}" --profile development --role all; docker compose -f "${DEV_COMPOSE_FILE:-docker-compose.generated.yml}" up --build; }

main_menu() {
  while true; do
    echo; echo "FederatedHomes Flower deployment setup"; echo "  1) Prepare host"; echo "  2) Generate server Compose"; echo "  3) Generate client Compose"; echo "  4) Start server infrastructure"; echo "  5) Start client infrastructure"; echo "  6) Run tests"; echo "  7) Show configuration"; echo "  8) Start local all-in-one development federation"; echo "  9) Exit"
    read -rp "Select an option [1-9]: " option
    case "$option" in 1) prepare_host ;; 2) generate_server_compose ;; 3) generate_client_compose ;; 4) start_server_federation ;; 5) start_client_federation ;; 6) run_tests ;; 7) show_config ;; 8) run_local_development_compose ;; 9) exit 0 ;; *) echo "ERROR: Invalid option." >&2 ;; esac
  done
}

main_menu
