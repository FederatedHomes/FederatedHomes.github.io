#!/usr/bin/env python3
"""Reconcile configured Flower SuperNode registrations with clients.yml."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


CLIENTS_FILE = Path(os.environ.get("CLIENTS_FILE", "/app/clients.yml"))
PUBLIC_KEY_DIR = Path(os.environ.get("PUBLIC_KEY_DIR", "/app/certificates/prod/auth"))
PROFILE = os.environ.get("FLOWER_PROFILE", "production-deployment")
FLOWER_CONFIG_DIR = Path(os.environ.get("FLOWER_CONFIG_DIR", "/app/.flwr"))
FLOWER_HOME = Path(os.environ.get("FLOWER_HOME", "/tmp/flower-cli-home"))
REGISTRY_STATE_FILE = Path(
    os.environ.get("REGISTRY_STATE_FILE", "/app/state/registered_nodes.json")
)
MIN_CLIENTS = 2
ALREADY_REGISTERED_MESSAGE = "Public key already in use"


class ConfigError(RuntimeError):
    pass


def unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_clients(path: Path) -> list[dict[str, str]]:
    """Parse the intentionally small clients.yml structure used by this deployment."""
    if not path.is_file():
        raise ConfigError(f"clients.yml not found: {path}")

    clients: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_clients = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "clients:":
            in_clients = True
            continue
        if not in_clients:
            continue
        if line.startswith("- "):
            if current is not None:
                clients.append(current)
            current = {}
            key_value = line[2:].strip()
            if ":" in key_value:
                key, value = key_value.split(":", 1)
                current[key.strip()] = unquote(value.strip())
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = unquote(value.strip())

    if current is not None:
        clients.append(current)

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for client in clients:
        client_id = client.get("id", "").strip()
        public_key = client.get("public_key", "").strip()
        if not client_id:
            raise ConfigError("Every configured client must define an id.")
        if client_id in seen:
            raise ConfigError(f"Duplicate client ID in clients.yml: {client_id}")
        if not public_key:
            raise ConfigError(f"Client '{client_id}' must define public_key.")
        seen.add(client_id)
        normalized.append({"id": client_id, "public_key": public_key})

    if len(normalized) < MIN_CLIENTS:
        raise ConfigError(
            f"At least {MIN_CLIENTS} clients are required; found {len(normalized)}."
        )
    return normalized


def canonical_public_key(client: dict[str, str]) -> Path:
    client_id = client["id"]
    configured = Path(client["public_key"])
    if configured.name != f"{client_id}.pub":
        raise ConfigError(
            f"Client '{client_id}' public_key must end with '{client_id}.pub'. "
            f"Got: {configured}"
        )
    return PUBLIC_KEY_DIR / f"{client_id}.pub"


def public_key_fingerprint(public_key: Path) -> str:
    return hashlib.sha256(public_key.read_bytes()).hexdigest()


def prepare_flower_home() -> Path:
    """Create a writable CLI home from the setup-generated Flower config."""
    source = FLOWER_CONFIG_DIR / "config.toml"
    if not source.is_file():
        raise ConfigError(
            f"Generated Flower configuration not found: {source}. "
            "Run './setup.sh' option 1 (Prepare host) first."
        )

    config = source.read_text(encoding="utf-8")
    if PROFILE == "production-deployment":
        marker = "[superlink.production-deployment]"
        if marker not in config:
            raise ConfigError(
                "Generated Flower configuration does not contain the production-deployment profile."
            )
        profile_section = config.split(marker, 1)[1].split("[", 1)[0]
        if "address =" not in profile_section:
            raise ConfigError(
                "Generated production SuperLink profile does not contain an address entry."
            )

    config_dir = FLOWER_HOME / ".flwr"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(config, encoding="utf-8")
    return FLOWER_HOME


def run_flower(home: Path, args: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    completed = subprocess.run(
        ["flwr", *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    ).strip()
    return completed.returncode, output


def parse_json(output: str) -> dict:
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ConfigError(f"Flower CLI returned invalid JSON: {output}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Flower CLI returned a JSON value that is not an object.")
    return payload


def load_registry() -> dict[str, dict[str, str]]:
    if not REGISTRY_STATE_FILE.exists():
        return {}
    try:
        payload = json.loads(REGISTRY_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"Unable to read registration state file: {REGISTRY_STATE_FILE}"
        ) from exc
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, dict):
        raise ConfigError(
            f"Invalid registration state file: {REGISTRY_STATE_FILE}. "
            "Expected a JSON object containing a 'nodes' mapping."
        )
    return {
        str(client_id): {
            "node-id": str(entry["node-id"]),
            "public-key-sha256": str(entry["public-key-sha256"]),
        }
        for client_id, entry in nodes.items()
        if isinstance(entry, dict)
        and "node-id" in entry
        and "public-key-sha256" in entry
    }


def save_registry(registry: dict[str, dict[str, str]]) -> None:
    REGISTRY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "nodes": registry}
    temporary = REGISTRY_STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(REGISTRY_STATE_FILE)


def list_registered(home: Path) -> tuple[bool, list[dict[str, str]], str]:
    returncode, output = run_flower(
        home,
        ["supernode", "list", PROFILE, "--format", "json", "--verbose"],
    )
    if returncode != 0:
        return False, [], output or "Flower SuperNode list command failed"
    try:
        payload = parse_json(output)
    except ConfigError as exc:
        return False, [], str(exc)
    if payload.get("success") is not True:
        return False, [], output or "Flower SuperNode list command was unsuccessful"
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return False, [], "Flower SuperNode list JSON does not contain a 'nodes' array"
    normalized: list[dict[str, str]] = []
    for node in nodes:
        if not isinstance(node, dict) or not str(node.get("node-id", "")).strip():
            return False, [], "Flower SuperNode list contains an entry without a node-id"
        normalized.append({"node-id": str(node["node-id"])})
    return True, normalized, output


def register_one(home: Path, client: dict[str, str]) -> tuple[str, str, str | None]:
    client_id = client["id"]
    public_key = canonical_public_key(client)
    if not public_key.is_file():
        return "FAILED", f"public key not mounted: {public_key}", None

    returncode, output = run_flower(
        home,
        ["supernode", "register", str(public_key), PROFILE, "--format", "json"],
    )

    try:
        payload = parse_json(output)
    except ConfigError:
        payload = {}

    success = payload.get("success")
    node_id = payload.get("node-id") or payload.get("node_id")
    if returncode == 0 and success is not False:
        return "REGISTERED", output or "registration completed", str(node_id) if node_id else None

    if ALREADY_REGISTERED_MESSAGE in output:
        return "ALREADY_REGISTERED", output, str(node_id) if node_id else None

    return "FAILED", output or "Flower registration command failed", None


def unregister_one(home: Path, node_id: str) -> tuple[bool, str]:
    returncode, output = run_flower(
        home,
        ["supernode", "unregister", node_id, PROFILE, "--format", "json"],
    )
    try:
        payload = parse_json(output)
        success = payload.get("success")
    except ConfigError:
        success = None
    return returncode == 0 and success is not False, output or "unregistration completed"


def main() -> int:
    try:
        clients = parse_clients(CLIENTS_FILE)
        home = prepare_flower_home()
        registry = load_registry()
    except (ConfigError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    desired_fingerprints: dict[str, str] = {}
    for client in clients:
        public_key = canonical_public_key(client)
        if not public_key.is_file():
            print(f"ERROR: Public key not mounted for {client['id']}: {public_key}", file=sys.stderr)
            return 1
        desired_fingerprints[client["id"]] = public_key_fingerprint(public_key)

    print(
        f"Reconciling {len(clients)} configured SuperNodes with Flower profile '{PROFILE}'.",
        flush=True,
    )

    list_ok, registered_nodes, listing = list_registered(home)
    if not list_ok:
        print(f"ERROR: Unable to list Flower SuperNodes: {listing}", file=sys.stderr)
        return 1

    actual_node_ids = {node["node-id"] for node in registered_nodes}

    # The list command intentionally exposes node IDs but not public keys. The
    # local manifest therefore records the node ID assigned to each configured
    # public key at registration time. Without it, we cannot safely decide
    # which pre-existing node is stale and must refuse to guess.
    known_registry: dict[str, dict[str, str]] = {}
    for client_id, entry in registry.items():
        if entry["node-id"] in actual_node_ids:
            known_registry[client_id] = entry

    unknown_existing = actual_node_ids - {
        entry["node-id"] for entry in known_registry.values()
    }
    if unknown_existing:
        print(
            "ERROR: Flower contains SuperNodes that are not present in the local "
            "registration manifest. The Flower list API does not expose public "
            "keys, so removing these nodes automatically would risk deleting a "
            "valid client.\n"
            f"Unmanaged node IDs: {', '.join(sorted(unknown_existing))}\n"
            f"Registration manifest: {REGISTRY_STATE_FILE}\n"
            "Perform the one-time migration of these existing registrations before "
            "enabling automatic stale-node removal.",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []

    # Register any configured public key which is not already represented by
    # the persistent manifest. Successful registration returns the new node ID.
    for client in clients:
        client_id = client["id"]
        fingerprint = desired_fingerprints[client_id]
        entry = known_registry.get(client_id)
        if entry and entry["public-key-sha256"] == fingerprint:
            print(f"  {client_id}: REGISTERED (existing)", flush=True)
            continue

        if entry and entry["public-key-sha256"] != fingerprint:
            ok, detail = unregister_one(home, entry["node-id"])
            if not ok:
                print(f"  {client_id}: FAILED to remove old registration: {detail}", file=sys.stderr)
                failures.append(client_id)
                continue
            known_registry.pop(client_id, None)

        print(f"\nRegistering {client_id}...", flush=True)
        status, detail, node_id = register_one(home, client)
        print(f"  {client_id}: {status}", flush=True)
        if status == "FAILED":
            failures.append(client_id)
            continue
        if not node_id:
            print(
                f"ERROR: Flower did not return a node-id for {client_id}; "
                "cannot maintain the registration manifest safely.",
                file=sys.stderr,
            )
            failures.append(client_id)
            continue
        known_registry[client_id] = {
            "node-id": node_id,
            "public-key-sha256": fingerprint,
        }

    if failures:
        print(
            f"ERROR: Registration failed for: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1

    # Remove configured clients from the manifest which are no longer present
    # in clients.yml. Their node IDs are known and can therefore be safely
    # unregistered.
    desired_ids = set(desired_fingerprints)
    stale_clients = set(known_registry) - desired_ids
    for client_id in sorted(stale_clients):
        node_id = known_registry[client_id]["node-id"]
        print(f"\nUnregistering stale client {client_id} (node {node_id})...", flush=True)
        ok, detail = unregister_one(home, node_id)
        if not ok:
            print(f"  {client_id}: FAILED to unregister: {detail}", file=sys.stderr)
            failures.append(client_id)
            continue
        print(f"  {client_id}: UNREGISTERED", flush=True)
        known_registry.pop(client_id, None)

    save_registry(known_registry)

    final_ok, final_nodes, final_listing = list_registered(home)
    if not final_ok:
        print(f"ERROR: Final Flower SuperNode listing failed: {final_listing}", file=sys.stderr)
        return 1

    final_ids = {node["node-id"] for node in final_nodes}
    expected_ids = {entry["node-id"] for entry in known_registry.values()}
    if final_ids != expected_ids:
        print(
            "ERROR: Flower registry does not match the managed registration manifest.\n"
            f"Expected node IDs: {', '.join(sorted(expected_ids)) or '(none)'}\n"
            f"Actual node IDs:   {', '.join(sorted(final_ids)) or '(none)'}",
            file=sys.stderr,
        )
        return 1

    print("\nFlower SuperNode registry reconciled successfully.", flush=True)
    print(final_listing, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
