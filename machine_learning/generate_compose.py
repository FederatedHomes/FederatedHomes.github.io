#!/usr/bin/env python3
"""Generate an N-client Flower 1.33.0 Docker Compose deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


SUPERNODE_PORT = 9094
SUPERNODE_IMAGE = "flwr/supernode:1.33.0"
SUPEREXEC_IMAGE = "flwr_superexec:local"


def validate_clients(clients: list[dict]) -> None:
    """Validate the client configuration."""

    if len(clients) < 2:
        raise ValueError("At least 2 clients are required.")

    ids = [str(client.get("id", "")).strip() for client in clients]

    if any(not client_id for client_id in ids):
        raise ValueError("Every client must define a non-empty 'id'.")

    if len(ids) != len(set(ids)):
        raise ValueError("Client IDs must be unique.")

    for client in clients:
        for key in ("data_dir", "checkpoint_dir"):
            if not str(client.get(key, "")).strip():
                raise ValueError(
                    f"Client '{client['id']}' must define '{key}'."
                )


def safe_id(client_id: str) -> str:
    """Convert a client ID into a Docker Compose service-name fragment."""

    return client_id.strip().lower().replace("_", "-").replace(" ", "-")


def node_name(client_id: str) -> str:
    """Return the SuperNode service name."""

    return f"supernode-{safe_id(client_id)}"


def app_name(client_id: str) -> str:
    """Return the SuperExec ClientApp service name."""

    return f"superexec-clientapp-{safe_id(client_id)}"


def build_compose(clients: list[dict]) -> dict:
    """Build the logical Docker Compose model."""

    validate_clients(clients)

    services = {
        "superlink": {
            "image": "flwr/superlink:1.33.0",
            "container_name": "flwr_superlink",
            "command": ["--insecure", "--isolation", "process"],
            "ports": ["9091:9091", "9092:9092", "9093:9093"],
            "networks": ["flwr-network"],
        }
    }

    node_services = []
    app_services = []

    for client in clients:
        client_id = str(client["id"]).strip()
        node = node_name(client_id)
        app = app_name(client_id)
        node_services.append(node)
        app_services.append(app)

        services[node] = {
            "container_name": f"flwr_{node.replace('-', '_')}",
            "command": [
                "--insecure",
                "--superlink",
                "superlink:9092",
                "--clientappio-api-address",
                f"0.0.0.0:{SUPERNODE_PORT}",
                "--isolation",
                "process",
            ],
            "networks": ["flwr-network"],
            "depends_on": ["superlink"],
        }

        services[app] = {
            "container_name": f"flwr_{app.replace('-', '_')}",
            "env_file": [".env"],
            "command": [
                "--insecure",
                "--plugin-type",
                "clientapp",
                "--appio-api-address",
                f"{node}:{SUPERNODE_PORT}",
            ],
            "networks": ["flwr-network"],
            "volumes": [
                f"{client['data_dir']}:${{DATA_DIR}}",
                f"{client['checkpoint_dir']}:${{CHECKPOINT_DIR}}",
            ],
            "environment": {"CLIENT_ID": client_id},
            "depends_on": [node, "superlink"],
        }

    services["superexec-serverapp"] = {
        "container_name": "flwr_superexec_serverapp",
        "env_file": [".env"],
        "command": [
            "--insecure",
            "--plugin-type",
            "serverapp",
            "--appio-api-address",
            "superlink:9091",
        ],
        "networks": ["flwr-network"],
        "volumes": [
            "./checkpoints/global:${CHECKPOINT_DIR}",
            "./data/global:${DATA_DIR}",
        ],
        "depends_on": ["superlink"],
    }

    services["trainer"] = {
        "image": "flwr/superexec:1.33.0",
        "container_name": "flwr_trainer",
        "entrypoint": ["flwr"],
        "command": ["run", ".", "local-deployment", "--stream"],
        "working_dir": "/app",
        "volumes": [".:/app"],
        "networks": ["flwr-network"],
        "depends_on": [
            "superlink",
            "superexec-serverapp",
            *node_services,
            *app_services,
        ],
    }

    services["test-runner"] = {
        "container_name": "flwr_test_runner",
        "entrypoint": ["pytest"],
        "command": ["tests/", "-v"],
        "working_dir": "/app",
        "environment": {"PYTHONPATH": "/app"},
        "volumes": [".:/app"],
        "networks": ["flwr-network"],
    }

    return {
        "networks": {"flwr-network": {"driver": "bridge"}},
        "services": services,
        "volumes": {"data": {}, "checkpoints": {}},
    }


def render_compose(compose: dict) -> str:
    """Render Docker Compose with shared image YAML anchors."""

    lines = [
        "networks:",
        "  flwr-network:",
        "    driver: bridge",
        "",
        "# Shared Flower SuperNode image",
        "x-flwr-supernode: &flwr_supernode",
        f"  image: {SUPERNODE_IMAGE}",
        "",
        "# Shared custom SuperExec image",
        "x-flwr-superexec: &flwr_superexec",
        f"  image: {SUPEREXEC_IMAGE}",
        "",
        "services:",
    ]

    for name, service in compose["services"].items():
        lines.append(f"  {name}:")

        if name.startswith("supernode-"):
            lines.append("    <<: *flwr_supernode")
        elif name.startswith("superexec-") or name == "test-runner":
            lines.append("    <<: *flwr_superexec")

        body = yaml.safe_dump(
            service,
            sort_keys=False,
            default_flow_style=False,
        ).rstrip()

        if body:
            lines.extend(f"    {line}" for line in body.splitlines())

        lines.append("")

    lines.extend(["volumes:", "  data: {}", "  checkpoints: {}", ""])
    return "\n".join(lines)


def main() -> None:
    """Load clients.yml and generate docker-compose.generated.yml."""

    parser = argparse.ArgumentParser(
        description="Generate an N-client Flower 1.33.0 Docker Compose deployment."
    )
    parser.add_argument("--config", default="clients.yml", help="Path to clients.yml")
    parser.add_argument(
        "--output",
        default="docker-compose.generated.yml",
        help="Output Docker Compose file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)

    if not config_path.exists():
        raise FileNotFoundError(f"Client configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    clients = config.get("clients", [])
    compose = build_compose(clients)

    output_path.write_text(render_compose(compose), encoding="utf-8")
    print(f"Generated {output_path} for {len(clients)} clients.")


if __name__ == "__main__":
    main()
