"""Tests for secure Docker Compose generation."""

from pathlib import Path

import pytest

from scripts.generate_compose import build_compose
from src.deployment_config import DeploymentConfigError, DeploymentProfile


CLIENTS = [
    {"id": "client1", "public_key": "./certificates/prod/auth/client1.pub"},
    {"id": "client2", "public_key": "./certificates/prod/auth/client2.pub"},
]


def configure_production_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ca = tmp_path / "ca.crt"
    cert = tmp_path / "superlink.crt"
    key = tmp_path / "superlink.key"
    auth_host_dir = tmp_path / "auth-host"
    state_host_dir = tmp_path / "state" / "superlink"
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    auth_host_dir.mkdir()
    state_host_dir.mkdir(parents=True)
    data_dir.mkdir()
    checkpoint_dir.mkdir()
    for path in (ca, cert, key):
        path.write_text("test", encoding="utf-8")
    for client in CLIENTS:
        (auth_host_dir / client["id"]).write_text("test", encoding="utf-8")

    monkeypatch.setenv("DEPLOYMENT_PROFILE", "production")
    monkeypatch.setenv("SUPERLINK_HOST", "fl.example.internal")
    monkeypatch.setenv("TLS_ROOT_CERTIFICATES", "/etc/flower/tls/ca.crt")
    monkeypatch.setenv("SUPERLINK_CERTIFICATE", "/etc/flower/tls/superlink.crt")
    monkeypatch.setenv("SUPERLINK_PRIVATE_KEY", "/etc/flower/tls/superlink.key")
    monkeypatch.setenv("TLS_CERTIFICATE_HOST_DIR", str(tmp_path))
    monkeypatch.setenv("SUPERNODE_AUTH_PRIVATE_KEY_DIR", "/etc/flower/auth")
    monkeypatch.setenv("SUPERNODE_AUTH_HOST_DIR", str(auth_host_dir))
    monkeypatch.setenv("SUPERLINK_STATE_HOST_DIR", str(state_host_dir))
    monkeypatch.setenv("SUPERLINK_STATE_DIR", "/var/lib/flower")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("CHECKPOINT_DIR", str(checkpoint_dir))


def test_production_profile_is_the_only_profile() -> None:
    assert list(DeploymentProfile) == [DeploymentProfile.PRODUCTION]


def test_production_compose_requires_explicit_tls_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DEPLOYMENT_PROFILE", "SUPERLINK_HOST", "TLS_ROOT_CERTIFICATES",
        "SUPERLINK_CERTIFICATE", "SUPERLINK_PRIVATE_KEY", "TLS_CERTIFICATE_HOST_DIR",
        "SUPERNODE_AUTH_PRIVATE_KEY_DIR", "SUPERNODE_AUTH_HOST_DIR",
        "SUPERLINK_STATE_HOST_DIR", "SUPERLINK_STATE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "production")
    with pytest.raises(DeploymentConfigError, match="required environment variables"):
        build_compose(CLIENTS, profile=DeploymentProfile.PRODUCTION)


def test_invalid_profile_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_production_environment(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Only the production deployment profile"):
        build_compose(CLIENTS, profile="development")


def test_production_server_compose_uses_tls_auth_and_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_production_environment(monkeypatch, tmp_path)
    compose = build_compose(CLIENTS, profile=DeploymentProfile.PRODUCTION, role="server")
    superlink_command = compose["services"]["superlink"]["command"]
    assert "--insecure" not in superlink_command
    assert "--ssl-ca-certfile" in superlink_command
    assert "--ssl-certfile" in superlink_command
    assert "--ssl-keyfile" in superlink_command
    assert "--enable-supernode-auth" in superlink_command
    assert "--database" in superlink_command
    assert superlink_command[superlink_command.index("--database") + 1] == "/var/lib/flower/superlink.db"
    assert set(compose["services"]) == {"superlink", "superexec-serverapp", "trainer", "client-registration"}
    assert compose["services"]["trainer"]["command"][2] == "production-deployment"


def test_production_client_compose_contains_only_one_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_production_environment(monkeypatch, tmp_path)
    compose = build_compose(CLIENTS, profile=DeploymentProfile.PRODUCTION, role="client", client_id="client1")
    assert set(compose["services"]) == {"supernode-client1", "superexec-clientapp-client1"}
    node_command = compose["services"]["supernode-client1"]["command"]
    assert "--insecure" not in node_command
    assert "--root-certificates" in node_command
    assert "--auth-supernode-private-key" in node_command
    assert node_command[node_command.index("--superlink") + 1] == "fl.example.internal:9092"
    assert compose["services"]["supernode-client1"]["volumes"] == [
        f"{tmp_path}/ca.crt:/etc/flower/tls/ca.crt:ro",
        f"{tmp_path / 'auth-host'}:/etc/flower/auth:ro",
    ]
    client_app = compose["services"]["superexec-clientapp-client1"]
    assert client_app["volumes"] == [
        f"{tmp_path / 'data'}:/app/data:ro",
        f"{tmp_path / 'checkpoints'}:/app/checkpoints:rw",
    ]


def test_production_supernodes_use_configured_external_superlink_address(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_production_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("SUPERLINK_HOST", "192.168.1.100")
    compose = build_compose(CLIENTS, profile=DeploymentProfile.PRODUCTION, role="client", client_id="client1")
    command = compose["services"]["supernode-client1"]["command"]
    assert command[command.index("--superlink") + 1] == "192.168.1.100:9092"
