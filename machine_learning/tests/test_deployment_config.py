"""Tests for the single secure distributed deployment configuration."""

from pathlib import Path

import pytest

from src.deployment_config import DeploymentConfigError, DeploymentProfile, load_deployment_config, validate_no_insecure_flag


def production_env(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "ca.crt"
    cert = tmp_path / "superlink.crt"
    key = tmp_path / "superlink.key"
    auth_dir = tmp_path / "auth"
    auth_host_dir = tmp_path / "auth-host"
    state_host_dir = tmp_path / "state-host"
    auth_dir.mkdir(); auth_host_dir.mkdir(); state_host_dir.mkdir()
    for path in (root, cert, key): path.write_text("test", encoding="utf-8")
    return {
        "DEPLOYMENT_PROFILE": "production",
        "SUPERLINK_HOST": "fl.example.internal",
        "TLS_ROOT_CERTIFICATES": str(root),
        "SUPERLINK_CERTIFICATE": str(cert),
        "SUPERLINK_PRIVATE_KEY": str(key),
        "TLS_CERTIFICATE_HOST_DIR": str(tmp_path),
        "SUPERNODE_AUTH_PRIVATE_KEY_DIR": str(auth_dir),
        "SUPERNODE_AUTH_HOST_DIR": str(auth_host_dir),
        "SUPERLINK_STATE_HOST_DIR": str(state_host_dir),
        "SUPERLINK_STATE_DIR": "/var/lib/flower",
    }


def test_production_is_the_only_profile() -> None:
    assert list(DeploymentProfile) == [DeploymentProfile.PRODUCTION]


def test_production_requires_explicit_environment_variables() -> None:
    with pytest.raises(DeploymentConfigError, match="required environment variables"):
        load_deployment_config({"DEPLOYMENT_PROFILE": "production"})


def test_production_configuration_loads(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path), require_files=True)
    assert config.profile is DeploymentProfile.PRODUCTION
    assert config.superlink_host == "fl.example.internal"
    assert config.superlink_address == "fl.example.internal:9092"
    assert config.superlink_control_address == "fl.example.internal:9093"
    assert config.tls_root_certificates == tmp_path / "ca.crt"
    assert config.superlink_certificate == tmp_path / "superlink.crt"
    assert config.superlink_private_key == tmp_path / "superlink.key"
    assert config.supernode_auth_enabled


def test_invalid_profile_is_rejected() -> None:
    with pytest.raises(DeploymentConfigError, match="DEPLOYMENT_PROFILE"):
        load_deployment_config({"DEPLOYMENT_PROFILE": "development", "SUPERLINK_HOST": "host"})


def test_invalid_role_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DeploymentConfigError, match="server, client"):
        load_deployment_config(production_env(tmp_path), role="all")


def test_production_requires_tls_auth_and_state_files_when_requested(tmp_path: Path) -> None:
    env = production_env(tmp_path)
    (tmp_path / "superlink.key").unlink()
    with pytest.raises(DeploymentConfigError, match="security files/directories"):
        load_deployment_config(env, require_files=True)


def test_production_rejects_insecure_flag() -> None:
    with pytest.raises(DeploymentConfigError, match="--insecure"):
        validate_no_insecure_flag(DeploymentProfile.PRODUCTION, ["--insecure", "--superlink", "fl.example.internal:9092"])


def test_production_allows_secure_command_without_insecure_flag() -> None:
    validate_no_insecure_flag(DeploymentProfile.PRODUCTION, ["--superlink", "fl.example.internal:9092", "--root-certificates", "/etc/flower/ca.crt"])


def test_production_generates_superlink_tls_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))
    assert config.superlink_tls_args() == [
        "--ssl-ca-certfile", str(tmp_path / "ca.crt"),
        "--ssl-certfile", str(tmp_path / "superlink.crt"),
        "--ssl-keyfile", str(tmp_path / "superlink.key"),
    ]


def test_production_generates_superlink_auth_args(tmp_path: Path) -> None:
    assert load_deployment_config(production_env(tmp_path)).superlink_auth_args() == ["--enable-supernode-auth"]


def test_production_generates_persistent_superlink_state_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))
    assert config.superlink_state_args() == ["--database", "/var/lib/flower/superlink.db"]


def test_production_generates_per_client_supernode_auth_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))
    assert config.supernode_auth_args("client-1") == ["--auth-supernode-private-key", str(tmp_path / "auth" / "client-1")]


def test_supernode_auth_requires_safe_client_id(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))
    with pytest.raises(DeploymentConfigError, match="client ID"):
        config.supernode_auth_args("../other-node")
    with pytest.raises(DeploymentConfigError, match="client ID"):
        config.supernode_auth_args(" ")


def test_supernode_auth_host_key_uses_safe_client_id(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))
    assert config.supernode_auth_host_key("client-1") == tmp_path / "auth-host" / "client-1"
    with pytest.raises(DeploymentConfigError, match="Invalid"):
        config.supernode_auth_host_key("../other-node")


def test_production_generates_supernode_tls_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))
    assert config.supernode_tls_args() == ["--root-certificates", str(tmp_path / "ca.crt")]


def test_cli_configuration_uses_root_certificates(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))
    assert config.cli_tls_config() == {"address": "fl.example.internal:9093", "root-certificates": str(tmp_path / "ca.crt")}
