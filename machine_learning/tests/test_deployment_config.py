"""Tests for deployment profile and production security configuration."""

from pathlib import Path

import pytest

from src.deployment_config import (
    DeploymentConfigError,
    DeploymentProfile,
    load_deployment_config,
    validate_no_insecure_flag,
)


def production_env(tmp_path: Path) -> dict[str, str]:
    """Return a complete production configuration using temporary security material."""

    root = tmp_path / "ca.crt"
    cert = tmp_path / "superlink.crt"
    key = tmp_path / "superlink.key"
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()

    for path in (root, cert, key):
        path.write_text("test", encoding="utf-8")

    return {
        "DEPLOYMENT_PROFILE": "production",
        "SUPERLINK_ADDRESS": "fl.example.internal:9092",
        "TLS_ROOT_CERTIFICATES": str(root),
        "SUPERLINK_CERTIFICATE": str(cert),
        "SUPERLINK_PRIVATE_KEY": str(key),
        "SUPERNODE_AUTH_PRIVATE_KEY_DIR": str(auth_dir),
    }


def test_development_profile_is_default() -> None:
    config = load_deployment_config({})

    assert config.profile is DeploymentProfile.DEVELOPMENT
    assert config.superlink_address == "superlink:9092"
    assert not config.is_production
    assert not config.supernode_auth_enabled


def test_development_profile_accepts_current_insecure_transport() -> None:
    validate_no_insecure_flag(
        DeploymentProfile.DEVELOPMENT,
        ["--insecure", "--superlink", "superlink:9092"],
    )


def test_invalid_profile_is_rejected() -> None:
    with pytest.raises(DeploymentConfigError, match="DEPLOYMENT_PROFILE"):
        load_deployment_config({"DEPLOYMENT_PROFILE": "staging"})


def test_production_requires_explicit_environment_variables() -> None:
    with pytest.raises(DeploymentConfigError, match="required environment variables"):
        load_deployment_config({"DEPLOYMENT_PROFILE": "production"})


def test_production_requires_tls_and_auth_directory_when_requested(tmp_path: Path) -> None:
    env = production_env(tmp_path)
    missing = tmp_path / "missing.key"
    env["SUPERLINK_PRIVATE_KEY"] = str(missing)

    with pytest.raises(DeploymentConfigError, match="security files/directories"):
        load_deployment_config(env, require_files=True)


def test_production_configuration_loads(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path), require_files=True)

    assert config.profile is DeploymentProfile.PRODUCTION
    assert config.superlink_address == "fl.example.internal:9092"
    assert config.tls_root_certificates is not None
    assert config.superlink_certificate is not None
    assert config.superlink_private_key is not None
    assert config.supernode_auth_private_key_dir == tmp_path / "auth"
    assert config.is_production
    assert config.supernode_auth_enabled


def test_production_rejects_insecure_flag() -> None:
    with pytest.raises(DeploymentConfigError, match="--insecure"):
        validate_no_insecure_flag(
            DeploymentProfile.PRODUCTION,
            ["--insecure", "--superlink", "fl.example.internal:9092"],
        )


def test_production_allows_secure_command_without_insecure_flag() -> None:
    validate_no_insecure_flag(
        DeploymentProfile.PRODUCTION,
        ["--superlink", "fl.example.internal:9092", "--root-certificates", "/etc/flower/ca.crt"],
    )


def test_production_generates_superlink_tls_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))

    assert config.superlink_tls_args() == [
        "--ssl-ca-certfile", str(tmp_path / "ca.crt"),
        "--ssl-certfile", str(tmp_path / "superlink.crt"),
        "--ssl-keyfile", str(tmp_path / "superlink.key"),
    ]


def test_production_enables_supernode_authentication() -> None:
    assert load_deployment_config({
        "DEPLOYMENT_PROFILE": "development",
    }).superlink_auth_args() == []


def test_production_generates_superlink_auth_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))

    assert config.superlink_auth_args() == ["--enable-supernode-auth"]


def test_production_generates_per_client_supernode_auth_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))

    assert config.supernode_auth_args("client-1") == [
        "--auth-supernode-private-key",
        str(tmp_path / "auth" / "client-1"),
    ]


def test_supernode_auth_requires_client_id(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))

    with pytest.raises(DeploymentConfigError, match="client ID"):
        config.supernode_auth_args(" ")


def test_production_generates_supernode_tls_args(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))

    assert config.supernode_tls_args() == [
        "--root-certificates", str(tmp_path / "ca.crt")
    ]


def test_development_generates_no_tls_or_auth_args() -> None:
    config = load_deployment_config({})

    assert config.superlink_tls_args() == []
    assert config.superlink_auth_args() == []
    assert config.supernode_tls_args() == []
    assert config.supernode_auth_args("client-1") == []
    assert config.cli_tls_config() == {
        "address": "superlink:9092",
        "insecure": True,
    }


def test_production_cli_configuration_uses_root_certificates(tmp_path: Path) -> None:
    config = load_deployment_config(production_env(tmp_path))

    assert config.cli_tls_config() == {
        "address": "fl.example.internal:9092",
        "root-certificates": str(tmp_path / "ca.crt"),
    }
