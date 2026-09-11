"""Secure distributed deployment configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


class DeploymentConfigError(ValueError):
    """Raised when deployment configuration is invalid or unsafe."""


class DeploymentProfile(str, Enum):
    """The single supported Flower deployment profile."""

    PRODUCTION = "production"


SUPERLINK_FLEET_PORT = 9092
SUPERLINK_CONTROL_PORT = 9093
SUPERLINK_RUNTIME_PORT = 9091
DEPLOYMENT_PROFILE_VALUE = DeploymentProfile.PRODUCTION.value


@dataclass(frozen=True)
class DeploymentConfig:
    """Resolved secure distributed deployment configuration."""

    profile: DeploymentProfile
    superlink_host: str
    superlink_address: str
    superlink_control_address: str
    tls_root_certificates: Path
    superlink_certificate: Path
    superlink_private_key: Path
    tls_certificate_host_dir: Path
    supernode_auth_private_key_dir: Path
    supernode_auth_host_dir: Path
    superlink_state_host_dir: Path
    superlink_state_dir: Path

    @property
    def is_production(self) -> bool:
        return True

    @property
    def supernode_auth_enabled(self) -> bool:
        return True

    def superlink_tls_args(self) -> list[str]:
        return [
            "--ssl-ca-certfile", str(self.tls_root_certificates),
            "--ssl-certfile", str(self.superlink_certificate),
            "--ssl-keyfile", str(self.superlink_private_key),
        ]

    def superlink_auth_args(self) -> list[str]:
        return ["--enable-supernode-auth"]

    def superlink_state_args(self) -> list[str]:
        return ["--database", str(self.superlink_state_dir / "superlink.db")]

    def supernode_tls_args(self) -> list[str]:
        return ["--root-certificates", str(self.tls_root_certificates)]

    def supernode_auth_args(self, client_id: str) -> list[str]:
        client_id = client_id.strip() if client_id else ""
        if not client_id or not _SAFE_CLIENT_ID.fullmatch(client_id):
            raise DeploymentConfigError(
                "SuperNode authentication requires a client ID containing only letters, numbers, '.', '_' or '-'."
            )
        return ["--auth-supernode-private-key", str(self.supernode_auth_private_key_dir / client_id)]

    def supernode_auth_host_key(self, client_id: str) -> Path:
        client_id = client_id.strip() if client_id else ""
        if not client_id or not _SAFE_CLIENT_ID.fullmatch(client_id):
            raise DeploymentConfigError("Invalid SuperNode client ID.")
        return self.supernode_auth_host_dir / client_id

    def cli_tls_config(self) -> dict[str, str]:
        return {"address": self.superlink_control_address, "root-certificates": str(self.tls_root_certificates)}


PROFILE_ENV = "DEPLOYMENT_PROFILE"
SUPERLINK_HOST_ENV = "SUPERLINK_HOST"
TLS_ROOT_CERTIFICATES_ENV = "TLS_ROOT_CERTIFICATES"
SUPERLINK_CERTIFICATE_ENV = "SUPERLINK_CERTIFICATE"
SUPERLINK_PRIVATE_KEY_ENV = "SUPERLINK_PRIVATE_KEY"
TLS_CERTIFICATE_HOST_DIR_ENV = "TLS_CERTIFICATE_HOST_DIR"
SUPERNODE_AUTH_PRIVATE_KEY_DIR_ENV = "SUPERNODE_AUTH_PRIVATE_KEY_DIR"
SUPERNODE_AUTH_HOST_DIR_ENV = "SUPERNODE_AUTH_HOST_DIR"
SUPERLINK_STATE_HOST_DIR_ENV = "SUPERLINK_STATE_HOST_DIR"
SUPERLINK_STATE_DIR_ENV = "SUPERLINK_STATE_DIR"
DEPLOYMENT_ROLE_ENV = "DEPLOYMENT_ROLE"

SERVER_REQUIRED_ENV = (
    SUPERLINK_HOST_ENV, TLS_ROOT_CERTIFICATES_ENV, SUPERLINK_CERTIFICATE_ENV,
    SUPERLINK_PRIVATE_KEY_ENV, TLS_CERTIFICATE_HOST_DIR_ENV,
    SUPERNODE_AUTH_PRIVATE_KEY_DIR_ENV, SUPERNODE_AUTH_HOST_DIR_ENV,
    SUPERLINK_STATE_HOST_DIR_ENV, SUPERLINK_STATE_DIR_ENV,
)
CLIENT_REQUIRED_ENV = (
    SUPERLINK_HOST_ENV, TLS_ROOT_CERTIFICATES_ENV, TLS_CERTIFICATE_HOST_DIR_ENV,
    SUPERNODE_AUTH_PRIVATE_KEY_DIR_ENV, SUPERNODE_AUTH_HOST_DIR_ENV,
)
_SAFE_CLIENT_ID = re.compile(r"[A-Za-z0-9._-]+")
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$|^[0-9a-fA-F:]+$")


def _profile_from_value(value: str | None) -> DeploymentProfile:
    normalized = (value or DEPLOYMENT_PROFILE_VALUE).strip().lower()
    try:
        return DeploymentProfile(normalized)
    except ValueError as exc:
        raise DeploymentConfigError(
            f"{PROFILE_ENV} must be '{DEPLOYMENT_PROFILE_VALUE}'; got '{normalized}'."
        ) from exc


def _normalize_host(value: str) -> str:
    host = value.strip()
    if not host:
        raise DeploymentConfigError(f"{SUPERLINK_HOST_ENV} must be set to the SuperLink hostname or IP address.")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not _HOST_RE.fullmatch(host):
        raise DeploymentConfigError(f"{SUPERLINK_HOST_ENV} contains an invalid hostname or IP address: '{host}'.")
    return host


def _endpoint(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"


def load_deployment_config(
    environ: Mapping[str, str] | None = None,
    *,
    require_files: bool = False,
    role: str | None = None,
) -> DeploymentConfig:
    env = os.environ if environ is None else environ
    resolved_role = (role or env.get(DEPLOYMENT_ROLE_ENV, "server")).strip().lower()
    if resolved_role not in {"server", "client"}:
        raise DeploymentConfigError("Deployment role must be one of: server, client.")
    profile = _profile_from_value(env.get(PROFILE_ENV))
    required_env = CLIENT_REQUIRED_ENV if resolved_role == "client" else SERVER_REQUIRED_ENV
    missing = [name for name in required_env if not env.get(name, "").strip()]
    if missing:
        raise DeploymentConfigError("Production deployment is missing required environment variables: " + ", ".join(missing))

    host = _normalize_host(env[SUPERLINK_HOST_ENV])
    paths = {
        "tls_root_certificates": Path(env[TLS_ROOT_CERTIFICATES_ENV]),
        "superlink_certificate": Path(env[SUPERLINK_CERTIFICATE_ENV]),
        "superlink_private_key": Path(env[SUPERLINK_PRIVATE_KEY_ENV]),
        "tls_certificate_host_dir": Path(env[TLS_CERTIFICATE_HOST_DIR_ENV]),
        "supernode_auth_private_key_dir": Path(env[SUPERNODE_AUTH_PRIVATE_KEY_DIR_ENV]),
        "supernode_auth_host_dir": Path(env[SUPERNODE_AUTH_HOST_DIR_ENV]),
        "superlink_state_host_dir": Path(env[SUPERLINK_STATE_HOST_DIR_ENV]),
        "superlink_state_dir": Path(env[SUPERLINK_STATE_DIR_ENV]),
    }
    if require_files:
        missing_files = []
        host_tls_dir = paths["tls_certificate_host_dir"]
        for name in ("ca.crt", "superlink.crt", "superlink.key"):
            if resolved_role == "client" and name != "ca.crt":
                continue
            if not (host_tls_dir / name).is_file():
                missing_files.append(f"tls_certificate_host_dir/{name}={host_tls_dir / name}")
        auth_host_dir = paths["supernode_auth_host_dir"]
        if not auth_host_dir.is_dir():
            missing_files.append(f"supernode_auth_host_dir={auth_host_dir}")
        if resolved_role == "server" and not paths["superlink_state_host_dir"].is_dir():
            missing_files.append(f"superlink_state_host_dir={paths['superlink_state_host_dir']}")
        if missing_files:
            raise DeploymentConfigError("Production security files/directories were not found: " + ", ".join(missing_files))

    return DeploymentConfig(
        profile=profile,
        superlink_host=host,
        superlink_address=_endpoint(host, SUPERLINK_FLEET_PORT),
        superlink_control_address=_endpoint(host, SUPERLINK_CONTROL_PORT),
        **paths,
    )


def validate_no_insecure_flag(profile: DeploymentProfile | str, command: Sequence[str]) -> None:
    _profile_from_value(profile.value if isinstance(profile, DeploymentProfile) else profile)
    if "--insecure" in command:
        raise DeploymentConfigError("Secure distributed deployment must not use Flower's --insecure flag.")


def validate_environment(environ: Mapping[str, str] | None = None, *, require_files: bool = False, role: str | None = None) -> DeploymentConfig:
    return load_deployment_config(environ, require_files=require_files, role=role)
