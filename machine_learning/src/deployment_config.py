"""Deployment profile and production security configuration validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


class DeploymentConfigError(ValueError):
    """Raised when deployment configuration is invalid or unsafe."""


class DeploymentProfile(str, Enum):
    """Supported deployment profiles."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


@dataclass(frozen=True)
class DeploymentConfig:
    """Resolved deployment configuration."""

    profile: DeploymentProfile
    superlink_address: str
    tls_root_certificates: Path | None = None
    superlink_certificate: Path | None = None
    superlink_private_key: Path | None = None
    supernode_auth_private_key_dir: Path | None = None

    @property
    def is_production(self) -> bool:
        """Return whether this configuration is production mode."""

        return self.profile is DeploymentProfile.PRODUCTION

    @property
    def supernode_auth_enabled(self) -> bool:
        """Return whether CLI-managed SuperNode authentication is enabled."""

        return self.is_production

    def superlink_tls_args(self) -> list[str]:
        """Return SuperLink TLS arguments for a production launch."""

        if not self.is_production:
            return []
        assert self.tls_root_certificates is not None
        assert self.superlink_certificate is not None
        assert self.superlink_private_key is not None
        return [
            "--ssl-ca-certfile",
            str(self.tls_root_certificates),
            "--ssl-certfile",
            str(self.superlink_certificate),
            "--ssl-keyfile",
            str(self.superlink_private_key),
        ]

    def superlink_auth_args(self) -> list[str]:
        """Return SuperLink SuperNode-authentication arguments."""

        if not self.supernode_auth_enabled:
            return []
        return ["--enable-supernode-auth"]

    def supernode_tls_args(self) -> list[str]:
        """Return SuperNode TLS arguments for a production launch."""

        if not self.is_production:
            return []
        assert self.tls_root_certificates is not None
        return ["--root-certificates", str(self.tls_root_certificates)]

    def supernode_auth_args(self, client_id: str) -> list[str]:
        """Return authentication arguments for a specific SuperNode."""

        if not self.supernode_auth_enabled:
            return []
        if not client_id or not client_id.strip():
            raise DeploymentConfigError(
                "A non-empty client ID is required for SuperNode authentication."
            )
        assert self.supernode_auth_private_key_dir is not None
        key_path = self.supernode_auth_private_key_dir / client_id.strip()
        return ["--auth-supernode-private-key", str(key_path)]

    def cli_tls_config(self) -> dict[str, str | bool]:
        """Return the Flower CLI federation configuration for this profile."""

        if self.is_production:
            assert self.tls_root_certificates is not None
            return {
                "address": self.superlink_address,
                "root-certificates": str(self.tls_root_certificates),
            }
        return {
            "address": self.superlink_address,
            "insecure": True,
        }


PROFILE_ENV = "DEPLOYMENT_PROFILE"
SUPERLINK_ADDRESS_ENV = "SUPERLINK_ADDRESS"
TLS_ROOT_CERTIFICATES_ENV = "TLS_ROOT_CERTIFICATES"
SUPERLINK_CERTIFICATE_ENV = "SUPERLINK_CERTIFICATE"
SUPERLINK_PRIVATE_KEY_ENV = "SUPERLINK_PRIVATE_KEY"
TLS_CERTIFICATE_HOST_DIR_ENV = "TLS_CERTIFICATE_HOST_DIR"
SUPERNODE_AUTH_PRIVATE_KEY_DIR_ENV = "SUPERNODE_AUTH_PRIVATE_KEY_DIR"

PRODUCTION_REQUIRED_ENV = (
    SUPERLINK_ADDRESS_ENV,
    TLS_ROOT_CERTIFICATES_ENV,
    SUPERLINK_CERTIFICATE_ENV,
    SUPERLINK_PRIVATE_KEY_ENV,
    SUPERNODE_AUTH_PRIVATE_KEY_DIR_ENV,
)


def _profile_from_value(value: str | None) -> DeploymentProfile:
    """Parse the deployment profile, defaulting to development."""

    normalized = (value or DeploymentProfile.DEVELOPMENT.value).strip().lower()

    try:
        return DeploymentProfile(normalized)
    except ValueError as exc:
        allowed = ", ".join(profile.value for profile in DeploymentProfile)
        raise DeploymentConfigError(
            f"{PROFILE_ENV} must be one of: {allowed}; got '{normalized}'."
        ) from exc


def validate_no_insecure_flag(
    profile: DeploymentProfile | str,
    command: Sequence[str],
) -> None:
    """Reject Flower's insecure transport flag in production commands."""

    resolved_profile = (
        profile
        if isinstance(profile, DeploymentProfile)
        else _profile_from_value(profile)
    )

    if resolved_profile is DeploymentProfile.PRODUCTION and "--insecure" in command:
        raise DeploymentConfigError(
            "Production deployment must not use Flower's --insecure flag. "
            "Configure TLS before starting the production federation."
        )


def load_deployment_config(
    environ: Mapping[str, str] | None = None,
    *,
    require_files: bool = False,
) -> DeploymentConfig:
    """Load and validate deployment configuration from environment variables."""

    env = os.environ if environ is None else environ
    profile = _profile_from_value(env.get(PROFILE_ENV))

    if profile is DeploymentProfile.DEVELOPMENT:
        superlink_address = env.get(SUPERLINK_ADDRESS_ENV, "").strip()
        if not superlink_address:
            superlink_address = "superlink:9092"
        return DeploymentConfig(profile=profile, superlink_address=superlink_address)

    missing = [name for name in PRODUCTION_REQUIRED_ENV if not env.get(name, "").strip()]
    if missing:
        raise DeploymentConfigError(
            "Production deployment is missing required environment variables: "
            + ", ".join(missing)
        )

    paths = {
        "tls_root_certificates": Path(env[TLS_ROOT_CERTIFICATES_ENV]),
        "superlink_certificate": Path(env[SUPERLINK_CERTIFICATE_ENV]),
        "superlink_private_key": Path(env[SUPERLINK_PRIVATE_KEY_ENV]),
        "supernode_auth_private_key_dir": Path(env[SUPERNODE_AUTH_PRIVATE_KEY_DIR_ENV]),
    }

    if require_files:
        missing_files = []
        if not paths["tls_root_certificates"].is_file():
            missing_files.append(
                f"tls_root_certificates={paths['tls_root_certificates']}"
            )
        if not paths["superlink_certificate"].is_file():
            missing_files.append(
                f"superlink_certificate={paths['superlink_certificate']}"
            )
        if not paths["superlink_private_key"].is_file():
            missing_files.append(
                f"superlink_private_key={paths['superlink_private_key']}"
            )
        if not paths["supernode_auth_private_key_dir"].is_dir():
            missing_files.append(
                "supernode_auth_private_key_dir="
                f"{paths['supernode_auth_private_key_dir']}"
            )
        if missing_files:
            raise DeploymentConfigError(
                "Production security files/directories were not found: "
                + ", ".join(missing_files)
            )

    return DeploymentConfig(
        profile=profile,
        superlink_address=env[SUPERLINK_ADDRESS_ENV].strip(),
        **paths,
    )


def validate_environment(
    environ: Mapping[str, str] | None = None,
    *,
    require_files: bool = False,
) -> DeploymentConfig:
    """Validate deployment environment and return the resolved configuration."""

    return load_deployment_config(environ, require_files=require_files)
