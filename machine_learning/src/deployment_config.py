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

    @property
    def is_production(self) -> bool:
        """Return whether this configuration is production mode."""

        return self.profile is DeploymentProfile.PRODUCTION


PROFILE_ENV = "DEPLOYMENT_PROFILE"
SUPERLINK_ADDRESS_ENV = "SUPERLINK_ADDRESS"
TLS_ROOT_CERTIFICATES_ENV = "TLS_ROOT_CERTIFICATES"
SUPERLINK_CERTIFICATE_ENV = "SUPERLINK_CERTIFICATE"
SUPERLINK_PRIVATE_KEY_ENV = "SUPERLINK_PRIVATE_KEY"

PRODUCTION_REQUIRED_ENV = (
    SUPERLINK_ADDRESS_ENV,
    TLS_ROOT_CERTIFICATES_ENV,
    SUPERLINK_CERTIFICATE_ENV,
    SUPERLINK_PRIVATE_KEY_ENV,
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
    """Load and validate deployment configuration from environment variables.

    Development mode is intentionally lightweight and remains compatible with
    the current local Docker workflow. Production mode fails closed unless the
    SuperLink address and TLS certificate/key paths are explicitly configured.
    """

    env = os.environ if environ is None else environ
    profile = _profile_from_value(env.get(PROFILE_ENV))
    superlink_address = env.get(SUPERLINK_ADDRESS_ENV, "").strip()

    if not superlink_address:
        if profile is DeploymentProfile.DEVELOPMENT:
            superlink_address = "superlink:9092"
        else:
            raise DeploymentConfigError(
                f"{SUPERLINK_ADDRESS_ENV} is required in production."
            )

    if profile is DeploymentProfile.DEVELOPMENT:
        return DeploymentConfig(
            profile=profile,
            superlink_address=superlink_address,
        )

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
    }

    if require_files:
        missing_files = [
            f"{name}={path}"
            for name, path in paths.items()
            if not path.is_file()
        ]
        if missing_files:
            raise DeploymentConfigError(
                "Production TLS files were not found: " + ", ".join(missing_files)
            )

    return DeploymentConfig(
        profile=profile,
        superlink_address=superlink_address,
        **paths,
    )


def validate_environment(
    environ: Mapping[str, str] | None = None,
    *,
    require_files: bool = False,
) -> DeploymentConfig:
    """Validate deployment environment and return the resolved configuration."""

    return load_deployment_config(environ, require_files=require_files)
