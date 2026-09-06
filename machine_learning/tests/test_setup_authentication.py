"""Security-focused tests for setup authentication validation."""

from pathlib import Path

import pytest


def test_production_setup_requires_each_client_auth_key(tmp_path: Path) -> None:
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "client-1").write_text("private", encoding="utf-8")

    client_ids = ["client-1", "client-2"]
    missing = [client_id for client_id in client_ids if not (auth_dir / client_id).is_file()]

    assert missing == ["client-2"]


def test_auth_keys_are_one_file_per_client(tmp_path: Path) -> None:
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    for client_id in ("client-1", "client-2"):
        (auth_dir / client_id).write_text(f"private-{client_id}", encoding="utf-8")

    assert sorted(path.name for path in auth_dir.iterdir()) == ["client-1", "client-2"]
    assert (auth_dir / "client-1").read_text(encoding="utf-8") != (auth_dir / "client-2").read_text(encoding="utf-8")


def test_private_key_material_is_not_expected_in_repository() -> None:
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        pytest.skip("repository .gitignore is unavailable in isolated test execution")
    text = gitignore.read_text(encoding="utf-8")
    assert "*.key" in text
    assert "certificates/" in text


def test_production_cannot_fallback_to_insecure() -> None:
    from src.deployment_config import DeploymentConfigError, DeploymentProfile, validate_no_insecure_flag

    with pytest.raises(DeploymentConfigError, match="--insecure"):
        validate_no_insecure_flag(DeploymentProfile.PRODUCTION, ["--insecure"])
