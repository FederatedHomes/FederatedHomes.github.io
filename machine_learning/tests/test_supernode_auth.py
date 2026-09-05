"""Tests for development SuperNode authentication identity generation."""

from pathlib import Path

import pytest

from scripts import generate_supernode_auth


def test_generate_credentials_creates_unique_key_pairs(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run_openssl(args: list[str]) -> None:
        commands.append(args)
        if "-out" in args:
            output = Path(args[args.index("-out") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("generated", encoding="utf-8")

    monkeypatch.setattr(generate_supernode_auth, "run_openssl", fake_run_openssl)

    credentials = generate_supernode_auth.generate_credentials(
        tmp_path / "auth", ["client-1", "client-2"]
    )

    assert [private.name for private, _ in credentials] == ["client-1", "client-2"]
    assert [public.name for _, public in credentials] == ["client-1.pub", "client-2.pub"]
    assert len(commands) == 4
    assert all("secp384r1" in command for command in commands[::2])
    assert all("-pubout" in command for command in commands[1::2])


def test_generate_credentials_rejects_duplicate_client_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        generate_supernode_auth.generate_credentials(
            tmp_path / "auth", ["client-1", "client-1"]
        )


def test_generate_credentials_rejects_empty_client_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        generate_supernode_auth.generate_credentials(tmp_path / "auth", [])


def test_generate_credentials_rejects_unsafe_client_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        generate_supernode_auth.generate_credentials(tmp_path / "auth", ["../client-1"])


def test_generate_credentials_sets_private_key_permissions(monkeypatch, tmp_path: Path) -> None:
    def fake_run_openssl(args: list[str]) -> None:
        if "-out" in args:
            output = Path(args[args.index("-out") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("generated", encoding="utf-8")

    monkeypatch.setattr(generate_supernode_auth, "run_openssl", fake_run_openssl)
    credentials = generate_supernode_auth.generate_credentials(tmp_path / "auth", ["client-1"])

    private_key, public_key = credentials[0]
    assert private_key.stat().st_mode & 0o777 == 0o600
    assert public_key.stat().st_mode & 0o777 == 0o644
