"""Tests for development SuperNode authentication identity generation."""

from pathlib import Path

import pytest

from scripts import generate_supernode_auth


def test_generate_credentials_creates_unique_key_pairs(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_ssh_keygen(args: list[str], *, capture_output: bool = False) -> str:
        commands.append(args)
        output = Path(args[args.index("-f") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("OPENSSH PRIVATE KEY", encoding="utf-8")
        Path(f"{output}.pub").write_text(
            "ecdsa-sha2-nistp384 AAAATEST\n", encoding="utf-8"
        )
        return ""

    monkeypatch.setattr(generate_supernode_auth, "run_ssh_keygen", fake_ssh_keygen)

    credentials = generate_supernode_auth.generate_credentials(
        tmp_path / "auth", ["client-1", "client-2"]
    )

    assert [private.name for private, _ in credentials] == ["client-1", "client-2"]
    assert [public.name for _, public in credentials] == ["client-1.pub", "client-2.pub"]
    assert len(commands) == 2
    assert all(command[:6] == ["-t", "ecdsa", "-b", "384", "-N", ""] for command in commands)
    assert all(command[-2] == "-f" for command in commands)
    assert all(
        public.read_text(encoding="utf-8").startswith("ecdsa-sha2-nistp384 ")
        for _, public in credentials
    )


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


def test_generate_credentials_rejects_existing_keypair(tmp_path: Path) -> None:
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (auth_dir / "client-1").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_supernode_auth.generate_credentials(auth_dir, ["client-1"])


def test_generate_credentials_sets_private_key_permissions(monkeypatch, tmp_path: Path) -> None:
    def fake_ssh_keygen(args: list[str], *, capture_output: bool = False) -> str:
        output = Path(args[args.index("-f") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("OPENSSH PRIVATE KEY", encoding="utf-8")
        Path(f"{output}.pub").write_text(
            "ecdsa-sha2-nistp384 AAAATEST\n", encoding="utf-8"
        )
        return ""

    monkeypatch.setattr(generate_supernode_auth, "run_ssh_keygen", fake_ssh_keygen)

    credentials = generate_supernode_auth.generate_credentials(tmp_path / "auth", ["client-1"])

    private_key, public_key = credentials[0]
    assert private_key.stat().st_mode & 0o777 == 0o600
    assert public_key.stat().st_mode & 0o777 == 0o644


def test_generate_credentials_uses_ssh_ecdsa_private_key(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_ssh_keygen(args: list[str], *, capture_output: bool = False) -> str:
        captured.append(args)
        output = Path(args[args.index("-f") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("OPENSSH PRIVATE KEY", encoding="utf-8")
        Path(f"{output}.pub").write_text(
            "ecdsa-sha2-nistp384 AAAATEST\n", encoding="utf-8"
        )
        return ""

    monkeypatch.setattr(generate_supernode_auth, "run_ssh_keygen", fake_ssh_keygen)

    private_key, public_key = generate_supernode_auth.generate_credentials(
        tmp_path / "auth", ["client-1"]
    )[0]

    assert captured == [[
        "-t", "ecdsa", "-b", "384", "-N", "", "-f", str(private_key)
    ]]
    assert private_key.read_text(encoding="utf-8") == "OPENSSH PRIVATE KEY"
    assert public_key.read_text(encoding="utf-8") == "ecdsa-sha2-nistp384 AAAATEST\n"
