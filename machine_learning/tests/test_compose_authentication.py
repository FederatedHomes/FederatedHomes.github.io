"""Tests for per-SuperNode authentication configuration in Compose."""

from pathlib import Path

from scripts.generate_compose import build_compose


def production_env(tmp_path: Path, monkeypatch) -> None:
    tls_dir = tmp_path / "tls"
    auth_dir = tmp_path / "auth"
    tls_dir.mkdir()
    auth_dir.mkdir()
    for name in ("ca.crt", "superlink.crt", "superlink.key"):
        (tls_dir / name).write_text("test", encoding="utf-8")
    for client_id in ("client-1", "client-2"):
        (auth_dir / client_id).write_text("private", encoding="utf-8")
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "production")
    monkeypatch.setenv("SUPERLINK_ADDRESS", "fl.example.internal:9092")
    monkeypatch.setenv("TLS_ROOT_CERTIFICATES", "/etc/flower/tls/ca.crt")
    monkeypatch.setenv("SUPERLINK_CERTIFICATE", "/etc/flower/tls/superlink.crt")
    monkeypatch.setenv("SUPERLINK_PRIVATE_KEY", "/etc/flower/tls/superlink.key")
    monkeypatch.setenv("TLS_CERTIFICATE_HOST_DIR", str(tls_dir))
    monkeypatch.setenv("SUPERNODE_AUTH_PRIVATE_KEY_DIR", "/etc/flower/auth")
    monkeypatch.setenv("SUPERNODE_AUTH_HOST_DIR", str(auth_dir))


def clients() -> list[dict]:
    return [
        {"id": "client-1", "data_dir": "./data/client-1", "checkpoint_dir": "./checkpoints/client-1"},
        {"id": "client-2", "data_dir": "./data/client-2", "checkpoint_dir": "./checkpoints/client-2"},
    ]


def test_production_superlink_enables_supernode_authentication(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    command = compose["services"]["superlink"]["command"]
    assert "--enable-supernode-auth" in command
    assert "--insecure" not in command


def test_each_supernode_gets_only_its_own_auth_key_and_ca(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    for client_id in ("client-1", "client-2"):
        service = compose["services"][f"supernode-{client_id}"]
        assert service["command"][-2:] == [
            "--auth-supernode-private-key", f"/etc/flower/auth/{client_id}"
        ]
        assert "--insecure" not in service["command"]
        assert service["volumes"] == [
            f"{tmp_path / 'tls'}/ca.crt:/etc/flower/tls/ca.crt:ro",
            f"{tmp_path / 'auth'}/{client_id}:/etc/flower/auth/{client_id}:ro",
        ]


def test_supernodes_cannot_receive_superlink_private_key(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    for client_id in ("client-1", "client-2"):
        volumes = compose["services"][f"supernode-{client_id}"]["volumes"]
        assert not any("superlink.key" in volume for volume in volumes)
        assert not any("superlink.crt" in volume for volume in volumes)
