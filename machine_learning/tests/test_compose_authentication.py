"""Tests for production SuperNode authentication and persistent SuperLink state."""

from pathlib import Path

from scripts.generate_compose import build_compose, render_compose


def production_env(tmp_path: Path, monkeypatch) -> None:
    tls_dir = tmp_path / "tls"
    auth_dir = tmp_path / "auth"
    state_dir = Path("state/superlink")
    tls_dir.mkdir()
    auth_dir.mkdir()
    for name in ("ca.crt", "superlink.crt", "superlink.key"):
        (tls_dir / name).write_text("test", encoding="utf-8")
    for client_id in ("client-1", "client-2"):
        # The same host-side auth directory is used uniformly. On a server
        # host it contains client public keys; on a client host it contains
        # that client's public/private key material.
        (auth_dir / client_id).write_text("public", encoding="utf-8")
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "production")
    monkeypatch.setenv("SUPERLINK_HOST", "fl.example.internal")
    monkeypatch.setenv("TLS_ROOT_CERTIFICATES", "/etc/flower/tls/ca.crt")
    monkeypatch.setenv("SUPERLINK_CERTIFICATE", "/etc/flower/tls/superlink.crt")
    monkeypatch.setenv("SUPERLINK_PRIVATE_KEY", "/etc/flower/tls/superlink.key")
    monkeypatch.setenv("TLS_CERTIFICATE_HOST_DIR", str(tls_dir))
    monkeypatch.setenv("SUPERNODE_AUTH_PRIVATE_KEY_DIR", "/etc/flower/auth")
    monkeypatch.setenv("SUPERNODE_AUTH_HOST_DIR", str(auth_dir))
    monkeypatch.setenv("SUPERLINK_STATE_HOST_DIR", str(state_dir))
    monkeypatch.setenv("SUPERLINK_STATE_DIR", "/var/lib/flower")


def clients() -> list[dict]:
    return [
        {"id": "client-1", "data_dir": "./data/client-1", "checkpoint_dir": "./checkpoints/client-1"},
        {"id": "client-2", "data_dir": "./data/client-2", "checkpoint_dir": "./checkpoints/client-2"},
    ]


def test_superlink_addresses_are_derived_from_one_host(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    command = compose["services"]["supernode-client-1"]["command"]
    index = command.index("--superlink")
    assert command[index + 1] == "fl.example.internal:9092"


def test_production_superlink_enables_supernode_authentication(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    command = compose["services"]["superlink"]["command"]
    assert "--enable-supernode-auth" in command
    assert "--database" in command
    assert command[command.index("--database") + 1] == "/var/lib/flower/superlink.db"
    assert "--insecure" not in command


def test_production_superlink_mounts_persistent_state(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    volumes = compose["services"]["superlink"]["volumes"]
    assert "./state/superlink:/var/lib/flower:rw" in volumes


def test_render_compose_preserves_relative_state_mount(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    rendered = render_compose(compose)
    assert "- ./state/superlink:/var/lib/flower:rw" in rendered
    assert "- state/superlink:/var/lib/flower:rw" not in rendered


def test_each_supernode_gets_uniform_auth_directory_and_ca_mount(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    for client_id in ("client-1", "client-2"):
        service = compose["services"][f"supernode-{client_id}"]
        assert service["command"][-2:] == [
            "--auth-supernode-private-key", f"/etc/flower/auth/{client_id}"
        ]
        assert "--insecure" not in service["command"]
        # Client hosts use the same auth directory mount convention as the
        # server host. The directory contains the individual client's key
        # material on the client host.
        assert service["volumes"] == [
            f"{tmp_path / 'tls'}/ca.crt:/etc/flower/tls/ca.crt:ro",
            f"{tmp_path / 'auth'}:/etc/flower/auth:ro",
        ]


def test_supernodes_cannot_receive_superlink_private_key(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production")
    for client_id in ("client-1", "client-2"):
        volumes = compose["services"][f"supernode-{client_id}"]["volumes"]
        assert not any("superlink.key" in volume for volume in volumes)
        assert not any("superlink.crt" in volume for volume in volumes)


def test_server_role_contains_server_infrastructure_and_federation_services(monkeypatch, tmp_path: Path) -> None:
    production_env(tmp_path, monkeypatch)
    compose = build_compose(clients(), profile="production", role="server")
    assert set(compose["services"]) == {
        "superlink",
        "superexec-serverapp",
        "trainer",
        "client-registration",
    }
    assert "test-runner" not in compose["services"]
    assert not any(name.startswith("supernode-") for name in compose["services"])
    assert not any(name.startswith("superexec-clientapp-") for name in compose["services"])


def test_all_role_retains_trainer_for_local_development(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "development")
    monkeypatch.delenv("SUPERLINK_HOST", raising=False)
    compose = build_compose(clients(), profile="development", role="all")
    assert "trainer" in compose["services"]
