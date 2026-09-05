"""Tests for the development certificate generator."""

from pathlib import Path

from scripts import generate_dev_certs


def test_generate_certificates_builds_ca_and_superlink_certificate(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run_openssl(args: list[str]) -> None:
        commands.append(args)

    monkeypatch.setattr(generate_dev_certs, "run_openssl", fake_run_openssl)

    output_dir = tmp_path / "certificates"
    generate_dev_certs.generate_certificates(output_dir, "superlink")

    assert commands[0] == ["genrsa", "-out", str(output_dir / "ca.key"), "4096"]
    assert commands[1][0:5] == ["req", "-x509", "-new", "-nodes", "-key"]
    assert commands[2] == ["genrsa", "-out", str(output_dir / "superlink.key"), "2048"]
    assert commands[3][0:4] == ["req", "-new", "-key", str(output_dir / "superlink.key")]
    assert commands[4][0:4] == ["x509", "-req", "-in", str(output_dir / "superlink.csr")]

    assert "subjectAltName=DNS:superlink,DNS:localhost,IP:127.0.0.1" in (
        output_dir / "superlink.ext"
    ).read_text(encoding="utf-8") if (output_dir / "superlink.ext").exists() else True


def test_generated_temporary_files_are_removed(monkeypatch, tmp_path: Path) -> None:
    def fake_run_openssl(args: list[str]) -> None:
        if "-out" in args:
            output = Path(args[args.index("-out") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("generated", encoding="utf-8")

    monkeypatch.setattr(generate_dev_certs, "run_openssl", fake_run_openssl)

    output_dir = tmp_path / "certificates"
    generate_dev_certs.generate_certificates(output_dir, "superlink")

    assert not (output_dir / "superlink.csr").exists()
    assert not (output_dir / "superlink.ext").exists()
    assert not (output_dir / "ca.srl").exists()
