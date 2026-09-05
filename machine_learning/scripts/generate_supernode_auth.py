#!/usr/bin/env python3
"""Generate development-only EC identities for Flower SuperNodes.

Production key material must be generated and managed by the deployment's
approved PKI/secret-management process. This helper exists only to make local
multi-node authentication testing reproducible.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_openssl(args: list[str]) -> None:
    """Run OpenSSL and fail clearly when it is unavailable."""

    if shutil.which("openssl") is None:
        raise RuntimeError("OpenSSL is required to generate SuperNode credentials.")
    subprocess.run(["openssl", *args], check=True)


def validate_client_id(client_id: str) -> str:
    """Validate a client ID before using it as a filesystem component."""

    value = client_id.strip()
    if not value or value in {".", ".."}:
        raise ValueError("Client IDs must be non-empty and cannot be '.' or '..'.")
    if Path(value).name != value:
        raise ValueError(f"Unsafe client ID '{client_id}'.")
    return value


def generate_supernode_keypair(output_dir: Path, client_id: str) -> tuple[Path, Path]:
    """Generate one EC private/public key pair for a SuperNode."""

    client_id = validate_client_id(client_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    private_key = output_dir / client_id
    public_key = output_dir / f"{client_id}.pub"

    run_openssl([
        "ecparam",
        "-name",
        "secp384r1",
        "-genkey",
        "-noout",
        "-out",
        str(private_key),
    ])
    run_openssl([
        "ec",
        "-in",
        str(private_key),
        "-pubout",
        "-out",
        str(public_key),
    ])

    # Private authentication keys should never be group/world readable.
    os.chmod(private_key, 0o600)
    os.chmod(public_key, 0o644)
    return private_key, public_key


def generate_credentials(output_dir: Path, client_ids: list[str]) -> list[tuple[Path, Path]]:
    """Generate one unique EC key pair for every configured client."""

    normalized_ids = [validate_client_id(client_id) for client_id in client_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Client IDs must be unique when generating authentication keys.")
    if not normalized_ids:
        raise ValueError("At least one client ID is required.")

    return [generate_supernode_keypair(output_dir, client_id) for client_id in normalized_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate development-only EC identities for Flower SuperNodes."
    )
    parser.add_argument(
        "client_ids",
        nargs="+",
        help="Configured client IDs, for example client-1 client-2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("certificates/dev/auth"),
        help="Directory for generated authentication keys",
    )
    args = parser.parse_args()

    credentials = generate_credentials(args.output_dir, args.client_ids)
    print(f"Generated {len(credentials)} SuperNode key pairs in {args.output_dir}.")
    print("Development credentials only; do not commit private keys or use them as production identities.")


if __name__ == "__main__":
    main()
