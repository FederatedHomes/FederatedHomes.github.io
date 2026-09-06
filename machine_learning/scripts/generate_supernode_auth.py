#!/usr/bin/env python3
"""Generate development-only EC identities for Flower SuperNodes.

Production key material must be generated and managed by the deployment's
approved PKI/secret-management process. This helper exists only to make local
multi-node authentication testing reproducible.

Flower CLI registration and SuperNode authentication both use an SSH-format
ECDSA key pair. P-384 is used to match the Flower authentication requirements.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


_SAFE_CLIENT_ID = re.compile(r"[A-Za-z0-9._-]+")


def run_ssh_keygen(args: list[str], *, capture_output: bool = False) -> str:
    """Run ssh-keygen and fail clearly when it is unavailable."""

    if shutil.which("ssh-keygen") is None:
        raise RuntimeError("ssh-keygen is required to generate SuperNode credentials.")
    result = subprocess.run(
        ["ssh-keygen", *args],
        check=True,
        capture_output=capture_output,
        text=True,
    )
    return result.stdout if capture_output else ""


def validate_client_id(client_id: str) -> str:
    """Validate a client ID before using it as a filesystem component."""

    value = client_id.strip()
    if not value or value in {".", ".."}:
        raise ValueError("Client IDs must be non-empty and cannot be '.' or '..'.")
    if not _SAFE_CLIENT_ID.fullmatch(value):
        raise ValueError(
            f"Unsafe client ID '{client_id}'. Client IDs may contain only "
            "letters, numbers, '.', '_' or '-'."
        )
    return value


def generate_supernode_keypair(output_dir: Path, client_id: str) -> tuple[Path, Path]:
    """Generate one SSH-format ECDSA P-384 key pair for a SuperNode."""

    client_id = validate_client_id(client_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    private_key = output_dir / client_id
    public_key = output_dir / f"{client_id}.pub"

    if private_key.exists() or public_key.exists():
        raise FileExistsError(
            f"Authentication material already exists for '{client_id}'. "
            "Remove the existing pair before regenerating it."
        )

    # Flower's SuperNode authentication expects an elliptic-curve private key
    # in SSH format. `ssh-keygen` also emits the matching OpenSSH public key.
    run_ssh_keygen([
        "-t",
        "ecdsa",
        "-b",
        "384",
        "-N",
        "",
        "-f",
        str(private_key),
    ])

    generated_public_key = Path(f"{private_key}.pub")
    if not generated_public_key.exists():
        raise RuntimeError(f"ssh-keygen did not create the public key: {generated_public_key}")
    generated_public_key.replace(public_key)

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
