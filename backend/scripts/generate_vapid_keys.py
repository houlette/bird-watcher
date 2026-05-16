"""One-shot: generate the VAPID key pair used by Web Push.

Run once per deployment. The private key is written to a PEM file (gitignored);
the public key is printed as a base64-url string for the frontend to use as
PushManager.subscribe({ applicationServerKey, ... }).

Usage:

    docker compose run --rm api python scripts/generate_vapid_keys.py

Output:
    secrets/vapid_private.pem        (private — keep this safe)
    Public key printed to stdout    (paste into .env as VAPID_PUBLIC_KEY,
                                     or fetch via /api/push/vapid_public_key)

Web Push spec: https://datatracker.ietf.org/doc/html/rfc8292
"""
from __future__ import annotations

import base64
import logging
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vapid")

OUT_DIR = Path(__file__).resolve().parent.parent / "secrets"
PRIVATE_PEM_PATH = OUT_DIR / "vapid_private.pem"


def main() -> int:
    if PRIVATE_PEM_PATH.exists():
        log.warning("Private key already exists at %s — refusing to overwrite", PRIVATE_PEM_PATH)
        log.warning("If you really want a fresh key (which invalidates every existing")
        log.warning("push subscription), delete the file first and re-run this script.")
        return 1

    private_key = ec.generate_private_key(ec.SECP256R1())

    # Save private key as PKCS8 PEM, unencrypted (it's already in a secrets/
    # directory that's gitignored; encryption-at-rest is the VM's job).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    PRIVATE_PEM_PATH.write_bytes(pem_bytes)
    PRIVATE_PEM_PATH.chmod(0o600)
    log.info("Wrote private key to %s (mode 600)", PRIVATE_PEM_PATH)

    # Public key as the 65-byte uncompressed P-256 point, base64-url encoded.
    # That's the exact form PushManager.subscribe expects for applicationServerKey.
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    public_b64url = base64.urlsafe_b64encode(public_bytes).rstrip(b"=").decode("ascii")

    log.info("=" * 60)
    log.info("VAPID public key (paste into .env as VAPID_PUBLIC_KEY):")
    log.info("%s", public_b64url)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
