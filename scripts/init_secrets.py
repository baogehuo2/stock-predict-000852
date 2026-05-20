from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

from cryptography.fernet import Fernet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_KEY_PATH = CONFIG_DIR / "secrets.key"
DEFAULT_PLAIN_PATH = CONFIG_DIR / "secrets.plain.yaml"
DEFAULT_ENC_PATH = CONFIG_DIR / "secrets.enc.yaml"


def _write_private_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite it.")
    path.write_bytes(content)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows may not fully honor POSIX permissions; the file still stays local.
        pass


def _load_or_create_key(path: Path) -> bytes:
    if path.exists():
        key = path.read_bytes().strip()
        # Validate early so a bad key fails before touching plaintext.
        base64.urlsafe_b64decode(key)
        Fernet(key)
        return key

    key = Fernet.generate_key()
    _write_private_file(path, key + b"\n")
    return key


def encrypt_secrets(plain_path: Path, enc_path: Path, key_path: Path, force: bool = False) -> None:
    if not plain_path.exists():
        raise FileNotFoundError(
            f"Plain secrets file not found: {plain_path}\n"
            "Copy config/secrets.plain.example.yaml to config/secrets.plain.yaml first."
        )

    if enc_path.exists() and not force:
        raise FileExistsError(f"{enc_path} already exists. Re-run with --force to overwrite it.")

    key = _load_or_create_key(key_path)
    plaintext = plain_path.read_bytes()
    token = Fernet(key).encrypt(plaintext)

    enc_path.parent.mkdir(parents=True, exist_ok=True)
    enc_path.write_bytes(token + b"\n")
    try:
        os.chmod(enc_path, 0o600)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt local secrets for the zz1000 project.")
    parser.add_argument("--plain", type=Path, default=DEFAULT_PLAIN_PATH, help="Plain YAML secrets path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_ENC_PATH, help="Encrypted secrets output path.")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH, help="Local Fernet key path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing encrypted secrets file.")
    args = parser.parse_args()

    encrypt_secrets(args.plain.resolve(), args.out.resolve(), args.key.resolve(), force=args.force)

    print(f"Key file: {args.key.resolve()}")
    print(f"Encrypted secrets: {args.out.resolve()}")
    print(f"Next step: delete the plaintext file after checking the encrypted file exists: {args.plain.resolve()}")


if __name__ == "__main__":
    main()

