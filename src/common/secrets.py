from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from cryptography.fernet import Fernet, InvalidToken


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_KEY_PATH = CONFIG_DIR / "secrets.key"
DEFAULT_ENC_PATH = CONFIG_DIR / "secrets.enc.yaml"


class SecretsError(RuntimeError):
    """Raised when encrypted secrets cannot be loaded."""


def encrypt_secrets(
    data: dict[str, Any],
    key: bytes | str,
) -> bytes:
    """Encrypt a YAML mapping with a Fernet key and return the token bytes."""
    if not isinstance(data, dict):
        raise SecretsError("Secrets to encrypt must be a YAML mapping.")
    if isinstance(key, str):
        key = key.encode("utf-8")
    plaintext = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=False,
    ).encode("utf-8")
    return Fernet(key).encrypt(plaintext)


def load_secrets(
    enc_path: str | Path = DEFAULT_ENC_PATH,
    key_path: str | Path = DEFAULT_KEY_PATH,
) -> dict[str, Any]:
    enc_file = Path(enc_path)
    key_file = Path(key_path)

    if not key_file.exists():
        raise SecretsError(f"Secret key file not found: {key_file}")
    if not enc_file.exists():
        raise SecretsError(f"Encrypted secrets file not found: {enc_file}")

    try:
        key = key_file.read_bytes().strip()
        token = enc_file.read_bytes().strip()
        plaintext = Fernet(key).decrypt(token)
    except InvalidToken as exc:
        raise SecretsError("Failed to decrypt secrets. Check that secrets.key matches secrets.enc.yaml.") from exc
    except Exception as exc:
        raise SecretsError(f"Failed to load encrypted secrets: {exc}") from exc

    data = yaml.safe_load(plaintext.decode("utf-8")) or {}
    if not isinstance(data, dict):
        raise SecretsError("Decrypted secrets must be a YAML mapping.")
    return data


def load_plain_secrets(path: str | Path) -> dict[str, Any]:
    plain_file = Path(path)
    if not plain_file.exists():
        raise SecretsError(f"Plain secrets file not found: {plain_file}")
    data = yaml.safe_load(plain_file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SecretsError("Plain secrets must be a YAML mapping.")
    return data


def write_encrypted_secrets(
    data: dict[str, Any],
    enc_path: str | Path = DEFAULT_ENC_PATH,
    key_path: str | Path = DEFAULT_KEY_PATH,
) -> Path:
    enc_file = Path(enc_path)
    key_file = Path(key_path)
    if not key_file.exists():
        raise SecretsError(f"Secret key file not found: {key_file}")
    token = encrypt_secrets(data, key_file.read_bytes().strip())
    enc_file.write_bytes(token)
    return enc_file


def get_secret(path: str, default: Any = None) -> Any:
    data: Any = load_secrets()
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    return data

