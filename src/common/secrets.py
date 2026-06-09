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


def load_secrets(
    enc_path: str | Path = DEFAULT_ENC_PATH,
    key_path: str | Path = DEFAULT_KEY_PATH,
) -> dict[str, Any]:
    enc_file = Path(enc_path)
    key_file = Path(key_path)

    if not key_file.exists():
        raise SecretsError(
            f"Secret key file not found: {key_file}\n"
            "This project cannot read MySQL/API passwords without the matching local key. "
            "Copy config/secrets.key from the original workspace, or recreate config/secrets.plain.yaml "
            "and run: python .\\scripts\\init_secrets.py --force"
        )
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


def get_secret(path: str, default: Any = None) -> Any:
    data: Any = load_secrets()
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    return data

