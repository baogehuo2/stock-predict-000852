from __future__ import annotations

import argparse
from pathlib import Path

from src.common.secrets import (
    DEFAULT_ENC_PATH,
    DEFAULT_KEY_PATH,
    load_plain_secrets,
    write_encrypted_secrets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 config/secrets.plain.yaml 加密写入 config/secrets.enc.yaml"
    )
    parser.add_argument(
        "--plain",
        type=Path,
        default=Path("config/secrets.plain.yaml"),
        help="明文 YAML 文件路径，默认 config/secrets.plain.yaml",
    )
    parser.add_argument(
        "--enc",
        type=Path,
        default=DEFAULT_ENC_PATH,
        help="输出的加密文件路径，默认 config/secrets.enc.yaml",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=DEFAULT_KEY_PATH,
        help="Fernet key 文件路径，默认 config/secrets.key",
    )
    args = parser.parse_args()
    data = load_plain_secrets(args.plain)
    output = write_encrypted_secrets(data, enc_path=args.enc, key_path=args.key)
    print(f"Encrypted secrets written to {output}")


if __name__ == "__main__":
    main()
