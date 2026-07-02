from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryptography.fernet import Fernet

from src.common.secrets import encrypt_secrets, load_secrets, write_encrypted_secrets


class SecretsToolTests(unittest.TestCase):
    def test_encrypt_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            key_file = root / "secrets.key"
            enc_file = root / "secrets.enc.yaml"
            key = Fernet.generate_key()
            key_file.write_bytes(key)

            payload = {"tushare": {"token": "ts-demo-token"}}
            token = encrypt_secrets(payload, key)
            enc_file.write_bytes(token)

            loaded = load_secrets(enc_path=enc_file, key_path=key_file)
            self.assertEqual(loaded["tushare"]["token"], "ts-demo-token")

    def test_write_encrypted_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            key_file = root / "secrets.key"
            enc_file = root / "secrets.enc.yaml"
            key = Fernet.generate_key()
            key_file.write_bytes(key)

            output = write_encrypted_secrets(
                {"database": {"password": "pw"}},
                enc_path=enc_file,
                key_path=key_file,
            )
            self.assertEqual(output, enc_file)
            self.assertTrue(enc_file.exists())


if __name__ == "__main__":
    unittest.main()
