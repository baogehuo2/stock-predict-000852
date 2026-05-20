from __future__ import annotations

import requests

from src.common.config import get_config
from src.common.network import disable_env_proxies
from src.common.secrets import get_secret


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self) -> None:
        disable_env_proxies()
        cfg = get_config()["llm"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.timeout = int(cfg.get("timeout_seconds", 90))
        self.temperature = float(cfg.get("temperature", 0))
        self.api_key = get_secret(cfg.get("api_key_secret_path", "llm.api_key"))
        if not self.api_key:
            raise LLMError("LLM API key is empty. Check encrypted secrets.")

    def chat(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        disable_env_proxies()
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code >= 400:
            raise LLMError(f"LLM request failed: {resp.status_code} {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LLMError(f"Unexpected LLM response: {data}") from exc
