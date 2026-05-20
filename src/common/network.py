from __future__ import annotations

import os


PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


def disable_env_proxies() -> None:
    """Ignore proxy variables because this project is configured for direct access."""
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")

