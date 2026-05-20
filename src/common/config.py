from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(Path, parts))


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = project_path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {file_path}")
    return data


@lru_cache(maxsize=16)
def get_config(name: str = "config.yaml") -> dict[str, Any]:
    return load_yaml(project_path("config", name))


def ensure_project_dirs() -> None:
    cfg = get_config()
    for value in cfg.get("paths", {}).values():
        project_path(value).mkdir(parents=True, exist_ok=True)

