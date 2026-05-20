from __future__ import annotations

from src.common.config import load_yaml, project_path


def render_prompt(name: str, **kwargs: str) -> str:
    templates = load_yaml(project_path("config", "prompt_templates.yaml"))
    return templates[name].format(**kwargs)

