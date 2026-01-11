import re
import yaml
from pathlib import Path

from app.core.config import settings


_PROMPT_CACHE = {}

def load_prompt(name: str) -> dict[str, str]:
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]

    path = Path(settings.prompt_path) / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {name}.yaml must be a dict")

    _PROMPT_CACHE[name] = data
    return data