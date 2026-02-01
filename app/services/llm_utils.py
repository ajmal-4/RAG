import yaml
from pathlib import Path

from langchain.messages import HumanMessage, AIMessage

from app.core.config import settings


_PROMPT_CACHE = {}

def load_prompt(name: str) -> dict[str, str]:
    # if name in _PROMPT_CACHE:
    #     return _PROMPT_CACHE[name]

    path = Path(settings.prompt_path) / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {name}.yaml must be a dict")

    # _PROMPT_CACHE[name] = data
    return data

def process_chat_history(history: list[dict]) -> str:
    """
    Process chat history into a formatted string.
    Each entry in history is expected to be a dict with 'role' and 'content' keys.
    """

    # Truncate the history to the limit
    history = history[-4:]

    formatted_history = []
    for entry in history:
        formatted_history.append(HumanMessage(content=entry.get("question", "")))
        formatted_history.append(AIMessage(content=entry.get("response", "")))
    
    return formatted_history