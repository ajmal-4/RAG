from openai import AsyncOpenAI
from app.core.config import settings

# Store created clients (singleton-like per model)
_llm_instances = {}

def get_llm(model_name: str = "deepseek") -> AsyncOpenAI:
    """Return LLM client for given model from registry."""
    if model_name not in settings.llm_models:
        raise ValueError(f"Model {model_name} not found in registry")

    if model_name not in _llm_instances:
        config = settings.llm_models[model_name]

        if config["provider"] == "openrouter":
            _llm_instances[model_name] = AsyncOpenAI(
                base_url=config["base_url"],
                api_key=settings.openrouter_api_key,
            )
        else:
            raise ValueError(f"Unknown provider: {config['provider']}")

    return _llm_instances[model_name]


def get_model_id(model_name: str = "deepseek") -> str:
    """Return actual model string (like deepseek/deepseek-chat-v3.1:free)."""
    if model_name not in settings.llm_models:
        raise ValueError(f"Model {model_name} not found in registry")
    return settings.llm_models[model_name]["model"]