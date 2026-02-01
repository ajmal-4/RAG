from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.openai_client import OpenAILLM
from app.llm.ollama_client import OllamaLLM


# Store created clients (singleton-like per model)
_llm_instances: dict[str, BaseLLM] = {}

def get_llm(model_name: str = "deepseek") -> BaseLLM:
    """Return LLM client for given model from registry."""

    if model_name not in settings.llm_models:
        raise ValueError(f"Model {model_name} not found in registry")

    if model_name not in _llm_instances:
        config = settings.llm_models[model_name]

        if config["provider"] == "openrouter":
            client = ChatOpenAI(
                model=config["model"],
                openai_api_key=settings.openrouter_api_key,
                openai_api_base=config["base_url"],
                streaming=True
            )

            _llm_instances[model_name] = OpenAILLM(
                client=client,
                model=config["model"],
            )

        elif config["provider"] == "local":
            _llm_instances[model_name] = OllamaLLM(
                model=config["model"]
            )

        else:
            raise ValueError(f"Unknown provider: {config['provider']}")

    return _llm_instances[model_name]


def get_model_id(model_name: str = "deepseek") -> str:
    """Return actual model string (like deepseek/deepseek-chat-v3.1:free)."""
    if model_name not in settings.llm_models:
        raise ValueError(f"Model {model_name} not found in registry")
    return settings.llm_models[model_name]["model"]