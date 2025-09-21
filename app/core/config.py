from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

# Get the absolute path to the current file
current_file = Path(__file__).resolve()

# Move up three levels to reach project root (app)
project_dir = current_file.parents[1]

# Construct .env path
env_path = project_dir / ".env"


class Settings(BaseSettings):
    api_key: str

    database_url: str

    supported_files: List[str]

    is_docling_retriever: bool
    docling_page_seperator: str = "<!-- page break -->"
    docling_response_format: str = "markdown"

    custom_splitter: bool
    chunk_split_size: int
    chunk_split_overlap: int

    embedding_model_name: str
    embed_device: str

    allowed_vector_db: List[str] # Qdrant, Weaviate, Milvus

    qdrant_url: str
    qdrant_collection_name: str

    retrieval_chunk_limit: int

    openrouter_api_key: str

    # Define model registry
    llm_models: dict = {
        "deepseek": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "deepseek/deepseek-chat-v3.1:free",
        },
        "gpt4": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4o-mini",
        }
    }

    class Config:
        env_file = env_path
        case_sensitive = False
        extra = "allow"


settings = Settings()
