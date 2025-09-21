from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

# Singleton pattern
_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embed_device},
        )
    return _embeddings_instance
