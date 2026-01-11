from typing import AsyncGenerator, List

from langchain_ollama import ChatOllama

from app.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model: str):
        self.llm = ChatOllama(model=model)

    async def stream(self, messages: List) -> AsyncGenerator[str, None]:

        async for chunk in self.llm.astream(messages):
            yield chunk.content