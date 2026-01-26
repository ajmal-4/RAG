from typing import AsyncGenerator, List

from langchain_ollama import ChatOllama

from app.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model: str):
        self.llm = ChatOllama(model=model)
        # self.llm = ChatOllama(model=model, stop=["<|im_start|>", "<|im_end|>"], top_p=1.0, temperature=0.0, top_k=1.0, num_predict=128, repeat_penalty=1.0)

    async def stream(self, messages: List) -> AsyncGenerator[str, None]:

        async for chunk in self.llm.astream(messages):
            yield chunk.content
    
    async def invoke(self, messages: List):

        response = await self.llm.ainvoke(messages)
        return response.content