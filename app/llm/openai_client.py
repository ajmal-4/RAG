from typing import AsyncGenerator, List

from langchain_openai import ChatOpenAI

from app.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, client: ChatOpenAI, model: str):
        self.client = client
        self.model = model

    async def stream(self, messages: List) -> AsyncGenerator[str, None]:

        async for chunk in self.client.astream(messages):
            yield chunk.content
    
    async def invoke(self, messages: List):

        response = await self.client.ainvoke(messages)
        return response
