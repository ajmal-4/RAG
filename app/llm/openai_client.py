from typing import AsyncGenerator, List

from openai import AsyncOpenAI

from app.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def stream(self, messages: List) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
