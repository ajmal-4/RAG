from typing import Protocol, AsyncGenerator, List

class BaseLLM(Protocol):
    async def stream(self, messsages: List) -> AsyncGenerator[str, None]:
        ...
    
    async def invoke(self, messsages: List):
        ...