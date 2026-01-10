import asyncio

from app.services.llm_service import LLMService
from app.schemas.rag_api_schema import ChatRequest

async def test_chat_completion():
    service = LLMService()
    
    async for response in service.generate_response(
        ChatRequest(
            question="Hello, how are you?",
            model_name="qwen"
        )
    ):
        print("Response chunk: ", response)

if __name__ == "__main__":
    asyncio.run(test_chat_completion())