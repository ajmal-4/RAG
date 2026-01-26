import asyncio
from app.services.web_service import WebService

async def test_web_service(query: str):
    service = WebService()

    response = await service.generate_tavily_response(query)
    print(response)
    return response

if __name__ == "__main__":
    query = "What is the rate of bitcoin today?"
    asyncio.run(test_web_service(query))