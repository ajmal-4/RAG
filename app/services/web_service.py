from typing import List, Dict, Any

from tavily import TavilyClient

from app.core.logger import logger
from app.core.config import settings


class WebService:

    def __init__(self):
        self.tavily_client = TavilyClient(settings.tavily_api_key) 

    async def generate_tavily_response(self, query: str):
        response = self.tavily_client.search(query, settings.tavily_search_depth)
        return response
    
    async def generate_web_search_results(self, query: str):

        if settings.web_search_agent == "tavily":
            results = await self.generate_tavily_response(query)
            results = results.get("results", [])
        else:
            pass

        return results
        