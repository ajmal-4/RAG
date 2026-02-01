import json
from json_repair import repair_json
from typing import List, Dict, Any, AsyncGenerator

import numpy as np
from sklearn.cluster import KMeans
from langchain.messages import SystemMessage, HumanMessage

from app.core.logger import logger
from app.core.llm import get_llm, get_model_id
from app.services.llm_utils import load_prompt
from app.schemas.rag_api_schema import ChatRequest
from app.services.vector_db_agent import VectorDbAgent
from app.services.summarize_service import SummarizeService
from app.services.web_service import WebService


class AgenticService:

    def __init__(self):
        self.vector_db_agent = VectorDbAgent()
        self.summarize_service = SummarizeService()
        self.web_service = WebService()
        
        self.tool_registry = {
            "vector_searh": self.vector_db_agent.retrieve_from_qdrant,
            "summarize": self.summarize_service.summarize_with_kmeans_clustering,
            "web_search": self.web_service.generate_web_search_results,
            "chart_generation": ""
        }


    def parse_tool_calls(self, model_output: str):
        """
        Parses LLM tool-call output into a list of dicts.
        Repairs malformed JSON if needed.
        """

        tool_calls = []

        # Split by lines (each line is expected to be one tool call)
        lines = [line.strip() for line in model_output.splitlines() if line.strip()]

        for line in lines:
            try:
                # Try strict JSON first
                tool_calls.append(json.loads(line))
            except json.JSONDecodeError:
                # Repair and retry
                repaired = repair_json(line)
                tool_calls.append(json.loads(repaired))

        return tool_calls
    
    
    async def process_tool_calls(self, tools: List[Dict]):
        """
        Process each tool one by one and return the result.
        Expected tools format:
        [{"tool": "vector_search", "arguments": {"query":"invoice number id"}}]
        """
        results = []
        function_call_result = None
        
        for tool in tools:

            tool_name = tool["tool"]
            arguments = tool["arguments"]

            if tool_name in self.tool_registry:
               
                try:
                    function_call_result = await self.tool_registry[tool_name](**arguments)
                except Exception as e:
                    logger.error(f"Error executing tool call function : {str(e)}")

            else:
                logger.error(f"Unknown function call: {tool_name}")
                
            results.append({"tool": tool_name, "arguments": arguments, "tool_result": function_call_result})
        
        return results

    
    async def execute_plan(self, planner_response):

        # Parse the response into desired format
        if isinstance(planner_response, str):
            tool_calls = self.parse_tool_calls(planner_response)
        else:
            pass

        # Process the tool calls one by one   
        tool_call_result = await self.process_tool_calls(tool_calls)

        return tool_call_result
        

    
    