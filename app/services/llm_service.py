import json
from typing import List, Dict, Any, AsyncGenerator

from langchain.messages import SystemMessage, HumanMessage, AIMessage

from app.core.config import settings
from app.core.llm import get_llm
from app.services.llm_utils import load_prompt, process_chat_history
from app.services.agentic_service import AgenticService
from app.services.summarize_service import SummarizeService
from app.services.web_service import WebService
from app.services.chart_service import ChartAgent
from app.schemas.rag_api_schema import (
    ChatRequest, 
    SummaryRequest,
    WebSearchRequest,
    ChartGenerationRequest
)


class LLMService:
    def __init__(self):
        self.agentic_service = AgenticService()
        self.summarize_service = SummarizeService()
        self.web_service = WebService()
        self.chart_agent = ChartAgent()

    async def generate_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        model_name = request.model_name or "qwen"
        llm_client = get_llm(model_name)

        system_prompt = load_prompt("simple_response")["SYSTEM"]
        if request.history:
            history_messages = process_chat_history(request.history)
        messages = [
            SystemMessage(content=system_prompt),
            *history_messages,
            HumanMessage(content=request.question)
        ]

        async for token in llm_client.stream(messages):
            yield token

    async def generate_agentic_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        model_name = request.model_name or "qwen"
        llm_client = get_llm(model_name)

        system_prompt = load_prompt("agentic_response")["SYSTEM"]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.question)
        ]

        planner_client = get_llm(settings.planner_model)
        planner_response = await planner_client.invoke(messages)
        executed_response = await self.agentic_service.execute_plan(planner_response)

        # Prepare messages for final response
        final_messages = [
            SystemMessage(content=load_prompt("agentic_response")["FINAL_RESPONSE"]),
            HumanMessage(content=request.question),
            AIMessage(content=json.dumps(executed_response))
        ]

        # Stream final response
        async for token in llm_client.stream(final_messages):
            yield token
    
    async def summarize(self, request: SummaryRequest):
        model_name = request.model_name or "qwen"
        llm_client = get_llm(model_name)

        if request.method == "kmeans":
            points = await self.summarize_service.summarize_with_kmeans_clustering(
                collection_name=request.collection_name,
                filters=request.filters,
                n_clusters=settings.n_clusters,
                top_k=settings.top_k,
                return_vectors=False
            )

            cleaned_points = [
                point["payload"] for point in points
            ]

            system_prompt = load_prompt("summarize_response")["SYSTEM"]
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(cleaned_points))
            ]

            async for token in llm_client.stream(messages):
                yield token
        
        else:
            pass
    
    async def web_Search(self, request: WebSearchRequest):
        model_name = request.model_name or "qwen"
        llm_client = get_llm(model_name)

        results = await self.web_service.generate_web_search_results(request.question)
        
        system_prompt = load_prompt("web_search_response")["SYSTEM"]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(results))
        ]

        async for token in llm_client.stream(messages):
            yield token
    
    async def generate_chart(self, request: ChartGenerationRequest):

        model_name = request.model_name or "qwen"
        llm_client = get_llm(model_name)

        system_prompt = load_prompt("chart_generation")["SYSTEM"]
        user_prompt = load_prompt("chart_generation")["USER"].format(
            user_question=request.question,
            user_data=request.data
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        python_code = await llm_client.invoke(messages)

        self.chart_agent.execute_python_chart_code(python_code=python_code)

        return python_code