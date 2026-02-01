from typing import Optional, Any, List, Dict

from pydantic import BaseModel, Field


# Requests
class ChatRequest(BaseModel):
    question: str = Field(..., description="Question to ask the LLM")
    history: Optional[List[Dict[str, Any]]] = Field(default=[], description="Chat history")
    model_name: str = Field(
        "deepseek", description="Model name from registry to use for response"
    )

class SummaryRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the collection")
    filters: Optional[dict] = Field(default={}, description="Must and must not filters")
    method: str = Field(default="kmeans", description="Technique for summarization")
    model_name: str = Field(
        "deepseek", description="Model name from registry to use for response"
    )

class WebSearchRequest(BaseModel):
    question: str = Field(..., description="Question for web search")
    model_name: str = Field(
        "deepseek", description="Model name from registry to use for response"
    )

class ChartGenerationRequest(BaseModel):
    question: str = Field(..., description="User question for chart generation")
    data: Any = Field(default=None, description="Data used for chart generation")
    model_name: str = Field(
        "deepseek", description="Model name from registry to use for chart generation"
    )

class IngestResponse(BaseModel):
    collection: str = Field(..., description="Collection of the vector db")
    job_id: str = Field(..., description="Ingestion job ID")
    ingestion_status: str = Field(..., description="Ingestion status")
    ingestion_message: str = Field(..., description="Ingestion message")
