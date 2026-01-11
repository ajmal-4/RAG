from typing import Optional

from pydantic import BaseModel, Field


# Requests
class ChatRequest(BaseModel):
    question: str = Field(..., description="Question to ask the LLM")
    model_name: str = Field(
        "deepseek", description="Model name from registry to use for response"
    )

class SummaryRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the collection")
    filters: Optional[dict] = Field(default={}, description="Must and must not filters")

class IngestResponse(BaseModel):
    collection: str = Field(..., description="Collection of the vector db")
    job_id: str = Field(..., description="Ingestion job ID")
    ingestion_status: str = Field(..., description="Ingestion status")
    ingestion_message: str = Field(..., description="Ingestion message")
