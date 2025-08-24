from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    collection: str = Field(..., description="Collection of the vector db")
    total_chunks: int = Field(..., description="Total chunks ingested")
