import os
import tempfile
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks

from app.api.deps import require_api_key
from app.core.config import settings
from app.schemas.rag_api_schema import IngestResponse, ChatRequest
from app.services.extraction_service import ExtractionService
from app.services.llm_service import LLMService
from app.services.ingest_jobs import create_job
from app.services.ingest_worker import process_ingest_job

router = APIRouter()

extraction_service = ExtractionService()
llm_service = LLMService()


@router.post(
    "/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)]
)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Query(settings.qdrant_collection_name),
):
    suffix = Path(file.filename or "upload.pdf").suffix.lower()
    if suffix not in settings.supported_files:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_metadata = {"file_name": file.filename}

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    content = await file.read()
    async with aiofiles.open(tmp_path, "wb") as out_file:
        await out_file.write(content)

    # ✅ Create job
    job_id = create_job(file.filename)

    background_tasks.add_task(
        process_ingest_job,
        job_id,
        tmp_path,
        file_metadata,
    )
    # ingested_chunks = extraction_service.extract_chunk_upsert_document(
    #     file_path=tmp_path, file_metadata=file_metadata
    # )

    return IngestResponse(
        collection=collection,
        job_id=job_id,
        ingestion_status="queued",
        ingestion_message="Ingestion Queued in background"
    )

@router.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        llm_service.generate_response(request), 
        media_type="text/plain"
    )

@router.post("/kmeans-summary")
async def kmeans_summary():
    result = await llm_service.summarize_with_kmeans_clustering(
        collection_name=settings.qdrant_collection_name,
        filters=None,
        n_clusters=2,
        top_k=1,
        return_vectors=False
    )

    return result