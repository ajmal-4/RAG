import os
import tempfile
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.deps import require_api_key
from app.core.config import settings
from app.schemas.rag_api_schema import IngestResponse
from app.services.extraction_service import ExtractionService

router = APIRouter()

extraction_service = ExtractionService()


@router.post(
    "/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)]
)
async def ingest(
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

    ingested_chunks = extraction_service.extract_chunk_upsert_document(
        file_path=tmp_path, file_metadata=file_metadata
    )

    return IngestResponse(collection=collection, total_chunks=ingested_chunks)
