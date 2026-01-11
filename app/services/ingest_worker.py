from app.services.extraction_service import ExtractionService
from app.services.ingest_jobs import update_job

extraction_service = ExtractionService()

def process_ingest_job(job_id: str, file_path: str, file_metadata: dict):
    try:
        update_job(job_id, status="processing")

        chunks = extraction_service.extract_chunk_upsert_document(
            file_path=file_path,
            file_metadata=file_metadata,
        )

        update_job(job_id, status="completed", chunks=chunks)

    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
        raise
