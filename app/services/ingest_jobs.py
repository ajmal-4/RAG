import uuid
from typing import Dict

JOB_STORE: Dict[str, dict] = {}

def create_job(filename: str):
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": filename,
        "chunks": 0,
        "error": None,
    }
    return job_id

def update_job(job_id: str, **kwargs):
    JOB_STORE[job_id].update(kwargs)

def get_job(job_id: str):
    return JOB_STORE.get(job_id)
