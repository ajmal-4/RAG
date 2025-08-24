from fastapi import APIRouter

from app.api.v1.endpoints.rag_endpoints import router as rag_router

api_router = APIRouter()

api_router.include_router(rag_router)