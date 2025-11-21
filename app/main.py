from fastapi import FastAPI

from app.core.middleware import setup_middlewares
from .api.v1.api import api_router
from .core.config import settings

app = FastAPI(
    title="RAG Application",
    description="RAG Application with various strategies",
    version="1.0.0",
)

# CORS middleware
setup_middlewares(app)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "RAG Application is Up & Running!"}
