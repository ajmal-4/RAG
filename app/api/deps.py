from fastapi import Depends, Header, HTTPException

from app.core.config import settings


def require_api_key(x_api_key: str = Header(None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")