from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db


def require_api_key(x_api_key: str = Header(None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
def get_database() -> Session:
    return get_db