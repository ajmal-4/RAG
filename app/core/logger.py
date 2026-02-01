from pathlib import Path
from datetime import datetime

from loguru import logger

from app.core.config import settings


LOG_DIR = Path(settings.log_path)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"

CUSTOM_FORMAT = (
    "<level>{level: <5}</level> | "
    "{extra[username]} | "
    "{extra[tenant]} | "
    "{module} | "
    "{function} | "
    "{message}"
)

# Set global default values so fields always exist
logger.configure(extra={"username": "N/A", "tenant": "N/A"})

logger.remove()
logger.add(
    LOG_FILE,
    format=CUSTOM_FORMAT,
    rotation=settings.log_rotation,         # New log file every day
    retention=settings.log_retention,      # Keep logs for 10 days
    compression=settings.log_compression,        # Compress old logs
    enqueue=settings.log_enqueue,             # Safe for multi-thread/multi-process
    serialize=settings.log_serialize,           # JSON format for structured logging
    level=settings.log_level,             # Default log level
)