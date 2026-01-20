"""API module for ml-service.

Usage:
    uvicorn ml_service.api:app --host 0.0.0.0 --port 8000
"""

from ml_service.api.inference import app

__all__ = ["app"]
