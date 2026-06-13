"""API routers."""

from .classify import router as classify_router
from .health import router as health_router

__all__ = ["classify_router", "health_router"]
