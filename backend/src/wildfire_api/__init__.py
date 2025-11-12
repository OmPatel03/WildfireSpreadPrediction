"""Shared package for the FastAPI + GraphQL wildfire service."""

from .config import get_settings
from .service import WildfireService

__all__ = ["get_settings", "WildfireService"]
