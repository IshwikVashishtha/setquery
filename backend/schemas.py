"""Pydantic v2 models for the VQA API."""

from pydantic import BaseModel


class VQAResponse(BaseModel):
    """Response body returned by ``POST /api/vqa``."""

    answer: str