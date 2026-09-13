"""FastAPI gateway for Phase 1 (Design.md §2).

Exposes ``GET /health`` and ``POST /api/vqa``. The only model-facing code is
behind ``vqa_service.answer_question`` — nothing else here knows about the VLM.
"""

import logging
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .schemas import VQAResponse
from .vqa_service import VQAServiceError, answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # dev: load HF_TOKEN / VQA_MODEL from .env

app = FastAPI(title="Remote Sensing VQA Agent", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/api/vqa", response_model=VQAResponse)
async def vqa(image: UploadFile = File(...), question: str = Form(...)) -> VQAResponse:
    """Answer ``question`` about the uploaded ``image``.

    Raises:
        HTTPException 400: if ``question`` is blank or ``image`` is missing/corrupt.
        HTTPException 502: if the upstream model call fails.
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image file was empty.")

    try:
        pil_image = Image.open(BytesIO(data))
        pil_image.load()  # force decode so corrupt files fail here, not at the model
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="image is not a valid image file.")

    try:
        answer = answer_question(pil_image, question)
    except VQAServiceError as exc:
        logger.error("VQA model call failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return VQAResponse(answer=answer)