"""FastAPI gateway for Phase 1 (Design.md §2).

Exposes ``GET /health`` and ``POST /api/vqa``. The only model-facing code is
behind ``vqa_service.answer_question`` — nothing else here knows about the VLM.
"""

import logging
import os
import tempfile
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .geo_tools import GeoError
from . import mcp_client
from .schemas import VQAResponse
from .vqa_service import VQAServiceError, answer_index_question, answer_question

GEOTIFF_EXTENSIONS = {"tif", "tiff"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # dev: load HF_TOKEN / VQA_MODEL from .env

app = FastAPI(title="Remote Sensing VQA Agent", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


async def _answer_geotiff(data: bytes, question: str) -> str:
    """Compute NDVI/NDWI via the MCP subprocess and answer from measured values.

    Phase 3: index computation is routed through ``earth_agent.mcp_server``
    instead of calling ``geo_tools`` directly (Backlog Phase 3 manual test).

    Raises:
        HTTPException 400: if the raster can't be read or no index bands resolve.
        HTTPException 502: if the upstream model call fails.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            indices = await mcp_client.compute_indices(tmp_path)
        except GeoError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not indices:
            raise HTTPException(
                status_code=400,
                detail="This GeoTIFF's bands could not be identified for NDVI/NDWI "
                       "computations. Add band descriptions (e.g. 'Red', 'NIR') or "
                       "pass explicit band indices.",
            )
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    logger.info("Computed indices %s for GeoTIFF upload", indices)
    try:
        return answer_index_question(question, indices)
    except VQAServiceError as exc:
        logger.error("VQA model call failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/vqa", response_model=VQAResponse)
async def vqa(image: UploadFile = File(...), question: str = Form(...)) -> VQAResponse:
    """Answer ``question`` about the uploaded ``image``.

    JPEG/PNG uploads go to the VLM as an image; GeoTIFF uploads get NDVI/NDWI
    computed locally and the measured values fed to the model instead
    (Design.md §7, Phase 2 cross-check).

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

    ext = (image.filename or "").rsplit(".", 1)[-1].lower()
    if ext in GEOTIFF_EXTENSIONS:
        return VQAResponse(answer=await _answer_geotiff(data, question))

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