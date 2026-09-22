"""FastAPI gateway: thin web layer over the orchestration graph (Phase 4).

Validates the request, hands it to the LangGraph state machine (or, for the
Phase 6 grounding route, to the grounding pipeline), and maps errors back to
HTTP. All answering logic lives in ``backend/orchestration.py`` and
``backend/grounding.py``; this module has no model or geospatial code.
"""

import base64
import io
import logging
import os
import tempfile
from typing import Annotated

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from . import analyzer, grounding
from .grounding import GroundingError
from .orchestration import GRAPH
from .schemas import AnalyzeResponse, GroundResponse, VQAResponse
from .vqa_service import VQAServiceError
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # dev: load HF_TOKEN / VQA_MODEL from .env


app = FastAPI(title="Remote Sensing VQA Agent", version="0.1.0")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEOTIFF_EXTENSIONS = {"tif", "tiff"}


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/api/preview")
async def preview(image: UploadFile = File(...)) -> Response:
    """Render a viewable RGB PNG preview for any image, especially multi-band GeoTIFFs."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image file was empty.")

    ext = (image.filename or "").rsplit(".", 1)[-1].lower()
    if ext in GEOTIFF_EXTENSIONS:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            preview_img, _, _ = grounding.render_rgb_preview(tmp_path)
            buf = io.BytesIO()
            preview_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as exc:
            logger.exception("GeoTIFF preview generation failed")
            raise HTTPException(status_code=400, detail=f"GeoTIFF preview failed: {exc}") from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception:
        return Response(content=data, media_type=image.content_type or "image/jpeg")


@app.post("/api/vqa", response_model=VQAResponse)
async def vqa(image: UploadFile = File(...), question: str = Form(...)) -> VQAResponse:
    """Answer ``question`` about the uploaded ``image``.

    Dispatches through the orchestration graph, which routes by request shape.
    JPEG/PNG go to the VLM as an image; GeoTIFFs are band-stretched to an RGB
    preview and answered directly. For the tool-augmented path (every upload
    run through the relevant MCP tools before answering) use ``/api/analyze``.

    Raises:
        HTTPException 400: if ``question`` is blank or ``image`` is missing/corrupt.
        HTTPException 501: if the request routes to an unimplemented path.
        HTTPException 502: if the upstream model call fails.
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image file was empty.")

    ext = (image.filename or "").rsplit(".", 1)[-1].lower()
    file_kind = "geotiff" if ext in GEOTIFF_EXTENSIONS else "raster"

    result = await GRAPH.ainvoke(
        {
            "image_bytes": data,
            "question": question,
            "file_kind": file_kind,
            "n_images": 1,
        }
    )

    if result.get("error"):
        raise HTTPException(status_code=501, detail=result["error"])

    return VQAResponse(answer=result["answer"])


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...), question: str = Form(...)
) -> AnalyzeResponse:
    """Tool-augmented analysis of the uploaded image (any format).

    The upload is run through the top-relevant MCP tools (selected via the
    ChromaDB tool index from the user's question), then the tool outputs plus
    a viewable form of the image go to the VLM for the final answer. JPEG/PNG
    uploads are decoded by PIL; GeoTIFFs are band-stretched to an RGB preview.

    Raises:
        HTTPException 400: ``question`` blank, empty upload, or undecodable file.
        HTTPException 502: if the upstream model call fails.
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image file was empty.")

    try:
        result = await analyzer.analyze(question, data, image.filename or "image")
    except VQAServiceError as exc:
        logger.exception("Tool-augmented analysis failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnalyzeResponse(**result)


@app.post("/api/change", response_model=VQAResponse)
async def change(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    question: str = Form(...),
    date1: str | None = Form(None),
    date2: str | None = Form(None),
) -> VQAResponse:
    """Describe the change between two co-registered images (Phase 5).

    Dispatches through the graph's bi_temporal branch: both images plus optional
    capture dates are sent to the VLM in a single message.

    Raises:
        HTTPException 400: if ``question`` is blank or either image is missing.
        HTTPException 502: if the upstream model call fails.
    """
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    data_before = await image1.read()
    data_after = await image2.read()
    if not data_before or not data_after:
        raise HTTPException(status_code=400, detail="both images are required.")

    result = await GRAPH.ainvoke(
        {
            "image_bytes": data_before,
            "image2_bytes": data_after,
            "question": question,
            "n_images": 2,
            "date_before": date1 or None,
            "date_after": date2 or None,
        }
    )

    if result.get("error"):
        raise HTTPException(status_code=501, detail=result["error"])

    return VQAResponse(answer=result["answer"])


@app.post("/api/ground", response_model=GroundResponse)
async def ground(
    image: UploadFile = File(...),
    query: str = Form(...),
) -> GroundResponse:
    """Ground every instance of ``query`` in ``image`` and return boxes + overlay.

    Phase 6 (Backlog.md): the hosted VLM localizes the requested object and
    returns JSON boxes; for GeoTIFF uploads the pixel boxes are mapped to
    real-world coordinates and a self-contained Folium/Leaflet map embeds the
    image + red box rectangles at their correct locations.

    Raises:
        HTTPException 400: empty ``query``, unreadable image, or no boxes found.
        HTTPException 502: if the upstream model call fails.
    """
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image file was empty.")

    ext = (image.filename or "").rsplit(".", 1)[-1].lower()
    is_geotiff = ext in GEOTIFF_EXTENSIONS

    # GeoTIFF needs to be on disk for rasterio reads/geo-mapping later.
    tmp_path = None
    try:
        if is_geotiff:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

        try:
            result = grounding.ground_image(data, query, is_geotiff, tmp_path)
        except GroundingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VQAServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        if tmp_path:
            import os

            os.unlink(tmp_path)

    return GroundResponse(**result)