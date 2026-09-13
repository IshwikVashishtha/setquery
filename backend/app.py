"""FastAPI gateway: thin web layer over the orchestration graph (Phase 4).

Validates the request, hands it to the LangGraph state machine, and maps the
graph's result/error back to HTTP. All answering logic lives in
``backend/orchestration.py``; this module has no model or geospatial code.
"""

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .orchestration import GRAPH
from .schemas import VQAResponse

load_dotenv()  # dev: load HF_TOKEN / VQA_MODEL from .env

app = FastAPI(title="Remote Sensing VQA Agent", version="0.1.0")

GEOTIFF_EXTENSIONS = {"tif", "tiff"}


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/api/vqa", response_model=VQAResponse)
async def vqa(image: UploadFile = File(...), question: str = Form(...)) -> VQAResponse:
    """Answer ``question`` about the uploaded ``image``.

    Dispatches through the orchestration graph, which routes by request shape.
    JPEG/PNG go to the VLM as an image; GeoTIFFs get NDVI/NDWI computed via the
    MCP subprocess and the measured values fed to the model instead.

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