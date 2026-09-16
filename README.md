# Remote Sensing VQA Agent

Upload satellite or aerial imagery, ask a question in plain English, and get a
text answer. Powered by a hosted vision-language model (VLM) served through
Hugging Face Inference Providers.

Phases 1–6 are implemented:

- **Tool-augmented analysis** (`POST /api/analyze`) — every upload, **any
  format**, is run through the MCP tools most relevant to the question: a
  ChromaDB index of ~104 tool descriptions (name + description, embedded once
  locally with all-MiniLM-L6-v2 and reused across restarts) retrieves the top-8,
  the callable subset is invoked over MCP, and the tool measurements plus a
  viewable form of the image go to the VLM for the final answer. GeoTIFFs are
  band-stretched to an RGB preview before being sent to the model.
- **JPEG/PNG** — the image is sent to the VLM and answered directly.
- **GeoTIFF** (`.tif`/`.tiff`) — rendered as a viewable RGB preview and answered
  by the VLM (the preferred `POST /api/analyze` path also runs it through tools).
- **Bi-temporal change** (`POST /api/change`) — two co-registered images plus
  optional capture dates are sent to the VLM together and the change between
  them is described.
- **Grounding & map overlay** (`POST /api/ground`) — the VLM localizes every
  instance of a requested object and returns bounding boxes; for GeoTIFF uploads
  the boxes are mapped to real-world coordinates and rendered on an interactive
  **Folium/Leaflet** map with the image overlaid at the correct geo footprint.

Requests are dispatched through a small LangGraph state machine
(`backend/orchestration.py`) that routes on request shape — single-image,
bi-temporal, and optical-SAR branches exist, with the latter two returning
explicit "not implemented" responses until Phases 5+ land. Map overlays remain
deferred — see `plan.md` and `Design.md` for the roadmap. The 109 upstream MCP
tools live in `mcp_servers/agent_tools/` (Analysis, Index, Inversion,
Perception, Statistics); `backend/tool_router.py` selects and calls them.

## How to run

Prerequisites: Python 3.11+, a Hugging Face account, and an `HF_TOKEN` with Read
access (<https://huggingface.co/settings/tokens>).

```bash
# 1. Create a venv and install dependencies
python -m venv .venv
.venv/Scripts/activate        # Windows
.venv/bin/activate            # macOS / Linux
pip install -r requirements.txt

# 2. Configure the token
cp .env.example .env          # then edit .env and set HF_TOKEN=hf_...

# 3. Start the backend API
uvicorn backend.app:app --reload

# 4. In another terminal, start the web UI
python frontend/app.py
```

Then open the Gradio URL printed by the frontend, upload an image, ask a question,
and read the answer.

### Bare-bones check (no UI)

```bash
# Tool-augmented analysis — every format runs through the relevant MCP tools
# first; measurements + image are handed to the model for the answer.
curl -F image=@sample.jpg -F question="What is in this image?" \
  http://localhost:8000/api/analyze

curl -F image=@sample_multispectral.tif -F question="Is the vegetation healthy?" \
  http://localhost:8000/api/analyze

# JPG/PNG — straight from the image (no tools)
curl -F image=@sample.jpg -F question="What is in this image?" \
  http://localhost:8000/api/vqa

# GeoTIFF — band-stretched RGB preview answered directly by the VLM
curl -F image=@sample_multispectral.tif -F question="Is the vegetation healthy?" \
  http://localhost:8000/api/vqa

# Change detection — two co-registered images + optional dates
curl -X POST http://localhost:8000/api/change \
  -F image1=@change_before.jpg -F image2=@change_after.jpg \
  -F date1="2025-03-01" -F date2="2025-09-01" \
  -F question="What changed between these two aerial images?"

# Grounding — find objects and get pixel boxes (JPG)
curl -F image=@sample.jpg -F query="buildings" \
  http://localhost:8000/api/ground

# Grounding — GeoTIFF: boxes + Leaflet map with image + rectangles at real geo coords
curl -F image=@sample_multispectral.tif -F query="colored region" \
  http://localhost:8000/api/ground
```

*Note on GeoTIFFs:* the red/NIR/green bands are found from band
descriptions (e.g. `Red`, `NIR`). If a raster has no descriptions, pass
explicit indices by calling `compute_ndvi(path, red=1, nir=4)` directly.

## Run the tests

```bash
pytest
```

Without an `HF_TOKEN` set, the smoke test mocks the model call so the suite still
passes in CI.

## Repo layout

```
backend/
  app.py          # FastAPI app: POST /api/analyze, /api/vqa, /api/change, /api/ground
  schemas.py      # Pydantic v2 request/response models (AnalyzeResponse, VQAResponse, GroundResponse)
  analyzer.py     # tool-augmented analysis: upload -> MCP tools -> VLM answer
  tool_router.py  # ChromaDB tool index: retrieve top-k relevant tools, call them over MCP
  orchestration.py# LangGraph state machine: router -> single/bi-temporal/SAR nodes (Phase 4)
  vqa_service.py  # answer_question / answer_with_tools / ground_objects — the ONLY model-facing code
  geo_tools.py    # compute_ndvi / compute_ndwi band math (Phase 2)
  grounding.py    # Phase 6: tolerant box parser, geo-mapping, Folium overlay builder
mcp_servers/
  agent_tools/    # 109 vendored Earth-Agent MCP tools (Analysis, Index, Inversion, Perception, Statistics)
  .tool_index/    # persistent ChromaDB index of tool descriptions (built once, reused)
frontend/
  app.py          # Gradio UI
tests/
  test_vqa_smoke.py
  test_geo_tools.py
  test_orchestration.py
  test_grounding.py
.env.example      # HF_TOKEN=, VQA_MODEL=
requirements.txt
```