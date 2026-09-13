# Remote Sensing VQA Agent

Upload satellite or aerial imagery, ask a question in plain English, and get a
text answer. Powered by a hosted vision-language model (VLM) served through
Hugging Face Inference Providers.

Phases 1–5 are implemented:

- **JPEG/PNG** — the image is sent to the VLM and answered directly.
- **GeoTIFF** (`.tif`/`.tiff`) — NDVI and NDWI are computed from the actual
  bands via the **Earth-Agent MCP server** (a `python -m earth_agent.mcp_server`
  subprocess), and the **measured values** are fed to the model so its answer can
  be checked against real numbers instead of guesses.
- **Bi-temporal change** (`POST /api/change`) — two co-registered images plus
  optional capture dates are sent to the VLM together and the change between
  them is described.

Requests are dispatched through a small LangGraph state machine
(`backend/orchestration.py`) that routes on request shape — single-image,
bi-temporal, and optical-SAR branches exist, with the latter two returning
explicit "not implemented" responses until Phases 5+ land. Map overlays remain
deferred — see `plan.md` and `Design.md` for the roadmap.

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
# JPG/PNG — answered straight from the image
curl -F image=@sample.jpg -F question="What is in this image?" \
  http://localhost:8000/api/vqa

# GeoTIFF — NDVI/NDWI computed locally, measured values given to the model
curl -F image=@sample_multispectral.tif -F question="Is the vegetation healthy?" \
  http://localhost:8000/api/vqa

# Change detection — two co-registered images + optional dates
curl -X POST http://localhost:8000/api/change \
  -F image1=@change_before.jpg -F image2=@change_after.jpg \
  -F date1="2025-03-01" -F date2="2025-09-01" \
  -F question="What changed between these two aerial images?"
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
  app.py          # FastAPI app: POST /api/vqa, GET /health (thin web layer)
  schemas.py      # Pydantic v2 request/response models
  orchestration.py# LangGraph state machine: router -> single/bi-temporal/SAR nodes (Phase 4)
  vqa_service.py  # answer_question / answer_index_question — the ONLY model-facing code
  geo_tools.py    # compute_ndvi / compute_ndwi band math (Phase 2)
  mcp_client.py   # spawns the MCP server subprocess, calls index tools (Phase 3)
earth_agent/
  mcp_server.py   # Earth-Agent MCP server exposing geo tools over stdio (Phase 3)
frontend/
  app.py          # Gradio UI
tests/
  test_vqa_smoke.py
  test_geo_tools.py
  test_mcp.py
.env.example      # HF_TOKEN=, VQA_MODEL=
requirements.txt
```