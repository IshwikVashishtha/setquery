# Remote Sensing VQA Agent

Upload satellite or aerial imagery, ask a question in plain English, and get a
text answer. Powered by a hosted vision-language model (VLM) served through
Hugging Face Inference Providers.

Phase 1 (single-image VQA) and Phase 2 (GeoTIFF NDVI/NDWI cross-check) are
implemented:

- **JPEG/PNG** — the image is sent to the VLM and answered directly.
- **GeoTIFF** (`.tif`/`.tiff`) — NDVI and NDWI are computed locally from the
  actual bands, and the **measured values** are fed to the model so its answer
  can be checked against real numbers instead of guesses.

MCP tools, orchestration graphs, and map overlays are deliberately deferred — see
`plan.md` and `Design.md` for the roadmap.

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
  app.py          # FastAPI app: POST /api/vqa, GET /health
  schemas.py      # Pydantic v2 request/response models
  vqa_service.py  # answer_question / answer_index_question — the ONLY model-facing code
  geo_tools.py    # compute_ndvi / compute_ndwi on GeoTIFFs (Phase 2)
frontend/
  app.py          # Gradio UI
tests/
  test_vqa_smoke.py
  test_geo_tools.py
.env.example      # HF_TOKEN=, VQA_MODEL=
requirements.txt
```