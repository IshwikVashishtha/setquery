# Remote Sensing VQA Agent

Upload satellite or aerial imagery, ask a question in plain English, and get a
text answer. Powered by a hosted vision-language model (VLM) served through
Hugging Face Inference Providers.

This is the **Phase 1 MVP**: a single image + single question in, one text answer
out. GeoTIFF indices, MCP tools, orchestration graphs, and map overlays are
deliberately deferred — see `plan.md` and `Design.md` for the roadmap.

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
curl -F image=@sample.jpg -F question="What is in this image?" \
  http://localhost:8000/api/vqa
```

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
  vqa_service.py  # answer_question(image, question) -> str  — the ONLY model-facing code
frontend/
  app.py          # Gradio UI
tests/
  test_vqa_smoke.py
.env.example      # HF_TOKEN=, VQA_MODEL=
requirements.txt
```