# Backlog.md

Check items off as they're completed. Work top to bottom within a phase; don't
start a later phase's tasks early (see `Agent.md` §1).

## Phase 0 — Scaffolding
- [x] `git init`; create the repo structure from `Design.md` §3
- [x] `requirements.txt`: fastapi, uvicorn[standard], pydantic, python-multipart,
      python-dotenv, openai, gradio, pillow, pytest, httpx
- [x] `.env.example` with `HF_TOKEN=` and `VQA_MODEL=`
- [x] `README.md`: one-paragraph project description + how to run (fill in as
      Phase 1 lands)
- **Manual test:** `pip install -r requirements.txt` succeeds in a clean venv. ✅

## Phase 1 — Single-Image VQA (MVP — current priority)
- [x] `backend/schemas.py`: `VQAResponse(BaseModel)` with `answer: str`
- [x] `backend/vqa_service.py`: implement `answer_question(image: PIL.Image, question: str) -> str`
      using `Design.md` §5 Option A. Raise a clear exception on API failure.
- [x] `backend/app.py`: FastAPI app with
  - `GET /health` -> `{"status": "ok"}`
  - `POST /api/vqa` (multipart: `image` file + `question` str) -> `VQAResponse`
  - 400 on missing/invalid image, 502 on model-call failure
- [x] **Manual test:** run `uvicorn backend.app:app --reload`, then
      `curl -F image=@sample.jpg -F question="What is in this image?" http://localhost:8000/api/vqa`
      returns a coherent JSON answer. ✅ (live on :8000)
- [x] `frontend/app.py`: Gradio `Interface` — image upload + question textbox ->
      calls the backend endpoint via `httpx`, displays the answer
- [x] **Manual test:** run `python frontend/app.py`, upload a real photo, ask a
      question, get a sensible answer in the browser within ~10 seconds.
      ✅ (Gradio serving on :7860; `ask_vqa` path verified live, ~4.4 s)
- [x] `tests/test_vqa_smoke.py`: pytest that calls `answer_question()` directly
      with a sample image and asserts a non-empty string is returned (mock the API
      call if running in CI without an `HF_TOKEN`) — 8 passed incl. a live call
- [x] Fill in `README.md` run instructions
- [x] Tick "Phase 1" in `plan.md`'s status table

## Phase 2 — Geospatial basics (don't start before Phase 1 is ticked off)
- [ ] Add `rasterio` to `requirements.txt`
- [ ] `backend/geo_tools.py`: `compute_ndvi(path: str, bbox: list[float]) -> float`,
      `compute_ndwi(...)` — real GeoTIFF band math, not a placeholder
- [ ] Extend `/api/vqa` to accept `.tif`/`.tiff` uploads, compute the relevant
      index, and include the number in the prompt sent to the model (the
      "cross-check" idea from the source docs — validate the model's textual claim
      against the real number)
- [ ] **Manual test:** upload a sample multispectral GeoTIFF, ask "is this
      vegetation healthy?", confirm the answer references a real NDVI value, not a
      hallucinated one.

## Phase 3 — MCP-ify the tools
- [ ] Add `mcp` to `requirements.txt`
- [ ] Move `geo_tools.py` functions into `earth_agent/mcp_server.py`, exposed as
      MCP tools
- [ ] `backend/mcp_client.py`: spawn the subprocess via `stdio_client` (pattern
      from the source doc's §5.1 / §4), call tools, close cleanly
- [ ] Add context-aware tool truncation only once tool count > ~10
- [ ] **Manual test:** same NDVI query as Phase 2, now routed through the MCP
      subprocess instead of a direct function call.

## Phase 4 — Orchestration
- [ ] Add `langgraph`/`langchain-core` to `requirements.txt`
- [ ] Replace the hardcoded single path with a graph: router node -> single-image /
      bi-temporal / optical-SAR node
- [ ] **Manual test:** a single-image query still returns the same answer as
      before — the graph must not regress existing behavior.

## Phase 5 — Multi-temporal / change detection
- [ ] Accept two co-registered images + dates
- [ ] Route to a ChangeVQA-capable model via the same `vqa_service`-style isolation
- [ ] **Manual test:** two sample images with an obvious visible change produce a
      correct change description.

## Phase 6 — Segmentation & map overlay
- [ ] Add a SAM-2 grounding path
- [ ] Add a Folium/Leaflet overlay in the frontend for boxes/masks + coordinates
- [ ] **Manual test:** a grounding query returns a box overlaid on the correct
      location on the map.

## Phase 7 — Hardening
- [ ] Structured JSON audit log (task, model, params, timing) per the source doc's
      "Auditable JSON Log Stream"
- [ ] Guardrails on input/output
- [ ] Basic auth on the FastAPI gateway
- [ ] Deployment (containerize backend + frontend)
