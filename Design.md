# Design.md — Architecture (MVP-first, phase-tagged)

Source material: `SIH_Architecture_Complete_Discussion_Summary` (md/pdf) plus the two
diagram screenshots. That material describes the **full target architecture**. This
doc keeps the same end goal but tags every component with the phase (from `plan.md`)
it actually gets built in, and fully specifies Phase 1.

## 1. Full target architecture (condensed reference)

```
[Web Dashboard: Gradio/Streamlit]                                    Phase 1 (dashboard),
        |                                                              Phase 6 (map overlay)
        v
[FastAPI Gateway] -- validates files, reads GeoTIFF headers          Phase 1 (basic),
        |                                                              Phase 2 (GeoTIFF)
        v
[LangGraph Router] -- picks: single-image / bi-temporal / SAR-fusion  Phase 4
        |
   -----+------------------+------------------------+
   v                       v                         v
Single-image          Bi-temporal              Optical-SAR
                       change                    fusion
   |                       |                          |
   +-----------------------+--------------------------+
                            v
            [Earth-Agent MCP: 104 tools, context-truncated]           Phase 3
                            v
   [HF ZeroGPU Space: PaliGemma/Florence-2 (VQA), SAM-2, ChangeVQA]    Phase 1 (hosted model
                            v                                          instead of own Space),
            [Output Synthesizer: text + Leaflet/Folium overlay]        Phase 6
                            v
                [Auditable JSON log stream]                           Phase 7
```

## 2. Phase 1 architecture — build this now

```
[Gradio UI] --image + question--> [FastAPI: POST /api/vqa] --> [vqa_service.answer_question()]
                                                                          |
                                                    +---------------------+
                                                    v
                                    [HF Inference Providers: chat.completions
                                     with a hosted VLM, e.g. Qwen2.5-VL-3B-Instruct]
                                                    |
        <---------------------- text answer --------+
```
No GeoTIFF, no MCP, no LangGraph, no map. One file type (JPG/PNG), one path, one answer.

## 3. Repo layout (Phase 1)
```
backend/
  app.py          # FastAPI app: POST /api/vqa, GET /health
  schemas.py      # Pydantic v2 request/response models
  vqa_service.py  # answer_question(image, question) -> str  <- the ONLY file that talks to a model
frontend/
  app.py          # Gradio: image upload + textbox + submit, calls backend
tests/
  test_vqa_smoke.py
.env.example       # HF_TOKEN=, VQA_MODEL=
requirements.txt
```

## 4. API contract (Phase 1)
`POST /api/vqa`
```
# request (multipart/form-data)
image: <file>
question: "How many buildings are visible?"

# response
{ "answer": "There appear to be roughly 12 distinct rooftop structures." }
```
`GET /health` -> `{ "status": "ok" }`

## 5. Model-serving decision

| Option | How | Pros | Cons | Use when |
|---|---|---|---|---|
| **A. HF Inference Providers (recommended)** | OpenAI-compatible `chat.completions` call to `https://router.huggingface.co/v1`, image passed as `image_url` (a base64 data URI works the same as a hosted URL) | Zero deployment, no GPU to manage, works today with small models like `Qwen/Qwen2.5-VL-3B-Instruct`, generous free tier | Not remote-sensing-specialized, adds a network dependency, provider/model lineup can shift over time | **Phase 1** |
| B. Local `transformers` pipeline | Load a small VLM in-process (CPU or local GPU) | Fully offline, easiest to debug line-by-line | Heavy install, slow on CPU, pulls in deep-learning deps before the project needs them | Only if there's no internet/HF token available |
| C. Custom HF ZeroGPU Space (the original design) | Deploy your own Gradio Space with `@spaces.GPU`, backend calls it | Matches the source docs exactly, supports a fine-tuned remote-sensing model, free serverless GPU | Real deployment overhead (Space config, cold starts) — pointless before a fine-tuned model exists to justify it | Phase 3+ |

**Decision:** Phase 1 uses Option A. Because `vqa_service.answer_question()` is the
only place that touches a model (see `Agent.md` §2), moving to Option C later once a
fine-tuned model exists is a one-file change, not a rewrite.

### Reference snippet for Option A
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

def answer_question(image_data_uri: str, question: str) -> str:
    completion = client.chat.completions.create(
        model=os.environ.get("VQA_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        }],
    )
    return completion.choices[0].message.content
```
Build `image_data_uri` as `f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"`.

## 6. Config
`.env`:
```
HF_TOKEN=hf_xxx
VQA_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

## 7. Deferred design (unchanged from the source docs, just phase-tagged)
- **Phase 2:** `rasterio`/GDAL functions for NDVI/NDWI on GeoTIFF input, as plain
  Python functions — no MCP yet.
- **Phase 3:** Wrap Phase 2's tools (plus more) behind an MCP server
  (`earth_agent.mcp_server`, stdio transport, per the source doc's subprocess
  lifecycle pattern). Add context-aware tool truncation only once the tool count is
  large enough to risk context overflow for a 7B model.
- **Phase 4:** Replace the hardcoded single path with a LangGraph state machine that
  routes on file count/type (single image / bi-temporal / optical-SAR).
- **Phase 5:** ChangeVQA path for bi-temporal image pairs.
- **Phase 6:** SAM-2 grounding/segmentation, Folium/Leaflet map overlays in the UI.
- **Phase 7:** Guardrails, the auditable JSON log stream, auth, deployment hardening.

Nothing about the *target* design changes from the source documents — only the
order it gets built in.
