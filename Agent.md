# Agent.md — Operating Rules for the Coding Agent

These rules apply to whichever AI coding agent (Claude Code or similar) is building
this repository. Read `plan.md` and `Design.md` first — this file is about *how* to
work, not *what* to build.

## 1. Scope discipline
- Build only what the current phase (see `plan.md`) requires. Don't pre-build
  LangGraph, MCP, or cloud-deployment plumbing "because the final architecture
  needs it eventually."
- If a `Backlog.md` task seems to require a deferred component to actually work,
  stop and say so instead of quietly expanding scope.

## 2. Isolate the parts that will change
The model-serving approach is expected to change across phases (hosted API ->
local model -> custom ZeroGPU Space — see `Design.md` §5). To keep that swap cheap:
- All model calls go through one function: `vqa_service.answer_question(image, question) -> str`.
- Nothing else in the codebase imports a model SDK directly.

## 3. Stack & conventions
- Python 3.11+, type hints on every function signature, docstrings on public functions.
- Dependencies pinned in `requirements.txt`. Don't add a dependency that isn't
  needed by the current phase's tasks (e.g. no `rasterio`/`langgraph`/`mcp` before
  Phase 2/3/4 respectively).
- FastAPI: async endpoints, Pydantic v2 models for request/response, `HTTPException`
  for error responses — no bare `except:`.
- Secrets (`HF_TOKEN`, etc.) via environment variables only, loaded with
  `python-dotenv` in dev. Never hardcode a token or commit `.env`.
- Logging via the standard `logging` module. Skip structured/observability logging
  until Phase 7 — that's over-engineering for an MVP.

## 4. Definition of done (applies to every task)
A `Backlog.md` task is only complete when:
1. It runs against a real input (a real image, not a placeholder).
2. The manual test steps listed next to the task were actually followed.
3. The checkbox is ticked, and — if it closes a phase — `plan.md`'s status table
   is updated too.

## 5. When something is ambiguous
Pick the simplest option that satisfies `Design.md`, note the assumption in the
commit message, and keep moving. Don't stall waiting for clarification on things
like "which exact model" — `Design.md` §5 already gives a default.

## 6. Don't build these until their phase arrives
| Component | Arrives in |
|---|---|
| Earth-Agent MCP server (104 tools) | Phase 3 |
| Context-aware tool truncation | Phase 3, and only once tool count justifies it |
| LangGraph state machine | Phase 4 |
| Bi-temporal / ChangeVQA | Phase 5 |
| SAM-2 segmentation, Folium/Leaflet overlay | Phase 6 |
| Guardrails, audit log, auth | Phase 7 |

## 7. Commits
One backlog task ~= one commit. Reference the task in the message, e.g.
`Phase 1: add /api/vqa endpoint`.
