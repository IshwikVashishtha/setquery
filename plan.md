# Plan.md — Remote Sensing VQA Agent

## Guiding principle
Ship one real, end-to-end feature before adding architectural sophistication.
The source material (`SIH_Architecture_Complete_Discussion_Summary`) describes the
**target architecture** — not the build order. Build order is defined here.

Everything not needed to answer "what's in this image?" for a single uploaded
image is deferred until the MVP works and is checked off.

## Why MVP-first

| Approach | Pros | Cons |
|---|---|---|
| **MVP-first (this plan)** | Fast feedback, a working demo early, avoids building orchestration (LangGraph/MCP) for tools that don't exist yet, fewer moving parts to debug at once | Some early code (e.g. the direct model call) gets swapped out in later phases |
| **Full architecture first** | No rework later, matches the final design from day one | High risk of the demo landing late or not at all — LangGraph + MCP + ZeroGPU + routing all have to work together before anything is testable; classic over-engineering trap for a hackathon-scoped project |

Given the source docs read as SIH (Smart India Hackathon)-style scope, MVP-first is
the right call — a working demo beats a complete diagram.

## Phase roadmap

### Phase 0 — Scaffolding (~half a day)
Repo structure, dependencies, env config. No features yet.

### Phase 1 — MVP: Single-Image VQA — **current priority**
One image in, one question in, one text answer out. This is the feature that must
work before anything else is touched. No GeoTIFF, no NDVI, no MCP, no LangGraph,
no map overlay.

### Phase 2 — Geospatial basics
Accept GeoTIFF input, compute NDVI/NDWI locally with `rasterio` as plain Python
functions (no MCP yet). Feed the computed number into the model's prompt so the
answer can be checked against a real value instead of a guess.

### Phase 3 — MCP-ify the tools
Wrap Phase 2's functions (plus more) as an MCP server (`earth_agent.mcp_server`)
once there are enough tools to justify a protocol boundary. Add context-aware tool
truncation only once tool count actually risks context overflow — don't build
truncation logic for 2 tools.

### Phase 4 — Orchestration
Introduce LangGraph only once there's more than one path to route between
(single-image vs. bi-temporal vs. cross-modal optical/SAR). Before this phase, a
plain `if/else` does the same routing job the source doc describes, just not yet
expressed as a graph.

### Phase 5 — Multi-temporal / change detection
Bi-temporal image pairs, ChangeVQA path.

### Phase 6 — Segmentation & map overlay
SAM-2 grounding, Folium/Leaflet visual overlays in the UI.

### Phase 7 — Hardening
Guardrails, the auditable JSON log stream, auth, deployment.

## Definition of done (per phase)
A phase is done when:
1. The feature works against a real sample file, not a mock.
2. There's a documented manual test (in `Backlog.md`) that reproduces it.
3. `Backlog.md` is updated and the phase is checked off below.

## Status
- [x] Phase 0 — Scaffolding
- [ ] Phase 1 — Single-Image VQA MVP **(current priority)**
- [ ] Phase 2 — Geospatial basics (NDVI/NDWI)
- [ ] Phase 3 — MCP tool server
- [ ] Phase 4 — LangGraph orchestration
- [ ] Phase 5 — Multi-temporal change detection
- [ ] Phase 6 — Segmentation & map overlay
- [ ] Phase 7 — Hardening

## Reference
- `Design.md` — architecture (MVP + full target, phase-tagged)
- `Agent.md` — how the coding agent should work in this repo
- `Backlog.md` — granular, checkable tasks
