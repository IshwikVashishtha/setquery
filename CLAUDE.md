# CLAUDE.md

Entry point for any AI coding agent (Claude Code or similar) working in this repo.
Read this first, then follow the links below, in order.

## What this project is
A remote sensing AI agent: upload satellite/aerial imagery, ask a question in plain
English, get an answer (VQA). The long-term target also includes geospatial index
computation (NDVI/NDWI), multi-temporal change detection, segmentation, and map
overlays — see `Design.md` for the full picture — but **do not build those yet**.
See `plan.md` for why the build order differs from the target architecture.

## Read next, in this order
1. `plan.md` — phased roadmap and the current priority phase
2. `Design.md` — architecture for the current phase + every deferred phase
3. `Agent.md` — rules for how you (the coding agent) should operate in this repo
4. `Backlog.md` — the actual task list; work top-to-bottom within the current phase

## Current priority
**Phase 1 — single-image VQA MVP.** Do not write FastAPI-orchestration, LangGraph,
MCP, or map-overlay code until Phase 1 is checked off in `plan.md` and its manual
test in `Backlog.md` passes. If a deferred component genuinely seems required to
finish Phase 1, say so and explain why before building it — don't silently expand
scope.

## Golden rule
One working feature beats five scaffolded ones. If a `Backlog.md` task isn't
runnable and testable by a human within the same session, break it down further
before writing code.
