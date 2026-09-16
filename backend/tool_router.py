"""ChromaDB-backed tool selection over the vendored Earth-Agent MCP servers.

The five upstream servers (Analysis, Index, Inversion, Perception, Statistics)
expose ~109 MCP tools. Instead of hardcoding which tool a question needs, we
embed every tool's ``name + description`` once into a persistent ChromaDB
collection (all-MiniLM-L6-v2 via onnxruntime — fully local, stored on disk, so
restarts skip re-embedding), then per user question retrieve the top-k most
relevant tools and call only those over MCP.

The router deliberately *reports* a tool as skipped when its inputs can't be
satisfied from what the request provides (e.g. ``mann_kendall_test`` needs a
time series, not a raster path) — the answer layer and tests can see exactly
which tools were used and which were not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from mcp import ClientSession, StdioServerParameters, stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "mcp_servers" / "agent_tools"

#: Persistent ChromaDB store: embeddings built once, reused across restarts.
DEFAULT_INDEX_DIR = REPO_ROOT / "mcp_servers" / ".tool_index"
COLLECTION = "agent_tools"
SERVERS = ("Analysis", "Index", "Inversion", "Perception", "Statistics")

#: Above this count the collection is considered already built and we skip it.
_INDEXED_MIN = 100


def server_params(server: str) -> StdioServerParameters:
    """Mirror ``.mcp.json``: spawn an upstream server with this venv's python."""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(TOOLS_DIR / f"{server}.py"), "--temp_dir", str(DEFAULT_INDEX_DIR / "tmp")],
        cwd=str(TOOLS_DIR),
        env={**os.environ, "PYTHONPATH": str(TOOLS_DIR), "PYTHONUNBUFFERED": "1"},
    )


def _collection(index_dir: Path, create: bool = False):
    """Open the persistent collection with a shared default embedding function."""
    client = chromadb.PersistentClient(path=str(index_dir))
    fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        COLLECTION, embedding_function=fn
    ) if create else client.get_collection(COLLECTION)


async def _list_server_tools(server: str):
    """Spawn ``server`` over stdio and return its registered tools."""
    async with stdio_client(server_params(server)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return (await session.list_tools()).tools


async def build_index(index_dir: Path | None = None, force: bool = False) -> bool:
    """Embed every tool's name+description into the persistent store.

    Returns:
        ``True`` if the index was (re)built, ``False`` if it already existed
        and was left untouched (the "skip if already embedded" behaviour).
    """
    index_dir = index_dir or DEFAULT_INDEX_DIR
    col = _collection(index_dir, create=True)
    if not force and col.count() >= _INDEXED_MIN:
        return False

    total = col.count() + 0  # keep additions idempotent per id
    for server in SERVERS:
        tools = await _list_server_tools(server)
        ids, docs, metas = [], [], []
        for tool in tools:
            ids.append(f"{server}::{tool.name}")
            docs.append(f"{tool.name}\n{tool.description}")
            metas.append(
                {
                    "server": server,
                    "name": tool.name,
                    "schema": json.dumps(tool.inputSchema or {}),
                }
            )
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            total += len(ids)
    return True


def _is_callable(tool: dict) -> bool:
    """Whether this tool's args can be supplied from the uploaded file."""
    return map_tool_args(tool, "", "") is not None


#: Generic queries used to fill callable tools when a question's top-k are all
#: ones whose inputs can't be supplied from a single uploaded file (e.g. LST
#: inversion needs a matched day/night band pair, not one raster).
_FILL_QUERIES = (
    "calculate basic statistics of an image or raster file: mean, standard "
    "deviation, median, minimum, maximum, sum, brightness",
    "object detection, counting and scene classification in an aerial or "
    "satellite image",
)


async def _query(query: str, k: int, index_dir: Path) -> list[dict]:
    col = _collection(index_dir)
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )
    tools: list[dict] = []
    for meta, doc, dist in zip(
        res["metadatas"][0], res["documents"][0], res["distances"][0]
    ):
        try:
            schema = json.loads(meta.get("schema", "{}"))
        except (TypeError, json.JSONDecodeError):
            schema = {}
        tools.append(
            {
                "server": meta["server"],
                "name": meta["name"],
                "description": doc,
                "schema": schema,
                "distance": dist,
            }
        )
    return tools


async def retrieve_tools(
    query: str, k: int = 8, min_callable: int = 3, index_dir: Path | None = None
) -> list[dict]:
    """Return the top-``k`` tools most relevant to ``query``.

    Relevance is the primary ranking, but if fewer than ``min_callable`` of
    them have arguments we can supply from one uploaded file, callable tools
    from a short internal query mix are appended (deduped) so every analysis
    request actually executes tools, not just retrieves them.

    The returned list always has callable tools first (sorted by distance),
    then non-callable ones, capped at ``k`` total — so the VLM always
    receives at least ``min_callable`` real measurements.

    Each entry is ``{"server", "name", "description", "schema", "distance"}``.
    """
    index_dir = index_dir or DEFAULT_INDEX_DIR
    tools = await _query(query, k, index_dir)

    callable_tools = [t for t in tools if _is_callable(t)]
    non_callable = [t for t in tools if not _is_callable(t)]
    seen = {t["name"] for t in tools}

    if len(callable_tools) < min_callable:
        for filler in _FILL_QUERIES:
            for tool in await _query(filler, k, index_dir):
                if tool["name"] not in seen and _is_callable(tool):
                    callable_tools.append(tool)
                    seen.add(tool["name"])
            if len(callable_tools) >= min_callable:
                break

    # Callable tools first (real measurements), then non-callable for context
    result = callable_tools + non_callable
    return result[:k]


# ---------------------------------------------------------------------------
# Argument mapping: which tool inputs can be satisfied from the request.
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, dict] = {
    # Perception tools
    "MSCN": {"input_image_path": "path"},
    "RemoteCLIP": {"input_image_path": "path"},
    "InstructSAM": {"input_image_path": "path", "text_prompt": "query"},
    "RemoteSAM": {"input_image_path": "path", "text_prompt": "query"},
    "count_skeleton_contours": {"image_path": "path"},
    "count_above_threshold": {"file_path": "path", "threshold": "default_128"},
    # Statistics image-stat tools take ``file_list`` / ``*_paths`` = [path]
    "calc_batch_image_mean": {"file_list": "paths"},
    "calc_batch_image_std": {"file_list": "paths"},
    "calc_batch_image_median": {"file_list": "paths"},
    "calc_batch_image_min": {"file_list": "paths"},
    "calc_batch_image_max": {"file_list": "paths"},
    "calc_batch_image_sum": {"file_list": "paths"},
    "calc_batch_image_kurtosis": {"file_list": "paths"},
    "calc_batch_image_skewness": {"file_list": "paths"},
    "calc_batch_image_mean_mean": {"file_list": "paths"},
    "calc_batch_image_mean_max": {"file_list": "paths"},
    "calc_batch_image_mean_min": {"file_list": "paths"},
    "calc_batch_image_mean_max_min": {"file_list": "paths"},
    "calc_batch_image_mean_threshold": {"file_list": "paths", "threshold": "default_0_above"},
    "calculate_threshold_ratio": {"image_paths": "paths"},
    "calculate_area": {"input_image_path": "path"},
    "calculate_tif_average": {"file_list": "paths"},
    # Inversion tools usable on a single raster
    "calculate_water_turbidity_ntu": {"input_red_path": "path"},
}


def map_tool_args(tool: dict, file_path: str, query: str) -> dict | None:
    """Build callable arguments for ``tool``, or ``None`` if we can't supply them.

    ``query`` is the user's question (used for prompt-style args like
    ``text_prompt``); ``file_path`` is the on-disk uploaded image/raster.
    """
    handlers = _HANDLERS.get(tool["name"])
    if handlers is None:
        return None
    args: dict = {}
    for arg, kind in handlers.items():
        if kind == "path":
            args[arg] = file_path
        elif kind == "paths":
            args[arg] = [file_path]
        elif kind == "query":
            args[arg] = query
        elif kind == "default_128":
            args[arg] = 128.0
        elif kind == "default_0_above":
            args[arg] = 0.0
    return args


async def call_tools(
    selected: list[dict], file_path: str, query: str
) -> list[dict]:
    """Invoke the arg-mappable tools over MCP, grouped by server.

    Returns one entry per attempted tool:
    ``{"server", "name", "output"|"error", "skipped": true/false}``.
    """
    outputs: list[dict] = []
    by_server: dict[str, list[dict]] = {}
    for tool in selected:
        args = map_tool_args(tool, file_path, query)
        if args is None:
            outputs.append(
                {
                    "server": tool["server"],
                    "name": tool["name"],
                    "skipped": True,
                    "output": "",
                    "reason": "no callable argument mapping for the uploaded file",
                }
            )
            continue
        tool["_args"] = args
        by_server.setdefault(tool["server"], []).append(tool)

    for server, tools in by_server.items():
        async with stdio_client(server_params(server)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for tool in tools:
                    entry = {"server": server, "name": tool["name"], "skipped": False}
                    try:
                        result = await session.call_tool(tool["name"], tool["_args"])
                        entry["output"] = ""
                        if result.content:
                            entry["output"] = result.content[0].text
                        if result.isError:
                            entry["error"] = entry["output"]
                    except Exception as exc:  # per-tool isolation
                        entry["error"] = str(exc)
                    outputs.append(entry)
    return outputs