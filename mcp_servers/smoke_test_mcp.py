"""Smoke-test the 5 upstream Earth-Agent MCP servers over stdio."""
import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from mcp import ClientSession, StdioServerParameters, stdio_client
from rasterio.transform import from_origin

TOOLS_DIR = Path(r"D:\Ai - Engineer\setquery\mcp_servers\agent_tools")
OUT_DIR = Path(tempfile.mkdtemp(prefix="mcp_smoke_"))
PYEXE = sys.executable


def make_tif(path: Path, values) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", width=4, height=4, count=1,
        dtype="float32", crs="EPSG:3857", transform=from_origin(0, 10, 1, 1),
    ) as dst:
        dst.write(np.array(values, dtype="float32").reshape(4, 4), 1)
    return path


def server_params(name: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=PYEXE,
        args=[str(TOOLS_DIR / f"{name}.py"), "--temp_dir", str(OUT_DIR)],
        cwd=str(TOOLS_DIR),
        env={"PYTHONPATH": str(TOOLS_DIR), "PYTHONUNBUFFERED": "1"},
    )


async def list_tools(name: str):
    async with stdio_client(server_params(name)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return (await s.list_tools()).tools


async def call_tool(name: str, tool: str, args: dict):
    async with stdio_client(server_params(name)) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(tool, args)
            text = res.content[0].text if res.content else "<no content>"
            return res.isError, text


async def main():
    tif = make_tif(Path(tempfile.mkdtemp()) / "vals.tif", np.arange(16).tolist())
    binary = make_tif(Path(tempfile.mkdtemp()) / "bin.tif",
                      (np.arange(16) % 2).tolist())  # 8 pixels == 1.0

    for name, expect, spot in [
        ("Index", 12, ["calculate_batch_ndvi", "compute_tvdi"]),
        ("Analysis", 10, ["compute_linear_trend", "mann_kendall_test"]),
        ("Inversion", 17, ["lst_single_channel", "split_window"]),
        ("Perception", 15, ["threshold_segmentation", "bboxes2centroids"]),
        ("Statistics", 50, ["mean", "count_above_threshold"]),
    ]:
        tools = await list_tools(name)
        names = {t.name for t in tools}
        missing = [s for s in spot if s not in names]
        status = "OK " if len(tools) == expect and not missing else "FAIL"
        print(f"{status} {name}: {len(tools)} tools (expect {expect}) missing={missing}")

    # --- functional calls ---
    print("\n-- calls --")
    for label, src, tool, args in [
        ("statistics.mean", "Statistics", "mean", {"x": [1, 2, 3, 4]}),
        ("statistics.difference", "Statistics", "difference", {"a": 5, "b": 3}),
        ("statistics.count_above_threshold", "Statistics", "count_above_threshold",
         {"file_path": str(tif), "threshold": 8.0}),
        ("analysis.compute_linear_trend", "Analysis", "compute_linear_trend",
         {"y": [1, 2, 3, 4]}),
        ("analysis.detect_seasonality_acf", "Analysis", "detect_seasonality_acf",
         {"values": [1, 2, 1, 2, 1, 2]}),
        ("perception.bboxes2centroids", "Perception", "bboxes2centroids",
         {"bboxes": [[0, 0, 2, 4]]}),
        ("perception.count_above_threshold", "Perception", "count_above_threshold",
         {"file_path": str(tif), "threshold": 8.0}),
        ("index.snow_loss_pct", "Index", "calc_extreme_snow_loss_percentage_from_binary_map",
         {"binary_map_path": str(binary)}),
    ]:
        is_err, text = await call_tool(src, tool, args)
        txt = text[:150].replace("\n", " ")
        tag = "ERR" if is_err else "ok "
        print(f"  [{tag}] {label}: {txt}")


asyncio.run(main())