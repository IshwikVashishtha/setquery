"""Compatibility shim: upstream Earth-Agent tool files do ``from fastmcp import
FastMCP`` and call ``FastMCP()`` with no arguments.

This repo already pins the ``mcp`` SDK (<2.0) for the backend, whose
``mcp.server.fastmcp.FastMCP`` needs a ``name`` argument. This shim sits in the
same directory as the tool files (PYTHONPATH resolves it before any installed
package) and forwards to the SDK's FastMCP with a default name injected, so the
upstream servers run unmodified.
"""

from mcp.server.fastmcp import FastMCP as _FastMCP


def FastMCP(*args, **kwargs):
    """``mcp.server.fastmcp.FastMCP`` with a default server name."""
    if not args and "name" not in kwargs:
        kwargs["name"] = "earth-agent"
    return _FastMCP(*args, **kwargs)