"""
app/mcp_client.py
─────────────────
MCP client that connects to mcp_server.py via stdio transport.

This is the Gap 1 fix: instead of calling SecToolRegistry Python methods
directly, the agent now communicates with the MCP server over the Model
Context Protocol — exactly as an external MCP host (Claude Desktop, a
remote agent, etc.) would do at runtime.

How it works:
  1. SyncSecMCPClient.list_tools() spawns mcp_server.py as a subprocess
     and calls ClientSession.list_tools() over stdin/stdout.
  2. SyncSecMCPClient.call_tool() sends a CallToolRequest over the same
     channel and receives a structured response.
  3. Both calls are async internally; the sync wrapper runs them in a
     fresh event loop so Streamlit / FastAPI / CLI can call them normally.
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ── Server spawn parameters ───────────────────────────────────────────────

def _server_params() -> StdioServerParameters:
    """
    Tell the MCP client how to spawn the server process.
    Uses the same Python interpreter that is currently running so the
    server inherits the active virtual-environment packages.
    """
    return StdioServerParameters(
        command=sys.executable,      # e.g. .venv/bin/python
        args=["-m", "app.mcp_server"],
        env=None,                    # inherit current environment (.env already loaded)
    )


# ── Async client (single session lifetime) ────────────────────────────────

class SecMCPClient:
    """
    Async context-manager wrapping mcp.ClientSession.

    Opens a stdio transport to mcp_server.py on entry, initialises the
    MCP session (capability negotiation), and tears down on exit.

    Usage:
        async with SecMCPClient() as client:
            tools  = await client.list_tools()
            result = await client.call_tool("get_financial_snapshot",
                                             {"ticker": "AAPL"})
    """

    def __init__(self) -> None:
        self._session:    ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "SecMCPClient":
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # 1. Open stdio transport → spawns mcp_server.py subprocess
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(_server_params())
        )

        # 2. Create MCP session and perform protocol handshake
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._exit_stack:
            await self._exit_stack.__aexit__(*exc_info)
        self._session    = None
        self._exit_stack = None

    # ── MCP operations ────────────────────────────────────────────────────

    async def list_tools(self) -> list[dict]:
        """
        Ask the MCP server which tools it exposes.

        Returns a list of dicts:
            [{"name": ..., "description": ..., "inputSchema": ...}, ...]

        This is the agent's runtime tool discovery — it does NOT import
        any Python module from app/; it queries the server over the wire.
        """
        if self._session is None:
            raise MCPClientError("Not connected — use 'async with SecMCPClient()'")

        response = await self._session.list_tools()
        return [
            {
                "name":        tool.name,
                "description": tool.description or "",
                "inputSchema": (
                    tool.inputSchema.model_dump()
                    if hasattr(tool.inputSchema, "model_dump")
                    else (tool.inputSchema or {})
                ),
            }
            for tool in response.tools
        ]

    async def call_tool(self, name: str,
                        arguments: dict[str, Any]) -> Any:
        """
        Invoke a named tool on the MCP server.

        The server executes the tool and returns a list of content blocks.
        We unwrap the first TextContent block and JSON-parse it so callers
        receive a native Python object (dict / list).
        """
        if self._session is None:
            raise MCPClientError("Not connected — use 'async with SecMCPClient()'")

        result = await self._session.call_tool(name, arguments)

        if result.isError:
            raise MCPClientError(
                f"MCP server returned error for tool '{name}': "
                + _content_to_str(result.content)
            )

        if not result.content:
            return {}

        # Unwrap first content block
        first = result.content[0]
        text  = getattr(first, "text", None)
        if text is None:
            return result.content          # return raw list as fallback

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text                    # plain string — return as-is


# ── Sync wrapper (for Streamlit / FastAPI / CLI) ──────────────────────────

class SyncSecMCPClient:
    """
    Synchronous facade over SecMCPClient.

    Opens a fresh async client for every call so there is no persistent
    subprocess between requests.  Acceptable for a prototype; a production
    system would pool connections.

    Each method blocks until the async coroutine completes.
    """

    def list_tools(self) -> list[dict]:
        """Discover available tools from the MCP server."""
        return asyncio.run(self._a_list_tools())

    def call_tool(self, name: str, arguments: dict) -> Any:
        """Invoke a tool on the MCP server and return its result."""
        return asyncio.run(self._a_call_tool(name, arguments))

    async def _a_list_tools(self) -> list[dict]:
        async with SecMCPClient() as client:
            return await client.list_tools()

    async def _a_call_tool(self, name: str, arguments: dict) -> Any:
        async with SecMCPClient() as client:
            return await client.call_tool(name, arguments)


# ── Helpers ───────────────────────────────────────────────────────────────

class MCPClientError(Exception):
    """Raised when the MCP server returns an error or is unreachable."""


def _content_to_str(content: Any) -> str:
    if isinstance(content, list):
        return " | ".join(
            getattr(c, "text", str(c)) for c in content
        )
    return str(content)
