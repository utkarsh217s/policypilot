import asyncio
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Protocol

from app.tools import policy_tools
from app.tools.policy_tools import PolicyToolError, ToolContext


class ToolboxError(RuntimeError):
    """A tool call failed (bad input, nothing indexed, server problem)."""


class PolicyToolbox(Protocol):
    async def search_policies(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        policy_id: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]: ...
    async def get_clause(
        self, policy_id: str, version: str, section_number: str
    ) -> dict[str, Any]: ...
    async def get_related_sections(
        self, policy_id: str, version: str, section_number: str
    ) -> dict[str, Any]: ...
    async def list_policy_versions(self, policy_id: str) -> dict[str, Any]: ...


class DirectToolbox:
    """In-process calls to app.tools.policy_tools."""

    def __init__(self, context: ToolContext) -> None:
        self._ctx = context

    async def _call(self, func, *args, **kwargs) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(func, self._ctx, *args, **kwargs)
        except PolicyToolError as exc:
            raise ToolboxError(str(exc)) from exc
        except Exception as exc:
            raise ToolboxError(f"Internal tool error ({type(exc).__name__})") from exc

    async def search_policies(
        self, query, k=5, mode="hybrid", policy_id=None, version=None
    ):
        return await self._call(
            policy_tools.search_policies,
            query,
            k=k,
            mode=mode,
            policy_id=policy_id,
            version=version,
        )

    async def get_clause(self, policy_id, version, section_number):
        return await self._call(
            policy_tools.get_clause, policy_id, version, section_number
        )

    async def get_related_sections(self, policy_id, version, section_number):
        return await self._call(
            policy_tools.get_related_sections, policy_id, version, section_number
        )

    async def list_policy_versions(self, policy_id):
        return await self._call(policy_tools.list_policy_versions, policy_id)


class McpToolbox:
    """Calls the tools through the MCP server, started as a subprocess.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._client = None

    async def __aenter__(self) -> "McpToolbox":
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters

        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.tools.mcp_server"],
            cwd=os.getcwd(),
            env=dict(os.environ),
        )
        self._client = await self._stack.enter_async_context(Client(server))
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.aclose()

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise ToolboxError("McpToolbox must be used inside 'async with'")
        # Drop unset optional arguments instead of sending nulls.
        arguments = {k: v for k, v in arguments.items() if v is not None}
        result = await self._client.call_tool(name, arguments)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise ToolboxError(text or f"{name} failed")
        if not isinstance(result.structured_content, dict):
            raise ToolboxError(f"{name} returned no structured result")
        return result.structured_content

    async def search_policies(
        self, query, k=5, mode="hybrid", policy_id=None, version=None
    ):
        return await self._call(
            "search_policies",
            {
                "query": query,
                "k": k,
                "mode": mode,
                "policy_id": policy_id,
                "version": version,
            },
        )

    async def get_clause(self, policy_id, version, section_number):
        return await self._call(
            "get_clause",
            {
                "policy_id": policy_id,
                "version": version,
                "section_number": section_number,
            },
        )

    async def get_related_sections(self, policy_id, version, section_number):
        return await self._call(
            "get_related_sections",
            {
                "policy_id": policy_id,
                "version": version,
                "section_number": section_number,
            },
        )

    async def list_policy_versions(self, policy_id):
        return await self._call("list_policy_versions", {"policy_id": policy_id})
