"""Connect to the PolicyPilot MCP server as an independent MCP client.

This starts the server as a separate process over stdio, discovers its tools
and calls each one, proving the server works with any MCP client, not only
our own agent.

Run from the project root:
    python -m app.tools.mcp_client_demo
    python -m app.tools.mcp_client_demo --mode hybrid      # also calls Gemini

Defaults use keyword search, so the demo makes no Gemini call. Policies must
already be indexed, and no other process may hold the local Qdrant folder.
"""

import argparse
import asyncio
import json
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters


def _print_result(name: str, result) -> None:
    print(f"\n--- {name} ---")
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        print(f"Tool error (expected for invalid input): {text}")
        return
    data = result.structured_content
    if data is None:  # fall back to text content
        data = [getattr(c, "text", str(c)) for c in result.content]
    output = json.dumps(data, indent=2, ensure_ascii=False)
    lines = output.splitlines()
    print("\n".join(lines[:40]) + ("\n  ..." if len(lines) > 40 else ""))


async def run(args: argparse.Namespace) -> None:
    server = StdioServerParameters(
        command=sys.executable,  # the same Python running this script
        args=["-m", "app.tools.mcp_server"],
        cwd=os.getcwd(),
        env=dict(os.environ),  # pass through .env-related variables and proxy settings
    )
    async with Client(server) as client:
        tools = await client.list_tools()
        print("Tools discovered:")
        for tool in tools.tools:
            first_line = (tool.description or "").strip().splitlines()[0]
            print(f"  - {tool.name}: {first_line}")

        _print_result(
            "search_policies",
            await client.call_tool(
                "search_policies",
                {"query": args.query, "k": 3, "mode": args.mode},
            ),
        )
        _print_result(
            "list_policy_versions",
            await client.call_tool(
                "list_policy_versions", {"policy_id": args.policy_id}
            ),
        )
        _print_result(
            "get_clause",
            await client.call_tool(
                "get_clause",
                {
                    "policy_id": args.policy_id,
                    "version": args.version,
                    "section_number": args.section,
                },
            ),
        )
        _print_result(
            "get_related_sections",
            await client.call_tool(
                "get_related_sections",
                {
                    "policy_id": args.policy_id,
                    "version": args.version,
                    "section_number": args.section,
                },
            ),
        )
        _print_result(
            "get_clause (invalid section)",
            await client.call_tool(
                "get_clause",
                {
                    "policy_id": args.policy_id,
                    "version": args.version,
                    "section_number": "99",
                },
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="PolicyPilot MCP client demo")
    parser.add_argument("--query", default="Can I work from home while on probation?")
    parser.add_argument(
        "--mode", choices=["keyword", "hybrid", "semantic"], default="keyword"
    )
    parser.add_argument("--policy-id", default="HR-REMOTE")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--section", default="3.1")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
        return 0
    except Exception as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
