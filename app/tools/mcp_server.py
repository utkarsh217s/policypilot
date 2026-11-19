"""PolicyPilot MCP server: exposes the four policy tools over MCP.

Any MCP client (our LangGraph agent, MCP Inspector, Claude Desktop, VS Code)
can discover and call these tools.

Run from the project root (stdio transport, the default for local clients):
    python -m app.tools.mcp_server

Run as an HTTP server instead (clients connect to http://127.0.0.1:8765/mcp):
    python -m app.tools.mcp_server --http --port 8765

Written for the MCP Python SDK v2 (package 'mcp' >= 2), where FastMCP was
renamed to MCPServer.

Local Qdrant storage allows ONE process at a time: stop this server before
running the index command against the same data/qdrant folder.
"""

import argparse
import atexit
import logging
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from app.tools import policy_tools
from app.tools.policy_tools import PolicyToolError, ToolContext

# stdio uses stdout for MCP messages, so logs must go to stderr only.
logging.basicConfig(level=logging.WARNING)

INSTRUCTIONS = """\
PolicyPilot gives read-only access to indexed HR and compliance policies.
Recommended workflow:
1. search_policies to find candidate clauses for a question.
2. get_clause to read the complete text of a clause before relying on it.
3. get_related_sections to find exceptions and cross-references.
4. list_policy_versions when the question concerns a date or an older rule.
Search scores rank candidates; they do not confirm that a clause answers the question.
Always cite policy_id, version and section_number.
"""

mcp = MCPServer(
    name="policypilot",
    title="PolicyPilot Policy Tools",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)

# One shared context per server process (single Qdrant client).
_context = ToolContext()
atexit.register(_context.close)

# All tools only read data.
READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True
)


def _run(func, *args, **kwargs) -> dict[str, Any]:
    """Call a tool function and turn expected errors into MCP tool errors.

    ToolError is returned to the client as a readable error result rather
    than crashing the server. Unexpected errors show only their type, so
    internal details (and the API key) are never sent to the client.
    """
    try:
        return func(_context, *args, **kwargs)
    except PolicyToolError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        logging.exception("Tool %s failed", func.__name__)
        raise ToolError(
            f"Internal error ({type(exc).__name__}). Check the server log."
        ) from exc


@mcp.tool(annotations=READ_ONLY)
def search_policies(
    query: str,
    k: int = 5,
    policy_id: str | None = None,
    version: str | None = None,
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
) -> dict[str, Any]:
    """Find policy chunks relevant to a question.

    Args:
        query: The question or search terms.
        k: Number of results to return (1-20).
        policy_id: Optional policy to restrict to. Requires version.
        version: Optional policy version. Requires policy_id.
        mode: 'hybrid' (default), 'semantic', or 'keyword' (no Gemini call).

    Returns candidate chunks with citation fields. Read the full clause with
    get_clause before using a chunk as evidence.
    """
    return _run(
        policy_tools.search_policies,
        query,
        k=k,
        policy_id=policy_id,
        version=version,
        mode=mode,
    )


@mcp.tool(annotations=READ_ONLY)
def get_clause(policy_id: str, version: str, section_number: str) -> dict[str, Any]:
    """Get the complete text of one clause.

    Args:
        policy_id: Policy identifier, e.g. 'HR-REMOTE'.
        version: Policy version, e.g. '1.0'. Required because sections differ between revisions.
        section_number: Section identifier, e.g. '3.1'.
    """
    return _run(policy_tools.get_clause, policy_id, version, section_number)


@mcp.tool(annotations=READ_ONLY)
def list_policy_versions(policy_id: str) -> dict[str, Any]:
    """List every indexed version of a policy with its effective dates.

    effective_from is inclusive, effective_to is exclusive; null means open-ended.
    Use this to choose the version that applies on a specific date.

    Args:
        policy_id: Policy identifier, e.g. 'HR-REMOTE'.
    """
    return _run(policy_tools.list_policy_versions, policy_id)


@mcp.tool(annotations=READ_ONLY)
def get_related_sections(
    policy_id: str, version: str, section_number: str
) -> dict[str, Any]:
    """Get clauses linked to a clause: those it cites and those that cite it.

    Use this to find exceptions. An exception often refers back to the general
    rule, so check 'cited_by' as well as 'cites'.

    Args:
        policy_id: Policy identifier.
        version: Policy version.
        section_number: Section whose links to follow.
    """
    return _run(policy_tools.get_related_sections, policy_id, version, section_number)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PolicyPilot MCP server")
    parser.add_argument(
        "--http", action="store_true", help="Serve over streamable HTTP"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        # Bound to localhost by default: the tools expose policy content.
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
