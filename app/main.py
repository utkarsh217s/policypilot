import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from app.api.ask import AgentRuntime
from app.api.ask import router as ask_router
from app.api.health import router as health_router
from app.core.config import Settings, get_settings

logger = logging.getLogger("policypilot.api")


async def create_agent_runtime(
    settings: Settings, stack: AsyncExitStack
) -> AgentRuntime:
    """Create the LLM client and tool connection once, for all requests."""
    from app.agent.llm import GeminiStructuredLLM, LLMError
    from app.agent.toolbox import DirectToolbox, McpToolbox
    from app.tools.policy_tools import ToolContext

    try:
        llm = GeminiStructuredLLM(settings)  # no network call at construction
    except LLMError as exc:
        logger.warning("Starting without an LLM: %s", exc)
        llm = None

    if settings.agent_tool_transport == "mcp":
        tools = await stack.enter_async_context(McpToolbox())
    else:
        context = ToolContext(settings)
        stack.callback(context.close)
        tools = DirectToolbox(context)

    logger.info("Agent ready (tool transport: %s)", settings.agent_tool_transport)
    return AgentRuntime(tools=tools, llm=llm, transport=settings.agent_tool_transport)


def create_app(runtime: AgentRuntime | None = None) -> FastAPI:
    """Build the API. Pass a runtime to use injected tools/LLM (tests)."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with AsyncExitStack() as stack:
            if runtime is None:
                application.state.agent_runtime = await create_agent_runtime(
                    settings, stack
                )
            yield  # resources are closed when the server stops

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Evidence-based HR and compliance policy assistant.",
        lifespan=lifespan,
    )
    if runtime is not None:
        application.state.agent_runtime = runtime

    application.include_router(health_router, prefix="/api/v1")
    application.include_router(ask_router, prefix="/api/v1")
    return application


app = create_app()
