import logging
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.graph import run_agent
from app.agent.llm import LLMError, StructuredLLM
from app.agent.toolbox import PolicyToolbox, ToolboxError

logger = logging.getLogger("policypilot.api")
router = APIRouter(tags=["Ask"])

MAX_ERROR_DETAIL = (
    300
)


@dataclass
class AgentRuntime:
    """Long-lived resources shared by all requests, created at API startup."""

    tools: PolicyToolbox
    llm: StructuredLLM | None  # None when GEMINI_API_KEY is not configured
    transport: str  # "mcp" or "direct", shown for diagnostics


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=2000,
        examples=["Can I work from home while on probation?"],
    )
    search_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


class Citation(BaseModel):
    evidence_id: str
    policy_id: str
    title: str | None = None
    version: str
    section_number: str
    heading: str | None = None
    source_file: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class AskResponse(BaseModel):
    status: Literal["answered", "clarify", "abstained"]
    answer: str
    citations: list[Citation]
    as_of_date: str | None = Field(
        description="Date the policies were checked against."
    )
    retrieval_attempts: int
    trace: list[str] = Field(
        description="Steps the agent took, for transparency and debugging."
    )


def get_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="The agent is not initialised.")
    return runtime


@router.post("/ask", response_model=AskResponse, summary="Ask a policy question")
async def ask(
    body: AskRequest, runtime: AgentRuntime = Depends(get_runtime)
) -> AskResponse:
    if runtime.llm is None:
        raise HTTPException(
            status_code=503, detail="GEMINI_API_KEY is not configured on the server."
        )

    try:
        result = await run_agent(
            body.question, runtime.llm, runtime.tools, body.search_mode
        )
    except LLMError as exc:
        # Full text goes to the server log only; it may contain proxy pages.
        logger.warning("LLM call failed: %s", str(exc)[:2000])
        raise HTTPException(
            status_code=502,
            detail="The language model could not be reached. Check network access to Gemini and the API key.",
        ) from exc
    except ToolboxError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Policy tools unavailable: {str(exc)[:MAX_ERROR_DETAIL]}",
        ) from exc

    if result.get("status") == "error":
        # Raised inside the graph when search fails (e.g. nothing indexed yet).
        raise HTTPException(
            status_code=503,
            detail=str(result.get("answer", "Policy search failed."))[
                :MAX_ERROR_DETAIL
            ],
        )

    return AskResponse(
        status=result["status"],
        answer=result["answer"],
        citations=[Citation(**c) for c in result.get("citations", [])],
        as_of_date=result.get("as_of_date"),
        retrieval_attempts=result.get("attempts", 0),
        trace=result.get("trace", []),
    )
