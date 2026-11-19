from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from app.core.config import Settings

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    """Anything that turns (system prompt, user prompt) into a schema instance."""

    async def generate(self, system: str, user: str, schema: type[T]) -> T: ...


class LLMError(RuntimeError):
    """The model call failed or returned something that does not fit the schema."""


class GeminiStructuredLLM:
    """Gemini via langchain-google-genai, using native structured output."""

    def __init__(self, settings: Settings) -> None:
        # Imported here so tests that use ScriptedLLM need no Gemini package.
        from langchain_google_genai import ChatGoogleGenerativeAI

        if settings.gemini_api_key is None:
            raise LLMError("GEMINI_API_KEY is not set in .env")
        self._model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,  # policy answers should be repeatable, not creative
        )

    async def generate(self, system: str, user: str, schema: type[T]) -> T:
        structured = self._model.with_structured_output(schema)
        try:
            result = await structured.ainvoke([("system", system), ("human", user)])
        except Exception as exc:  # network, quota, blocked by proxy, etc.
            raise LLMError(f"Gemini call failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(result, schema):
            raise LLMError(f"Gemini returned no valid {schema.__name__}")
        return result


class ScriptedLLM:

    def __init__(self, script: dict[type[BaseModel], list[Any]]) -> None:
        self._script = {schema: list(items) for schema, items in script.items()}
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((schema.__name__, user))
        queue = self._script.get(schema)
        if not queue:
            raise LLMError(f"ScriptedLLM has no response left for {schema.__name__}")
        item = queue.pop(0)
        result = item(user) if isinstance(item, Callable) else item
        if not isinstance(result, schema):
            raise LLMError(f"Scripted item is not a {schema.__name__}")
        return result
