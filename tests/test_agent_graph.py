import re
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from app.agent.graph import (
    MAX_RETRIEVAL_ATTEMPTS,
    Claim,
    ClaimCheck,
    DraftAnswer,
    EvidenceGrade,
    QuestionAnalysis,
    Verification,
    run_agent,
)
from app.agent.llm import ScriptedLLM
from app.agent.toolbox import DirectToolbox
from app.core.config import Settings
from app.retrieval.vector_store import index_policy
from app.tools.policy_tools import ToolContext

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "policies" / "remote_work_policy_v1.pdf"
TODAY = date(2026, 6, 1)  # the sample policy is effective from 2026-04-01


# --- helpers that build scripted LLM responses from the prompt text ---------
def ids_in(prompt: str) -> list[str]:
    return re.findall(r"\[(E\d+)\]", prompt)


def id_for(prompt: str, section: str) -> str:
    match = re.search(rf"\[(E\d+)\] \S+ v\S+ §{re.escape(section)} ", prompt)
    if not match:
        raise AssertionError(f"Section {section} not in evidence")
    return match.group(1)


def query(text: str, as_of: str | None = None) -> QuestionAnalysis:
    return QuestionAnalysis(
        needs_clarification=False, search_query=text, as_of_date=as_of
    )


def sufficient(prompt: str) -> EvidenceGrade:
    return EvidenceGrade(sufficient=True, relevant_evidence_ids=ids_in(prompt))


def insufficient(next_query: str | None):
    return lambda prompt: EvidenceGrade(
        sufficient=False, missing_information="core hours", next_search_query=next_query
    )


def answer_citing(*sections: str):
    def build(prompt: str) -> DraftAnswer:
        claims = [
            Claim(text=f"Rule from section {s}", evidence_ids=[id_for(prompt, s)])
            for s in sections
        ]
        return DraftAnswer(answer="Synthetic answer.", claims=claims)

    return build


def all_supported(prompt: str) -> Verification:
    count = len(re.findall(r"^Claim \d+:", prompt, re.MULTILINE))
    return Verification(
        checks=[ClaimCheck(claim_index=i, supported=True) for i in range(count)]
    )


def none_supported(prompt: str) -> Verification:
    count = len(re.findall(r"^Claim \d+:", prompt, re.MULTILINE))
    return Verification(
        checks=[
            ClaimCheck(claim_index=i, supported=False, reason="not stated")
            for i in range(count)
        ]
    )


class AgentGraphTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.settings = Settings(
            gemini_api_key="unused", qdrant_path=cls.tmp / "qdrant", _env_file=None
        )
        cls.context = ToolContext(cls.settings)
        embeddings = DeterministicFakeEmbedding(size=cls.settings.embedding_dimensions)
        cls.context.set_embeddings(embeddings)
        index_policy(PDF, cls.context.client, cls.settings, embeddings)

    @classmethod
    def tearDownClass(cls):
        cls.context.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    async def ask(self, llm: ScriptedLLM, question: str = "Test question?"):
        return await run_agent(
            question,
            llm,
            DirectToolbox(self.context),
            search_mode="keyword",
            today=TODAY,
        )

    async def test_answer_includes_linked_exception(self):
        # Searching "probation" finds §2; §2 cites §4, so §4 is fetched too.
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("probation")],
                EvidenceGrade: [sufficient],
                DraftAnswer: [answer_citing("2", "4")],
                Verification: [all_supported],
            }
        )
        result = await self.ask(llm, "Can I work remotely while on probation?")
        self.assertEqual(result["status"], "answered")
        sections = {c["section_number"] for c in result["citations"]}
        self.assertEqual(sections, {"2", "4"})
        self.assertEqual(result["attempts"], 1)
        self.assertIn("2026-06-01", result["answer"])  # assumed date is disclosed

    async def test_retry_with_better_query(self):
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("pension gratuity")],  # finds nothing
                EvidenceGrade: [insufficient("core hours"), sufficient],
                DraftAnswer: [answer_citing("3.2")],
                Verification: [all_supported],
            }
        )
        result = await self.ask(llm)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["queries_tried"], ["pension gratuity", "core hours"])

    async def test_abstains_after_max_attempts(self):
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("pension")],
                EvidenceGrade: [
                    insufficient("gratuity"),
                    insufficient("bonus"),
                    insufficient("travel"),
                ],
            }
        )
        result = await self.ask(llm)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["attempts"], MAX_RETRIEVAL_ATTEMPTS)
        self.assertEqual(result["citations"], [])

    async def test_repeated_query_does_not_loop(self):
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("pension")],
                EvidenceGrade: [
                    insufficient("pension")
                ],  # proposes the same query again
            }
        )
        result = await self.ask(llm)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["attempts"], 1)

    async def test_invalid_citation_is_repaired(self):
        bad = DraftAnswer(answer="x", claims=[Claim(text="x", evidence_ids=["E99"])])
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("core hours")],
                EvidenceGrade: [sufficient],
                DraftAnswer: [bad, answer_citing("3.2")],
                Verification: [all_supported],
            }
        )
        result = await self.ask(llm)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["repair_count"], 1)
        # The regeneration prompt must explain what was wrong.
        last_draft_prompt = [u for name, u in llm.calls if name == "DraftAnswer"][-1]
        self.assertIn("E99", last_draft_prompt)

    async def test_unsupported_answer_abstains_after_repair(self):
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("core hours")],
                EvidenceGrade: [sufficient],
                DraftAnswer: [answer_citing("3.2"), answer_citing("3.2")],
                Verification: [none_supported, none_supported],
            }
        )
        result = await self.ask(llm)
        self.assertEqual(result["status"], "abstained")
        self.assertIn("not produce an answer", result["answer"])

    async def test_clarification_skips_search(self):
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [
                    QuestionAnalysis(
                        needs_clarification=True,
                        clarification_question="Which policy?",
                        search_query="",
                    )
                ]
            }
        )
        result = await self.ask(llm, "What about it?")
        self.assertEqual(result["status"], "clarify")
        self.assertNotIn("attempts", result)  # no retrieval happened

    async def test_date_before_policy_excludes_evidence(self):
        # The policy starts 2026-04-01, so nothing applies on 2025-01-01.
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("core hours", as_of="2025-01-01")],
                EvidenceGrade: [insufficient(None)],
            }
        )
        result = await self.ask(llm)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["evidence"], {})
        self.assertIn("2025-01-01", result["answer"])

    async def test_unreadable_date_asks_instead_of_guessing(self):
        llm = ScriptedLLM(
            {QuestionAnalysis: [query("core hours", as_of="last spring")]}
        )
        result = await self.ask(llm)
        self.assertEqual(result["status"], "clarify")


if __name__ == "__main__":
    unittest.main()
