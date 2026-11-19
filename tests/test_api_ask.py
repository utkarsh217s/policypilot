import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.agent.graph import DraftAnswer, EvidenceGrade, QuestionAnalysis, Verification
from app.agent.llm import ScriptedLLM
from app.agent.toolbox import DirectToolbox
from app.api.ask import AgentRuntime
from app.core.config import Settings
from app.main import create_app
from app.retrieval.vector_store import index_policy
from app.tools.policy_tools import ToolContext
from tests.test_agent_graph import (
    PDF,
    all_supported,
    answer_citing,
    insufficient,
    query,
    sufficient,
)


class AskEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        settings = Settings(
            gemini_api_key="unused", qdrant_path=cls.tmp / "q", _env_file=None
        )
        cls.context = ToolContext(settings)
        embeddings = DeterministicFakeEmbedding(size=settings.embedding_dimensions)
        cls.context.set_embeddings(embeddings)
        index_policy(PDF, cls.context.client, settings, embeddings)

        # A second context with nothing indexed, to test the "not indexed" path.
        empty = Settings(
            gemini_api_key="unused", qdrant_path=cls.tmp / "empty", _env_file=None
        )
        cls.empty_context = ToolContext(empty)

    @classmethod
    def tearDownClass(cls):
        cls.context.close()
        cls.empty_context.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def client(self, llm, context=None) -> TestClient:
        runtime = AgentRuntime(
            tools=DirectToolbox(context or self.context), llm=llm, transport="direct"
        )
        return TestClient(create_app(runtime))

    def ask(self, client, question="Can I work remotely on probation?", **extra):
        return client.post(
            "/api/v1/ask",
            json={"question": question, "search_mode": "keyword", **extra},
        )

    def test_answered_with_citations(self):
        llm = ScriptedLLM(
            {
                QuestionAnalysis: [query("probation")],
                EvidenceGrade: [sufficient],
                DraftAnswer: [answer_citing("2", "4")],
                Verification: [all_supported],
            }
        )
        response = self.ask(self.client(llm))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "answered")
        self.assertEqual({c["section_number"] for c in body["citations"]}, {"2", "4"})
        self.assertEqual(body["retrieval_attempts"], 1)
        self.assertTrue(body["trace"])

    def test_abstained_is_200_not_error(self):
        llm = ScriptedLLM(
            {QuestionAnalysis: [query("pension")], EvidenceGrade: [insufficient(None)]}
        )
        body = self.ask(self.client(llm)).json()
        self.assertEqual(body["status"], "abstained")
        self.assertEqual(body["citations"], [])

    def test_invalid_requests_rejected(self):
        client = self.client(ScriptedLLM({}))
        self.assertEqual(self.ask(client, question="").status_code, 422)
        self.assertEqual(self.ask(client, question="   ").status_code, 422)
        self.assertEqual(self.ask(client, unexpected=True).status_code, 422)
        self.assertEqual(
            client.post(
                "/api/v1/ask", json={"question": "x", "search_mode": "magic"}
            ).status_code,
            422,
        )

    def test_missing_llm_returns_503(self):
        response = self.ask(self.client(None))
        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

    def test_llm_failure_returns_502_without_details(self):
        response = self.ask(
            self.client(ScriptedLLM({}))
        )  # raises LLMError on first call
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("ScriptedLLM", response.json()["detail"])

    def test_nothing_indexed_returns_503(self):
        llm = ScriptedLLM({QuestionAnalysis: [query("probation")]})
        response = self.ask(self.client(llm, self.empty_context))
        self.assertEqual(response.status_code, 503)
        self.assertIn("index", response.json()["detail"].lower())

    def test_health_still_works(self):
        self.assertEqual(self.client(None).get("/api/v1/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
