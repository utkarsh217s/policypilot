# PolicyPilot

PolicyPilot answers questions about HR and compliance policies using only the policy documents you give it. Every answer points to the policy, version and section it came from. If the documents don't actually answer the question, it says so rather than making something up.

Under the hood it's a RAG pipeline with a LangGraph agent on top. After retrieving clauses, the agent checks whether they really cover the question. If they don't, it searches again with a better query. Once it has an answer, it checks each statement against the clause it cites before returning it.

All policies in this repo are fictional. Please don't index real employee or company data.

## How it works

Indexing a document:

1. A LangChain loader reads the PDF, DOCX or text file.
2. Policy details (ID, version, effective date and so on) are read from the header at the top of the document.
3. The body is split into numbered sections, and each section becomes a clause.
4. Clauses are chunked, embedded with Gemini (`gemini-embedding-001`) and stored in Qdrant.

Answering a question:

1. The Streamlit UI sends the question to the FastAPI backend.
2. The LangGraph agent turns it into a search query and calls the policy tools through an MCP server.
3. Search combines Gemini semantic search with BM25 keyword search, merged with reciprocal rank fusion.
4. The agent grades the evidence, retries the search if something is missing (three searches at most), writes an answer and verifies it.
5. If it can't produce a supported answer, it explains what was missing and suggests checking with HR.

The agent also follows references between sections in both directions. That matters for exceptions. In the sample remote work policy, section 3.1 sets a two-day limit and never mentions section 4, but section 4 (medical exceptions) refers back to 3.1. Looking only at what 3.1 cites would miss the exception.

## Requirements

- Python 3.12 or newer
- A Gemini API key

The tests don't need Gemini, so you can run them without a key or network access.

## Setup

The commands below are for PowerShell, run from the project folder.

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768
```

Keep `.env` out of Git. The same goes for `data/qdrant/`, which is where the local vector database lives.

Then check that Gemini is reachable:

```powershell
python -m app.core.test_gemini
```

## Writing policy documents

Put your files in `data/policies/`. PDF (with real text, not scans), DOCX, TXT and Markdown all work.

Start the document with a short header before the first numbered section, like this:

```text
Remote Work Policy
Policy ID: HR-REMOTE
Version: 1.0
Effective Date: 1 April 2026
Applies To: Permanent employees
Departments: Engineering, Finance
```

The version is the only field you can't leave out. Two revisions saved under the same version would be indistinguishable, so files without one are rejected. If there's no policy ID, one is built from the title. Fields you leave out are recorded as unknown rather than guessed. Dates like `2026-04-01` or `1 April 2026` are safest. Slash dates are read day first.

Sections need numbered headings (`1 Scope`, `2. Eligibility`, `3.1 Remote Days`, `Section 4: Exceptions`). When a section says "see Section 4", the link is picked up automatically.

To check how a file will be read before you index it, run:

```powershell
python -m app.ingestion.document_loader data\policies\remote_work_policy_v1.pdf
```

It prints each metadata value along with where it came from, then lists the sections it found.

If you can't edit a document, you can put a `<filename>.meta.json` next to it to override any header values. Normally you won't need one.

## Running it

Index a policy:

```powershell
python -m app.retrieval.vector_store index data\policies\remote_work_policy_v1.pdf
```

Running this twice on the same file is fine. If you change a document's content, bump its version first. Otherwise indexing will refuse the change.

Try a search:

```powershell
python -m app.retrieval.hybrid_search "Can I work from home on probation?"
```

Ask the agent directly. `--trace` prints each step it takes, which helps when an answer looks wrong:

```powershell
python -m app.agent.graph "Can I work from home while on probation?" --trace
```

To run the full app, start the API in one terminal and the UI in another:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
python -m streamlit run ui/streamlit_app.py
```

API docs are at http://127.0.0.1:8000/docs.

One thing to watch: local Qdrant only allows one process to use its folder at a time. The API starts the MCP server, and the MCP server opens that folder. Stop the API before indexing anything new.

## The API

`POST /api/v1/ask` takes a question and an optional search mode (`hybrid`, `semantic` or `keyword`):

```json
{ "question": "How many days a week can I work remotely?", "search_mode": "hybrid" }
```

The response has a `status`, the `answer`, a list of `citations`, the date the policies were checked against, and a `trace` of the agent's steps. The status is `answered`, `clarify` (the question needs more detail) or `abstained` (the policies don't support an answer).

An abstention comes back as a normal 200, because "the policies don't cover this" is a valid result. Errors come back as 502 if Gemini can't be reached, and 503 if nothing is indexed or the key is missing.

Questions are answered independently. There's no chat history between them.

## MCP tools

The agent uses four read-only tools, also available to any other MCP client:

- `search_policies` finds candidate chunks.
- `get_clause` returns the full text of one section of one policy version.
- `list_policy_versions` lists every indexed version with its effective dates.
- `get_related_sections` returns sections a clause cites and sections that cite it.

To see them working from a separate client:

```powershell
python -m app.tools.mcp_client_demo
```

You can also run the server on its own with `python -m app.tools.mcp_server`, or over HTTP with `--http --port 8765`.

## Tests

The tests use fake embeddings and scripted LLM responses, so they run without Gemini or a network connection:

```powershell
python -m unittest discover -s tests -v
```

They cover the schemas, metadata extraction, the `/ask` endpoint, and each path through the agent: answering with an exception, retrying a search, giving up after three searches, fixing a bad citation, abstaining, asking for clarification, and filtering by date.

## Settings

Everything is configured through `.env`. Besides the Gemini settings above:

- `QDRANT_URL` points at a Qdrant server. Leave it empty to use local storage in `QDRANT_PATH` (default `data/qdrant`).
- `AGENT_TOOL_TRANSPORT` is `mcp` (default) or `direct`. `direct` skips the MCP subprocess and calls the tools in-process.
- `POLICYPILOT_API_URL` tells the UI where the API is (default `http://127.0.0.1:8000`).

The Qdrant collection name includes the embedding model and dimension. Changing either starts a fresh collection instead of mixing vectors that can't be compared.

