# Research Agent

A ReAct-style research agent built with [LangGraph](https://langchain-ai.github.io/langgraph/)
and GPT-4o. Ask it a question and it decides for itself, one step at a
time, how to research the answer: searching a Pinecone-indexed corpus of
ArXiv papers, fetching a specific paper's abstract, or running a general
web search — looping until it has enough to write a structured report.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-informational)

## How it works

```
 START -> oracle --(router)--> rag_search_filter -> oracle
                    |---------> rag_search         -> oracle
                    |---------> fetch_arxiv         -> oracle
                    |---------> web_search           -> oracle
                    '---------> final_answer -> END
```

On every visit to `oracle`, an LLM looks at the question and a scratchpad of
every tool call made so far, and picks exactly one tool to call next. That
loop repeats — adaptively, not a fixed number of times — until the oracle
itself decides to call `final_answer`.

## Features

- **Adaptive ReAct loop** — the number and order of tool calls is decided
  by the model at runtime, not hardcoded.
- **Three information sources** — a Pinecone RAG index over ingested ArXiv
  papers, direct ArXiv abstract lookup, and general web search via SerpAPI.
- **A separate, idempotent ingestion pipeline** — `python -m src.ingest`
  builds and extends the knowledge base as its own batch job, independent
  of the query-time agent, and skips PDFs it's already downloaded.
- **Clean separation of concerns** — config, state, tools, the oracle's
  decision logic, graph wiring, and report formatting each live in their
  own module.
- **Testable without API keys** — every external client (OpenAI, Pinecone,
  SerpAPI) is built lazily, so the graph's wiring can be (and is) unit
  tested offline with mocks. See [`tests/`](tests/).
- **A real CLI** for both halves of the system — argument parsing, friendly
  errors for missing API keys or an unbuilt index, streamed progress, and a
  saved `.md` report per question.

## Project structure

```
.
├── .env.example              # Template for your local .env
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt           # Top-level dependencies
├── requirements-lock.txt      # Exact pinned versions (reproducible installs)
├── .github/workflows/ci.yml   # Runs the offline test suite on every push
├── docs/
│   ├── architecture.md        # How the agent loop and ingestion pipeline work
│   ├── decisions.md           # ADR-style log of engineering decisions
│   └── development-log.md    # Notebook to project: what changed, and why
├── src/
│   ├── __init__.py            # Public API: build_graph(), build_report(), AgentState
│   ├── config.py               # Every model name / index name / tunable, env-overridable
│   ├── state.py                # AgentState TypedDict
│   ├── tools.py                 # fetch_arxiv, web_search, rag_search, rag_search_filter
│   ├── report.py                # final_answer tool + build_report formatter
│   ├── oracle.py                 # System prompt, scratchpad, oracle chain, graph node fns
│   ├── graph.py                   # StateGraph assembly
│   ├── ingest.py                   # Data pipeline: ArXiv -> PDFs -> chunks -> Pinecone
│   └── main.py                     # CLI: ask the agent a question
├── tests/
│   ├── test_graph.py            # router() + a full mocked multi-hop graph run
│   ├── test_ingest.py            # expand_df's chunk-linking logic, mocked PDF loading
│   └── test_report.py            # final_answer / build_report formatting
├── data/                       # ArXiv metadata + downloaded PDFs (gitignored)
└── outputs/                    # Generated reports land here (gitignored)
```

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) (chat model + embeddings)
- A [Pinecone API key](https://app.pinecone.io) (vector index)
- A [SerpAPI key](https://serpapi.com) (web search)

## Installation

```bash
git clone git@github.com:DEmosus/Research-Agent.git
cd Research-Agent

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-lock.txt   # exact versions this was built and tested against
# or: pip install -r requirements.txt  # looser floors, gets you the latest compatible versions

cp .env.example .env
# then open .env and fill in OPENAI_API_KEY, PINECONE_API_KEY, and SERPAPI_KEY
```

## Usage

### 1. Build the knowledge base (once, before your first question)

```bash
# Default: 20 recent cs.AI papers
python -m src.ingest

# Customize the corpus
python -m src.ingest --query "cat:cs.CL" --max-results 50 --chunk-size 400
```

This fetches paper metadata, downloads PDFs into `data/`, chunks and embeds
them, and creates/populates a Pinecone index (`langgraph-research-agent` by
default). Re-run it any time to add more papers — already-downloaded PDFs
are skipped.

### 2. Ask the agent a question

```bash
python -m src.main "What are the main approaches to LLM agent memory?"

# Raise the safety cap on tool-call round-trips for a harder question
python -m src.main "Compare RAG and fine-tuning for domain adaptation" --max-iterations 15

# Quiet mode — just the final report, no step-by-step tool-call output
python -m src.main "Topic" --quiet
```

Every run prints each tool the oracle chooses and saves the final report as
a timestamped Markdown file under `outputs/`.

Run `python -m src.main --help` or `python -m src.ingest --help` for the
full list of flags.

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable                                   | Default                    | Purpose                                            |
| ------------------------------------------ | -------------------------- | -------------------------------------------------- |
| `OPENAI_API_KEY`                           | _(required)_               | OpenAI API key (chat + embeddings)                 |
| `PINECONE_API_KEY`                         | _(required)_               | Pinecone API key                                   |
| `SERPAPI_KEY`                              | _(required)_               | SerpAPI key                                        |
| `OPENAI_CHAT_MODEL`                        | `gpt-4o`                   | The oracle's chat model                            |
| `OPENAI_EMBED_MODEL`                       | `text-embedding-3-small`   | Embedding model for ingestion and RAG queries      |
| `PINECONE_INDEX_NAME`                      | `langgraph-research-agent` | Pinecone index name                                |
| `PINECONE_CLOUD` / `PINECONE_REGION`       | `aws` / `us-east-1`        | Pinecone serverless spec                           |
| `SERPAPI_NUM_RESULTS`                      | `5`                        | Web search results per query                       |
| `RAG_TOP_K` / `RAG_FILTERED_TOP_K`         | `5` / `6`                  | Matches returned by rag_search / rag_search_filter |
| `MAX_AGENT_ITERATIONS`                     | `10`                       | Safety cap on oracle<->tool round-trips            |
| `ARXIV_SEARCH_QUERY` / `ARXIV_MAX_RESULTS` | `cat:cs.AI` / `20`         | Default ingestion corpus                           |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`             | `512` / `64`               | PDF chunking parameters                            |
| `EMBED_BATCH_SIZE`                         | `64`                       | Chunks embedded/upserted per batch                 |

## Running tests

```bash
pip install -r requirements-lock.txt   # includes pytest
python -m pytest tests/ -v
```

No test calls OpenAI, Pinecone, SerpAPI, or arxiv.org — the oracle and every
tool are mocked where needed, and the ingestion pipeline's PDF-chunking
logic is tested with a mocked loader. The suite runs offline, in CI, with no
API keys and no cost. See `.github/workflows/ci.yml`.

## Using it as a library

```python
from src import build_graph, build_report

graph = build_graph()
result = graph.invoke({
    "input": "What's new in retrieval-augmented generation?",
    "chat_history": [],
    "intermediate_steps": [],
})
report = build_report(result["intermediate_steps"][-1].tool_input)
print(report)
```

## Roadmap ideas

- Multi-turn conversation — `chat_history` is already threaded through
  `AgentState`; the CLI just never populates it across separate runs.
- A `prechunk`/`postchunk` expansion tool, using the sibling-chunk IDs
  `ingest.py` already stores but nothing currently reads.
- Streaming the oracle's tool-call reasoning to the CLI token-by-token
  instead of only after each step completes.

## License

[MIT](LICENSE)

## Acknowledgments

The agent design (an oracle choosing tools in a loop, terminating via a
`final_answer` tool) follows the pattern taught in DeepLearning.AI's _AI
Agents in LangGraph_ course. This repository is an original, from-scratch
reimplementation and restructuring of that pattern into a tested,
documented, standalone project.
