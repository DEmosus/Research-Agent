# Architecture

## Overview

The Research Agent answers a question by researching it first: it can search
a Pinecone-indexed corpus of ArXiv papers, fetch a specific paper's abstract,
or run a general web search — deciding for itself, one step at a time, which
tool to reach for and when it has gathered enough to answer. This is a
**ReAct-style agent loop**, not a fixed pipeline: unlike a linear sequence of
steps, the number of tool calls and their order is chosen at runtime by an
LLM ("the oracle"), and the loop only ends when the oracle itself decides to
call `final_answer`.

There are two independent halves to this project:

1. **The ingestion pipeline** (`src/ingest.py`) — an offline batch job that
   builds the knowledge base: fetch ArXiv paper metadata, download PDFs,
   chunk them, embed each chunk, and upsert into Pinecone. Run this once
   (and again whenever you want to add more papers) before asking questions.
2. **The agent** (`src/graph.py`, `src/oracle.py`, `src/tools.py`,
   `src/report.py`) — the online path: given a question, loop between the
   oracle and its tools until a report can be produced.

## The agent loop

```
 START -> oracle --(router)--> rag_search_filter -> oracle
                    |---------> rag_search         -> oracle
                    |---------> fetch_arxiv         -> oracle
                    |---------> web_search           -> oracle
                    '---------> final_answer -> END
```

On every visit to `oracle`, the LLM is given the original question, the
chat history, and a **scratchpad**: a plain-text log of every tool call made
so far and what it returned. Using OpenAI tool-calling with
`tool_choice="any"` (which forces the model to pick some tool rather than
reply in free text), the oracle names exactly one tool to call next. That
choice is recorded, the named tool actually runs, and control returns to the
oracle — which now sees the updated scratchpad — to decide again. The loop
ends the moment the oracle names `final_answer`, which has an edge straight
to `END` instead of back to `oracle`.

This shape is what makes the agent adaptive: a broad question might take
four tool calls across all three information sources before the oracle is
satisfied; a narrow one might take one. Nothing about the *number* of steps
is hardcoded — the system prompt's own instructions ("don't use a tool
more than twice", "collect information from a diverse range of sources
before answering") are what actually bound the behavior, backed by a hard
`recursion_limit` at the LangGraph level as a last-resort safety net (see
"Safety: the recursion limit" below).

## The shared state (`src/state.py`)

```python
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[AgentAction], operator.add]
```

`intermediate_steps` is the scratchpad in data form: every `AgentAction` the
oracle has decided on, in order. It's declared with LangGraph's
`operator.add` reducer, which changes how updates are merged: instead of a
node's return value *replacing* the field (the default), a node returning
`{"intermediate_steps": [one_new_action]}` **appends** that one action to
the existing list. That's what lets `run_oracle` and `run_tool` each add
exactly one entry per hop without either of them needing to know — or
re-supply — the full history so far.

Each `AgentAction` goes through two states: `run_oracle` creates it with
`log="TBD"` (the decision, not yet executed); `run_tool` immediately
appends a second `AgentAction` for the *same* tool and args, this time with
`log` set to the tool's actual string output. `create_scratchpad` (in
`oracle.py`) filters to only the completed ones (`log != "TBD"`) when
rendering the log the oracle reads — so the oracle never sees its own
most recent, not-yet-executed decision reflected back as if it were a
completed step.

## Tools (`src/tools.py`, `src/report.py`)

| Tool | Purpose | Backing service |
|---|---|---|
| `rag_search` | Broad semantic search over every indexed paper chunk | Pinecone + OpenAI embeddings |
| `rag_search_filter` | Semantic search scoped to one paper by ArXiv ID | Pinecone + OpenAI embeddings |
| `fetch_arxiv` | Fetch a specific paper's abstract by ArXiv ID | arxiv.org (HTML scrape) |
| `web_search` | General-knowledge Google search | SerpAPI |
| `final_answer` | Not really a "tool" in the I/O sense — its call *is* the answer | none (pure formatting) |

`final_answer` is deliberately implemented as an `@tool` even though it
performs no I/O. Its real job is to give the oracle a **structured shape**
to fill in — `introduction`, `research_steps`, `main_body`, `conclusion`,
`sources` — via the same tool-calling mechanism used for every other tool,
rather than asking for free-form prose that would then need to be parsed.
Calling `final_answer` is simultaneously how the oracle signals "I'm done
researching" *and* how it hands back a report with a guaranteed shape.
`src/report.py`'s `build_report()` then renders those five fields into the
human-readable report the CLI prints and saves.

As in the essay-writer project, every external client — the OpenAI chat
model, the embedding encoder, the Pinecone index handle, the SerpAPI
params — is built lazily via a cached `get_*()` factory rather than at
import time. This means `src/tools.py`, `src/oracle.py`, and `src/graph.py`
can all be imported, and the graph built and introspected, without any API
keys set — which is what lets `tests/test_graph.py` exercise the whole
oracle/tool loop offline.

## Safety: the recursion limit

The system prompt asks the oracle not to repeat a tool call and to move to
`final_answer` once it has enough information — but a prompt is an
instruction, not a guarantee. `src/main.py` sets LangGraph's
`recursion_limit` (derived from `--max-iterations`, doubled plus a small
margin to account for each round-trip being two graph steps: one for
`oracle`, one for the tool) so a model that somehow never calls
`final_answer` fails loudly with a clear, catchable error instead of
looping indefinitely and burning API calls.

## The ingestion pipeline (`src/ingest.py`)

```
extract_from_arxiv -> download_pdfs -> expand_df (chunk) -> ensure_pinecone_index -> populate_index
```

1. **extract_from_arxiv** — queries the ArXiv API (`export.arxiv.org`) and
   saves paper metadata (title, summary, authors, ArXiv ID, PDF link) as
   JSON.
2. **download_pdfs** — downloads each paper's PDF, skipping any already on
   disk so re-running ingestion doesn't re-fetch everything.
3. **expand_df** — for each paper, loads its PDF (`PyPDFLoader`) and splits
   it into overlapping chunks (`RecursiveCharacterTextSplitter`), producing
   one output row per chunk. Each chunk gets an `id` of the form
   `{arxiv_id}#{chunk_index}` and `prechunk_id`/`postchunk_id` fields
   linking it to its neighbors within the same paper — useful metadata for
   a future "expand context" tool, even though the current RAG tools only
   use the chunk text itself.
4. **ensure_pinecone_index** — creates the Pinecone index if it doesn't
   already exist, sized to the embedding model's actual output dimension
   (computed by embedding one real chunk rather than hardcoding a number
   that would silently drift if the embedding model were changed).
5. **populate_index** — embeds each batch of chunks and upserts
   `(id, embedding, metadata)` triples into Pinecone, where `metadata`
   carries exactly what `format_rag_contexts` needs at query time:
   `arxiv_id`, `title`, `chunk`.

This is a genuinely separate concern from the agent: it's a batch job you
run occasionally to build or extend the knowledge base, not something that
runs per-question. See `docs/decisions.md` for why it's a separate CLI
(`python -m src.ingest`) rather than something the agent triggers itself.
