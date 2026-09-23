"""
Tests for the research-agent graph.

These tests never call OpenAI, Pinecone, or SerpAPI: `router` is tested
directly as a pure function, and `test_full_graph_run_with_mocks` patches
`src.oracle.get_oracle` and every tool's `.invoke` so the whole graph
(oracle -> tool -> oracle -> ... -> final_answer -> END) can be exercised
end to end, offline, with no API keys.

Run with: python -m pytest tests/ -v
"""

from unittest.mock import MagicMock

from langchain_core.agents import AgentAction

from src.graph import build_graph
from src.oracle import TOOLS, router


def _action(tool, tool_input=None, log="TBD"):
    return AgentAction(tool=tool, tool_input=tool_input or {}, log=log)


def test_router_routes_to_the_named_tool():
    state = {"intermediate_steps": [_action("web_search")]}
    assert router(state) == "web_search"


def test_router_routes_to_final_answer_on_empty_steps():
    assert router({"intermediate_steps": []}) == "final_answer"
    assert router({"intermediate_steps": "not-a-list"}) == "final_answer"


def test_router_uses_the_most_recent_action():
    state = {"intermediate_steps": [_action("rag_search"), _action("fetch_arxiv")]}
    assert router(state) == "fetch_arxiv"


def _mock_ai_message(tool_name, args):
    """Build a fake AIMessage-like object with the .tool_calls shape run_oracle expects."""
    msg = MagicMock()
    msg.tool_calls = [{"name": tool_name, "args": args}]
    return msg


def test_full_graph_run_with_mocks(monkeypatch):
    """oracle picks web_search, then final_answer; graph should stop at END
    with the report fields intact in the last AgentAction's tool_input."""
    fake_oracle = MagicMock()
    fake_oracle.invoke.side_effect = [
        _mock_ai_message("web_search", {"query": "test topic"}),
        _mock_ai_message(
            "final_answer",
            {
                "introduction": "Intro.",
                "research_steps": ["Searched the web for 'test topic'."],
                "main_body": "Body.",
                "conclusion": "Conclusion.",
                "sources": ["example.com"],
            },
        ),
    ]
    monkeypatch.setattr("src.oracle.get_oracle", lambda: fake_oracle)

    fake_web_search = MagicMock()
    fake_web_search.invoke.return_value = "Result: some web content."
    monkeypatch.setattr(
        "src.oracle.TOOL_REGISTRY",
        {
            **{t.name: t for t in TOOLS},
            "web_search": fake_web_search,
        },
    )

    graph = build_graph()
    final_state = graph.invoke(
        {
            "input": "Tell me about test topic",
            "chat_history": [],
            "intermediate_steps": [],
        }
    )

    steps = final_state["intermediate_steps"]
    assert steps[0].tool == "web_search"
    assert steps[1].tool == "web_search"  # the executed result, log filled in
    assert steps[1].log == "Result: some web content."
    assert steps[-1].tool == "final_answer"
    assert steps[-1].tool_input["introduction"] == "Intro."
    assert fake_oracle.invoke.call_count == 2
