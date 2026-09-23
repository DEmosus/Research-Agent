"""
Graph assembly for the research agent.

ReAct-style loop: the oracle picks one tool, that tool runs, and control
returns to the oracle to pick again — until it picks `final_answer`, the
only tool with an edge to END instead of back to `oracle`.

    START -> oracle --(router)--> rag_search_filter -> oracle
                       |--------> rag_search         -> oracle
                       |--------> fetch_arxiv         -> oracle
                       |--------> web_search           -> oracle
                       '--------> final_answer -> END

Every tool node runs the *same* function, `run_tool` — it looks up which
tool to actually call from the state's own `intermediate_steps`, so one
node function serves all five tool nodes.
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .oracle import TOOLS, router, run_oracle, run_tool
from .state import AgentState


def build_graph() -> CompiledStateGraph:
    """Assemble and compile the research-agent StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("oracle", run_oracle)
    for t in TOOLS:
        builder.add_node(t.name, run_tool)

    builder.set_entry_point("oracle")
    path_map = {t.name: t.name for t in TOOLS}
    builder.add_conditional_edges("oracle", router, path_map)

    for t in TOOLS:
        if t.name != "final_answer":
            builder.add_edge(t.name, "oracle")
    builder.add_edge("final_answer", END)

    return builder.compile()
