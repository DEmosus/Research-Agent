"""
Research Agent
===============

A ReAct-style research agent built with LangGraph, GPT-4o, and
retrieval-augmented generation over a Pinecone-indexed corpus of ArXiv
papers, with ArXiv-abstract and Google-web-search tools rounding out its
sources. See docs/architecture.md for the full design.

Public API:
    build_graph()  -> a compiled, runnable LangGraph graph (the agent).
    build_report()  -> formats the agent's final_answer output as a report.
    AgentState       -> the shared state type the graph operates on.
"""

from .graph import build_graph
from .report import build_report
from .state import AgentState

__all__ = ["build_graph", "build_report", "AgentState"]
__version__ = "1.0.0"
