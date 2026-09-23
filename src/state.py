"""
Shared state for the research-agent graph.

Every node receives this state and returns a partial dict that LangGraph
merges back in. `intermediate_steps` is special: it's declared with the
`operator.add` reducer, so returning a single-item list from a node
*appends* to the running list instead of overwriting it — that's what lets
`oracle` and each tool each contribute one more entry to the scratchpad on
every hop through the loop, rather than clobbering what came before.
"""

import operator
from typing import Annotated, List, TypedDict

from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across every node in the research-agent graph.

    Attributes:
        input: The user's question.
        chat_history: Prior conversation turns, if any (empty for a single
            one-shot question, which is all the CLI currently sends).
        intermediate_steps: The running scratchpad of every AgentAction the
            oracle has decided on and every tool has executed, in order.
            Accumulates via `operator.add` rather than being overwritten.
    """

    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[AgentAction], operator.add]
