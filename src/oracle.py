"""
The oracle: the decision-making core of the agent.

On every hop through the graph, the oracle looks at the question, the chat
history, and a "scratchpad" summarizing every tool call made so far, and
picks exactly one tool to call next (via OpenAI tool-calling with
`tool_choice="any"`, which forces it to always pick *some* tool rather than
replying in plain text). Repeat until it picks `final_answer`.

The LLM and the compiled oracle chain are built lazily (`get_llm()`,
`get_oracle()`, both cached) so this module — and the pure `router()`
function in particular — can be imported and unit-tested without an
OpenAI API key.
"""

from functools import lru_cache
from typing import List

from langchain_core.agents import AgentAction
from langchain_core.messages import ToolCall
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from . import config
from .report import final_answer
from .state import AgentState
from .tools import fetch_arxiv, rag_search, rag_search_filter, web_search

SYSTEM_PROMPT = """You are the oracle, the great AI decision-maker.
Given the user's query, you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (i.e., if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad), use the final_answer tool."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

# Every tool the oracle can choose between, and the lookup used to dispatch
# an actual call once the oracle has named one.
TOOLS = [rag_search_filter, rag_search, fetch_arxiv, web_search, final_answer]
TOOL_REGISTRY = {t.name: t for t in TOOLS}


def create_scratchpad(intermediate_steps: List[AgentAction]) -> str:
    """Render completed tool calls (log != 'TBD') into a readable log."""
    steps = [
        f"Tool: {action.tool}, input: {action.tool_input}\nOutput: {action.log}"
        for action in intermediate_steps
        if action.log != "TBD"
    ]
    return "\n---\n".join(steps)


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached ChatOpenAI instance, constructed on first use."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.OPENAI_CHAT_MODEL, temperature=config.OPENAI_CHAT_TEMPERATURE)


@lru_cache(maxsize=1)
def get_oracle():
    """Return the cached oracle chain: state dict -> AIMessage with a tool call."""
    return (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "scratchpad": lambda x: create_scratchpad(x["intermediate_steps"]),
        }
        | PROMPT
        | get_llm().bind_tools(TOOLS, tool_choice="any")
    )


def run_oracle(state: AgentState) -> dict:
    """Node: ask the oracle which tool to call next, record it as an AgentAction."""
    out = get_oracle().invoke(state)
    call: ToolCall = out.tool_calls[0]
    action = AgentAction(tool=call["name"], tool_input=call["args"], log="TBD")
    return {"intermediate_steps": [action]}


def router(state: AgentState) -> str:
    """Conditional edge out of 'oracle': route to whichever tool it named.

    Pure function of state — no model or network call — so it's tested
    directly in tests/test_graph.py without any mocking.
    """
    if isinstance(state["intermediate_steps"], list) and state["intermediate_steps"]:
        return state["intermediate_steps"][-1].tool
    return "final_answer"


def run_tool(state: AgentState) -> dict:
    """Node: execute whichever tool the most recent AgentAction named."""
    last_action = state["intermediate_steps"][-1]
    tool_name = last_action.tool
    tool_args = last_action.tool_input
    output = TOOL_REGISTRY[tool_name].invoke(input=tool_args)
    action = AgentAction(tool=tool_name, tool_input=tool_args, log=str(output))
    return {"intermediate_steps": [action]}
