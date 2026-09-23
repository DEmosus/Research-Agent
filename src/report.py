"""
The final_answer tool and the report formatter that consumes its output.

final_answer is deliberately still a `@tool` even though it does no I/O: its
job is to give the oracle a structured *shape* to fill in (introduction,
research steps, main body, conclusion, sources) via tool-calling, the same
mechanism used for every other tool. Calling it is how the oracle signals
"I'm done researching" and hands back a structured answer instead of another
research step.
"""

from typing import List, Union

from langchain_core.tools import tool


def _as_bullets(value: Union[str, List[str]]) -> str:
    """Render a str-or-list field as either itself or a bullet list."""
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    return value


@tool
def final_answer(
    introduction: str,
    research_steps: Union[str, List[str]],
    main_body: str,
    conclusion: str,
    sources: Union[str, List[str]],
) -> str:
    """Returns a natural language response in the form of a research report.

    Args:
        introduction: A short paragraph introducing the user's question and the topic.
        research_steps: Bullet points or text explaining the steps taken for research.
        main_body: The bulk of the answer, 3-4 paragraphs long, providing high-quality information.
        conclusion: A short paragraph summarizing the findings.
        sources: A list or text providing the sources referenced during the research.

    Returns:
        A formatted research report string.
    """
    return (
        f"{introduction}\n\n"
        f"Research Steps:\n{_as_bullets(research_steps)}\n\n"
        f"Main Body:\n{main_body}\n\n"
        f"Conclusion:\n{conclusion}\n\n"
        f"Sources:\n{_as_bullets(sources)}"
    )


def build_report(output: dict) -> str:
    """Builds a nicely formatted report from the final_answer tool's arguments.

    Args:
        output: The dict of arguments the oracle passed to final_answer
            (introduction, research_steps, main_body, conclusion, sources) —
            i.e. `intermediate_steps[-1].tool_input` after the graph finishes.

    Returns:
        A formatted, human-readable research report string.
    """
    research_steps = _as_bullets(output["research_steps"])
    sources = _as_bullets(output["sources"])

    return f"""
        INTRODUCTION
        ------------
        {output['introduction']}

        RESEARCH STEPS
        --------------
        {research_steps}

        REPORT
        ------
        {output['main_body']}

        CONCLUSION
        ----------
        {output['conclusion']}

        SOURCES
        -------
        {sources}
    """
