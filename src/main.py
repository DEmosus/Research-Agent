"""
Command-line entry point for asking the research agent a question.

Usage:
    python -m src.main "What is the future of LLM agents?"

Run `python -m src.main --help` for all options.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .graph import build_graph
from .report import build_report

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY", "SERPAPI_KEY"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Ask the research agent a question; it researches (RAG + ArXiv + web) and reports back.",
    )
    parser.add_argument("question", help="The question to research.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=config.MAX_AGENT_ITERATIONS,
        help=f"Safety cap on oracle<->tool round-trips (default: {config.MAX_AGENT_ITERATIONS}).",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help="Directory to save the report into (default: outputs/).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step tool-call output; only print the final report.",
    )
    return parser.parse_args(argv)


def check_environment() -> None:
    """Fail fast with a clear message if required API keys are missing."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        print(
            "Missing required environment variable(s): " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Copy .env.example to .env and fill in your API keys, then try again.",
            file=sys.stderr,
        )
        sys.exit(1)


def run(question: str, max_iterations: int, quiet: bool) -> dict:
    """Run the graph to completion and return the final state."""
    graph = build_graph()
    initial_state = {"input": question, "chat_history": [], "intermediate_steps": []}
    config_dict = {"recursion_limit": max_iterations * 2 + 5}

    final_state: dict = {}
    for event in graph.stream(initial_state, config_dict):
        for node_name, node_output in event.items():
            final_state = {**final_state, **node_output}
            if not quiet:
                action = node_output["intermediate_steps"][-1]
                if action.log == "TBD":
                    print(
                        f"  [oracle] decided to call: {action.tool}({action.tool_input})"
                    )
                else:
                    preview = (
                        action.log
                        if len(action.log) <= 160
                        else action.log[:160] + "..."
                    )
                    print(f"  [{action.tool}] -> {preview}")
    return final_state


def save_report(report: str, question: str, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_q = "-".join(question.lower().split())[:40]
    out_path = out_dir / f"{timestamp}_{safe_q}.md"
    out_path.write_text(f"# {question}\n\n{report}\n", encoding="utf-8")
    return out_path


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    check_environment()

    print(f'Researching: "{args.question}"\n')

    try:
        final_state = run(args.question, args.max_iterations, args.quiet)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - surface any failure clearly to the CLI user
        print(f"\nThe agent failed: {exc}", file=sys.stderr)
        return 1

    intermediate_steps = final_state.get("intermediate_steps", [])
    if not intermediate_steps or intermediate_steps[-1].tool != "final_answer":
        print(
            "\nThe agent stopped without producing a final answer "
            "(likely hit --max-iterations). Try raising it or narrowing the question."
        )
        return 1

    report = build_report(intermediate_steps[-1].tool_input)
    out_path = save_report(report, args.question, args.output_dir)

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
