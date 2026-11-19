"""CLI:  python -m agent "Review https://github.com/pallets/markupsafe" [options]"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .faults import inject_faults
from .llm import make_llm
from .loop import run_agent
from .models import LLMError
from .report import to_markdown
from .tools import GitSource, build_registry
from .trace import Tracer


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agent", description="Autonomous, LLM-driven GitHub repo code-quality reviewer")
    p.add_argument("goal", help="natural-language goal, e.g. 'Review https://github.com/pallets/markupsafe and suggest fixes'")
    p.add_argument("--provider", choices=["anthropic", "openai"], default=None,
                   help="model provider (default: auto-detect from ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    p.add_argument("--model", default=None, help="model id (or set AGENT_MODEL)")
    p.add_argument("--inject-fault", action="append", default=[], metavar="TOOL:MODE[:N]",
                   help="sabotage a tool to demo recovery (modes: timeout, rate_limit, malformed, not_found)")
    p.add_argument("--out", default="runs/latest", help="directory for trace.jsonl, report.json, report.md, transcript.txt")
    p.add_argument("--max-turns", type=int, default=30, help="model turn budget")
    p.add_argument("--backoff", type=float, default=1.0, help="base retry delay in seconds (doubles each retry)")
    p.add_argument("--timeout", type=float, default=90.0, help="per tool-call timeout in seconds")
    p.add_argument("--no-cache", action="store_true", help="clone into a fresh temp dir instead of reusing .cache/repos")
    p.add_argument("--quiet", action="store_true", help="suppress live trace on stdout")
    a = p.parse_args(argv)

    try:
        llm = make_llm(a.provider, a.model)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(out / "trace.jsonl", quiet=a.quiet)
    cache = Path(tempfile.mkdtemp(prefix="repo-agent-")) if a.no_cache else Path(".cache/repos")
    registry = build_registry(GitSource(cache_dir=cache), a.timeout)
    try:
        if a.inject_fault:
            inject_faults(registry, a.inject_fault)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report = run_agent(a.goal, llm, registry, tracer, max_turns=a.max_turns, backoff=a.backoff, tool_timeout=a.timeout)
    tracer.close()
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(report) if "findings" in report or report["status"] == "needs_clarification"
                                   else f"# Run failed\n\n{report.get('error')}\n")
    (out / "transcript.txt").write_text("\n".join(tracer.lines) + "\n")
    print(f"\nWrote {out}/report.md, report.json, trace.jsonl, transcript.txt")
    return 0 if report["status"] in ("success", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
