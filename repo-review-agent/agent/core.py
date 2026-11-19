"""Top-level orchestration: plan -> execute -> report."""
from __future__ import annotations

from typing import Optional

from .executor import Executor
from .llm import LLM
from .models import Clarification
from .planner import make_plan
from .report import build_report
from .tools.base import ToolRegistry
from .trace import Tracer


def run_agent(goal: str, llm: LLM, registry: ToolRegistry, tracer: Tracer, backoff: float = 0.5,
              tool_timeout: float = 90.0) -> dict:
    tracer.emit("info", f"Goal: {goal}")
    tracer.emit("info", f"Backend: {llm.name}; tools: {', '.join(registry.names())}")
    try:
        plan = make_plan(goal, llm, registry, tracer)
    except Clarification as c:
        tracer.emit("report", f"Cannot plan: {c}")
        return build_report(goal, None, [], llm.name, clarification=str(c))
    results = Executor(registry, llm, tracer, backoff=backoff, tool_timeout=tool_timeout).run(plan)
    report = build_report(goal, plan, results, llm.name)
    try:
        facts = {k: report[k] for k in ("repository", "quality_score", "metrics", "worst_files") if report.get(k)}
        report["summary"] = llm.narrate(facts) if report["metrics"] else None
    except Exception as e:  # narration is optional; never fail the run because of it
        tracer.emit("error", f"narration skipped: {e}")
    tracer.emit("report", f"Finished with status '{report['status']}'"
                + (f", quality score {report['quality_score']}" if report["quality_score"] is not None else ""))
    return report
