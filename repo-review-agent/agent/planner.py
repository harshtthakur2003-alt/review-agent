"""Planner: goal -> validated Plan. Invalid LLM plans are fed back once for self-correction."""
from __future__ import annotations

import re
from typing import Optional

from .llm import LLM, OfflineLLM
from .models import Plan, Step
from .tools.base import ToolRegistry
from .trace import Tracer

_REF = re.compile(r"\$\{?(s\d+|item)\b")


def validate_plan(raw: dict, registry: ToolRegistry) -> Plan:
    """Raises ValueError with an actionable message if the plan is unusable."""
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("plan has no steps")
    steps: list[Step] = []
    seen: set[str] = set()
    for s in steps_raw:
        sid, tool = s.get("id"), s.get("tool")
        if not sid or sid in seen:
            raise ValueError(f"missing or duplicate step id {sid!r}")
        if tool not in registry.names():
            raise ValueError(f"step {sid} uses unknown tool {tool!r}; available: {registry.names()}")
        args = s.get("args") or {}
        refs = set(_REF.findall(str(args) + str(s.get("foreach") or "")))
        for ref in refs - {"item"}:
            if ref not in seen:
                raise ValueError(f"step {sid} references {ref}, which does not run before it")
        if "item" in refs and not s.get("foreach"):
            raise ValueError(f"step {sid} uses $item but has no foreach")
        seen.add(sid)
        steps.append(Step(sid, s.get("description", ""), tool, args, s.get("foreach"),
                          s.get("rationale", ""), bool(s.get("optional", False))))
    return Plan(goal=raw.get("goal", ""), reasoning=raw.get("reasoning", ""), steps=steps,
                assumptions=list(raw.get("assumptions", [])))


def make_plan(goal: str, llm: LLM, registry: ToolRegistry, tracer: Tracer, max_attempts: int = 2) -> Plan:
    """May raise Clarification. Falls back to the deterministic template if the LLM keeps failing."""
    desc = registry.describe()
    feedback: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        raw = llm.plan(goal if not feedback else f"{goal}\n\nYour previous plan was rejected: {feedback}", desc)
        raw["goal"] = goal
        try:
            plan = validate_plan(raw, registry)
            break
        except ValueError as e:
            feedback = str(e)
            tracer.emit("error", f"plan rejected (attempt {attempt}): {e}")
    else:
        tracer.emit("recover", "LLM plan invalid twice; falling back to deterministic template plan")
        raw = OfflineLLM().plan(goal, desc)
        raw["goal"] = goal
        plan = validate_plan(raw, registry)

    tracer.emit("plan", f"Reasoning: {plan.reasoning}")
    for a in plan.assumptions:
        tracer.emit("plan", f"Assumption: {a}")
    for s in plan.steps:
        loop = f" (for each in {s.foreach})" if s.foreach else ""
        tracer.emit("plan", f"{s.id}. {s.description} -> {s.tool}{loop}")
    return plan
