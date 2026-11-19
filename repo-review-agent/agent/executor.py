"""Executor: runs a Plan step by step with retries, fallbacks, LLM repair and graceful degradation."""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from .llm import LLM
from .models import Plan, Step, StepResult, ToolError
from .tools.base import ToolRegistry
from .trace import Tracer

_WHOLE = re.compile(r"^\$([A-Za-z_]\w*(?:\.\w+|\[\d+\])*)$")
_EMBED = re.compile(r"\$\{([A-Za-z_]\w*(?:\.\w+|\[\d+\])*)\}")
_PATH_TOKEN = re.compile(r"\.(\w+)|\[(\d+)\]")
_DEP = re.compile(r"\$\{?(s\d+)\b")

TRANSIENT_KINDS = ("transient", "rate_limited", "invalid_output")


def lookup(path: str, ctx: dict) -> Any:
    root, rest = re.match(r"([A-Za-z_]\w*)(.*)", path).groups()
    if root not in ctx:
        raise ToolError("invalid_input", f"unresolved reference '{root}' (step did not produce output)")
    cur = ctx[root]
    for key, idx in _PATH_TOKEN.findall(rest):
        try:
            cur = cur[int(idx)] if idx else cur[key]
        except (KeyError, IndexError, TypeError):
            raise ToolError("invalid_input", f"reference '${path}' cannot be resolved at '{key or idx}'")
    return cur


def resolve(value: Any, ctx: dict) -> Any:
    if isinstance(value, str):
        m = _WHOLE.match(value)
        if m:
            return lookup(m.group(1), ctx)
        return _EMBED.sub(lambda mm: str(lookup(mm.group(1), ctx)), value)
    if isinstance(value, dict):
        return {k: resolve(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, ctx) for v in value]
    return value


class Executor:
    def __init__(self, registry: ToolRegistry, llm: LLM, tracer: Tracer, max_attempts: int = 4,
                 backoff: float = 0.5, tool_timeout: float = 90.0):
        self.reg, self.llm, self.t = registry, llm, tracer
        self.max_attempts, self.backoff, self.tool_timeout = max_attempts, backoff, tool_timeout

    # ---- single tool call with the full recovery ladder ----------------------------------
    def call_with_recovery(self, tool: str, args: dict, label: str) -> tuple[Any, int, list[str], list[str]]:
        errors: list[str] = []
        recoveries: list[str] = []
        attempts = 0
        tried_fallback = False
        tried_llm = False
        while True:
            attempts += 1
            self.t.emit("call", f"{label} {tool}({self.t.short(args, 110)})  [attempt {attempts}]")
            try:
                out = self.reg.call(tool, args, self.tool_timeout)
                self.t.emit("result", f"{label} {tool} -> {self.t.short(out, 110)}")
                return out, attempts, errors, recoveries
            except ToolError as e:
                errors.append(str(e))
                self.t.emit("error", f"{label} {tool}: {e}")
                e.attempts, e.history, e.recoveries = attempts, errors, recoveries  # keep audit trail on give-up
                if attempts >= self.max_attempts + 2:
                    raise
                if e.kind in TRANSIENT_KINDS and attempts <= self.max_attempts - 1:
                    delay = self.backoff * (2 ** (attempts - 1))
                    msg = f"{e.kind} error -> retry in {delay:.1f}s (exponential backoff)"
                    recoveries.append(msg)
                    self.t.emit("recover", f"{label} {msg}")
                    time.sleep(delay)
                    continue
                spec = self.reg.get(tool)
                if e.kind in ("not_found", "invalid_input") and spec.fallback and not tried_fallback:
                    tried_fallback = True
                    try:
                        fb = spec.fallback(args, e, self.reg)
                    except ToolError as fe:
                        fb = None
                        self.t.emit("error", f"{label} fallback itself failed: {fe}")
                    if fb:
                        args, why = fb
                        recoveries.append(f"fallback: {why}")
                        self.t.emit("recover", f"{label} fallback: {why}")
                        continue
                if e.kind in ("not_found", "invalid_input") and not tried_llm:
                    tried_llm = True
                    fix = self.llm.repair({"tool": tool, "args": args}, str(e), self.reg.describe())
                    if fix and isinstance(fix.get("args"), dict):
                        args = fix["args"]
                        why = f"LLM repair: {fix.get('why', 'adjusted args')}"
                        recoveries.append(why)
                        self.t.emit("recover", f"{label} {why}")
                        continue
                e.attempts, e.history, e.recoveries = attempts, errors, recoveries
                raise

    # ---- one plan step (optionally fanned out) -------------------------------------------
    def run_step(self, step: Step, ctx: dict) -> StepResult:
        res = StepResult(step.id, "ok")
        if step.foreach is None:
            try:
                out, res.attempts, res.errors, res.recoveries = self.call_with_recovery(
                    step.tool, resolve(step.args, ctx), step.id)
            except ToolError as e:
                res.status, res.note = "failed", f"{e}"
                res.attempts = getattr(e, "attempts", 1)
                res.errors, res.recoveries = list(getattr(e, "history", [str(e)])), list(getattr(e, "recoveries", []))
                self.t.emit("error", f"{step.id} FAILED: {e}")
                return res
            res.output = out
            return res

        items = resolve(step.foreach, ctx)
        oks, fails = [], []
        for n, item in enumerate(items, 1):
            sub = f"{step.id}[{n}/{len(items)}]"
            try:
                out, attempts, errs, recs = self.call_with_recovery(step.tool, resolve(step.args, {**ctx, "item": item}), sub)
                oks.append(out)
                res.attempts += attempts
                res.errors += errs
                res.recoveries += recs
            except ToolError as e:
                fails.append({"item": item, "error": str(e)})
                res.attempts += getattr(e, "attempts", 1)
                res.errors += [f"{sub}: {x}" for x in getattr(e, "history", [str(e)])]
                res.recoveries += getattr(e, "recoveries", [])
                self.t.emit("error", f"{sub} gave up: {e} (continuing with remaining items)")
        res.output = {"results": oks, "failures": fails}
        if not oks:
            res.status, res.note = "failed", f"all {len(items)} items failed"
        elif fails:
            res.status, res.note = "partial", f"{len(fails)}/{len(items)} items failed and were excluded"
        return res

    def run(self, plan: Plan) -> list[StepResult]:
        ctx: dict[str, Any] = {}
        results: list[StepResult] = []
        status: dict[str, str] = {}
        for step in plan.steps:
            self.t.emit("step", f"{step.id}: {step.description}")
            broken = sorted({d for d in _DEP.findall(str(step.args) + str(step.foreach or "")) if status.get(d) in ("failed", "skipped")})
            if broken:
                r = StepResult(step.id, "skipped", note=f"depends on failed/skipped step(s) {broken}")
                self.t.emit("skip", f"{step.id} skipped: {r.note}")
            else:
                r = self.run_step(step, ctx)
            status[step.id] = r.status
            if r.status in ("ok", "partial"):
                ctx[step.id] = r.output
            results.append(r)
        return results
