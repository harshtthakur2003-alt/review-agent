"""The agent: a model-driven observe-think-act loop.

The model decides everything: it must first publish a plan (`make_plan`), then picks tools one step at a
time, reads real results, revises its plan when something fails (`make_plan` again), asks for clarification
if the goal is unusable (`ask_user`), and finally authors the structured report (`submit_report`).

Code in this file only provides: the tool plumbing, transient-failure retries with backoff, guard-rails
(turn budget, repeated-failure warnings, "you must finish" nudges), tracing, and report validation.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .llm import LLMClient
from .models import LLMError, ToolError
from .report import REPORT_SCHEMA, REQUIRED, validate_report
from .tools.base import Tool, ToolRegistry
from .trace import Tracer

SYSTEM_PROMPT = """You are an autonomous software-engineering agent. The user gives you a high-level goal about a GitHub repository.
You complete it on your own, using tools against live data (nothing is pre-loaded).

How you must work:
1. FIRST call make_plan: state your reasoning and an ordered list of concrete steps. Keep it short and bounded.
2. Execute step by step. After every tool result, decide what to do next based on what you actually observed.
3. Be economical: sample the most important files rather than reading everything. Read real code before judging it;
   treat static-analysis output as a hint to verify, not as truth. Every finding needs evidence you actually saw.
4. If a tool fails or returns something unexpected, diagnose it, then try a different approach (fix arguments,
   fix a misspelled name, use another tool, narrow the scope). If your approach changes materially, call make_plan
   again with the revised plan. Never repeat an identical failing call more than once.
5. If the goal does not identify a repository you can review, call ask_user instead of guessing.
6. Finish by calling submit_report exactly once. Be honest: if something could not be done, use status 'partial' or
   'failed', and list what went wrong in 'recoveries' and 'limitations'. Record guesses as 'assumptions'.
Older tool outputs may later be shortened to save context, so when you see important evidence, note the path:line
and what it shows in your own message before moving on.
Do not write the report as plain text; the structured submit_report call is the deliverable."""

MAX_RESULT_CHARS = 7000
HINTS = {
    "not_found": "The thing you asked for does not exist. Check spelling/paths; repo_search or repo_list_files can help.",
    "invalid_input": "Your arguments were rejected. Re-read the tool schema and fix the call.",
    "invalid_output": "The tool returned a malformed response (retries exhausted). Try another approach.",
    "transient": "A temporary failure persisted after automatic retries.",
    "rate_limited": "Rate limited even after backoff; consider a different tool or reporting the limitation.",
    "fatal": "Unexpected failure. Try an alternative or report the limitation.",
}


def _meta_tools() -> list[dict]:
    return [
        {"name": "make_plan", "description": "Publish (or revise) your plan. Call this first, and again whenever your approach changes.",
         "input_schema": {"type": "object", "properties": {
             "reasoning": {"type": "string", "description": "how you interpret the goal and why this approach"},
             "steps": {"type": "array", "items": {"type": "string"}},
             "assumptions": {"type": "array", "items": {"type": "string"}},
             "revision_reason": {"type": "string", "description": "only when revising: what went wrong"}},
             "required": ["reasoning", "steps"]}},
        {"name": "ask_user", "description": "Use ONLY if the goal does not say which repository to review. Never use it to ask for access or files: you already have tools that fetch any public repository. Ends the run.",
         "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
        {"name": "submit_report", "description": "Submit the final structured report. Ends the run.",
         "input_schema": {"type": "object", "properties": REPORT_SCHEMA, "required": REQUIRED}},
    ]


class Agent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, tracer: Tracer, max_turns: int = 30,
                 backoff: float = 1.0, tool_timeout: float = 90.0, retries: int = 3, context_chars: int = 60000):
        self.llm, self.reg, self.t = llm, registry, tracer
        self.context_chars = max(4000, context_chars)
        self.result_chars = min(MAX_RESULT_CHARS, self.context_chars // 3)
        self.max_turns, self.backoff, self.tool_timeout, self.retries = max_turns, backoff, tool_timeout, retries
        self.meta = _meta_tools()
        self.plans: list[dict] = []
        self.tool_log: list[dict] = []
        self.usage = {"in": 0, "out": 0}
        self._failed_sigs: dict[str, int] = {}

    # ---- tools: infrastructure-level retry only; reasoning about failures is the model's job --------------
    def _run_tool(self, name: str, args: dict) -> tuple[Any, dict]:
        entry = {"tool": name, "args": args, "status": "ok", "attempts": 0, "error": None}
        self.tool_log.append(entry)
        for attempt in range(1, self.retries + 1):
            entry["attempts"] = attempt
            self.t.emit("call", f"{name}({self.t.short(args, 120)})" + (f"  [attempt {attempt}]" if attempt > 1 else ""))
            try:
                out = self.reg.call(name, args, self.tool_timeout)
                self.t.emit("result", f"{name} -> {self.t.short(out, 120)}")
                entry["status"], entry["error"] = "ok", None
                return out, entry
            except ToolError as e:
                entry["status"], entry["error"] = e.kind, str(e)
                self.t.emit("error", f"{name}: {e}")
                if e.kind in ("transient", "rate_limited", "invalid_output") and attempt < self.retries:
                    delay = self.backoff * 2 ** (attempt - 1)
                    self.t.emit("recover", f"{e.kind} -> automatic retry in {delay:.1f}s (exponential backoff)")
                    time.sleep(delay)
                    continue
                raise
        raise AssertionError("unreachable")

    def _tool_result(self, call, msgs_note: list) -> dict:
        sig = f"{call.name}:{json.dumps(call.args, sort_keys=True, default=str)}"
        try:
            if "__malformed_json__" in call.args:
                raise ToolError("invalid_input", "your tool arguments were not valid JSON; resend the call")
            out, _ = self._run_tool(call.name, call.args)
            text = json.dumps(out, default=str)
            if len(text) > self.result_chars:
                text = text[:self.result_chars] + f'... [truncated {len(text) - self.result_chars} chars; narrow your request]'
            return {"id": call.id, "name": call.name, "content": text, "is_error": False}
        except ToolError as e:
            self._failed_sigs[sig] = self._failed_sigs.get(sig, 0) + 1
            extra = ""
            if self._failed_sigs[sig] >= 2:
                extra = " You have now repeated this exact failing call; change your approach or report the limitation."
                self.t.emit("recover", f"loop guard: identical failing call repeated ({call.name})")
            return {"id": call.id, "name": call.name, "is_error": True,
                    "content": f"ERROR kind={e.kind}: {e.message}. Hint: {HINTS.get(e.kind, '')}{extra}"}

    # ---- main loop --------------------------------------------------------------------------------------
    def run(self, goal: str) -> dict:
        self.t.emit("info", f"Goal: {goal}")
        self.t.emit("info", f"Model: {self.llm.name}; tools: {', '.join(self.reg.names())}")
        t0 = time.monotonic()
        catalog = "\n".join(f"- {t['name']}: {t['description']}" for t in self.reg.specs())
        msgs: list[dict] = [{"role": "user", "content": f"GOAL: {goal}\n\nYou have live access to these tools; they become "
                             f"callable right after make_plan (the repository is fetched on demand, nothing is needed "
                             f"from the user):\n{catalog}"}]
        nudges, turn, final = 0, 0, None
        try:
            while turn < self.max_turns and final is None:
                turn += 1
                last_chance = turn == self.max_turns
                if last_chance:
                    msgs.append({"role": "user", "content": "Turn budget exhausted. Call submit_report NOW with what you have "
                                 "(status 'partial' or 'failed' if incomplete) and list the gaps honestly."})
                    self.t.emit("recover", "turn budget exhausted: forcing final report")
                if last_chance:
                    tools = [m for m in self.meta if m["name"] == "submit_report"]
                elif not self.plans:
                    tools = [m for m in self.meta if m["name"] in ("make_plan", "ask_user")]   # plan before acting
                else:
                    tools = self.reg.specs() + self.meta
                turn_out = self._complete(msgs, tools)
                self.usage["in"] += turn_out.usage.get("in", 0)
                self.usage["out"] += turn_out.usage.get("out", 0)
                if turn_out.text.strip():
                    self.t.emit("think", turn_out.text.strip().replace("\n", " ")[:600])
                msgs.append({"role": "assistant", "text": turn_out.text,
                             "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in turn_out.tool_calls]})
                if not turn_out.tool_calls:
                    nudges += 1
                    if nudges > 2:
                        self.t.emit("error", "model stopped without submitting a report")
                        break
                    self.t.emit("recover", "model replied without a tool call; nudging it to continue or submit_report")
                    msgs.append({"role": "user", "content": "Continue with tools, or call submit_report if you are done. "
                                 + ("Call make_plan first." if not self.plans else "")})
                    continue
                results = []
                for c in turn_out.tool_calls:
                    if c.name == "make_plan":
                        results.append(self._on_plan(c))
                    elif c.name == "ask_user":
                        q = str(c.args.get("question", "")).strip() or "Please clarify the goal."
                        self.t.emit("report", f"Needs clarification: {q}")
                        final = {"status": "needs_clarification", "question": q}
                        results.append({"id": c.id, "name": c.name, "content": "ok", "is_error": False})
                    elif c.name == "submit_report":
                        problems = validate_report(c.args)
                        if problems:
                            self.t.emit("error", f"report rejected: {'; '.join(problems)}")
                            results.append({"id": c.id, "name": c.name, "is_error": True,
                                            "content": "Report invalid: " + "; ".join(problems) + ". Fix and call submit_report again."})
                        else:
                            final = dict(c.args)
                            self.t.emit("report", f"Report submitted: status={final['status']}, score={final['quality_score']}, "
                                        f"{len(final['findings'])} findings")
                            results.append({"id": c.id, "name": c.name, "content": "accepted", "is_error": False})
                    elif not self.plans:
                        results.append({"id": c.id, "name": c.name, "is_error": True, "content": "Call make_plan first."})
                    else:
                        results.append(self._tool_result(c, msgs))
                msgs.append({"role": "tool", "results": results})
        except LLMError as e:
            self.t.emit("error", f"model API failure: {e}")
            final = {"status": "failed", "error": str(e)}
        if final is None:
            final = {"status": "failed", "error": "the model did not produce a valid report within the turn budget"}
            self.t.emit("error", final["error"])
        final["run"] = {"goal": goal, "llm": self.llm.name, "turns": turn, "plans": self.plans, "usage": self.usage,
                        "duration_s": round(time.monotonic() - t0, 1),
                        "tool_log": [{"tool": e["tool"], "args": e["args"], "status": e["status"], "attempts": e["attempts"],
                                      "error": e["error"]} for e in self.tool_log]}
        return final

    def _complete(self, msgs: list, tools: list):
        """Call the model inside the context budget. If the provider still says the request is too large
        (HTTP 413 / context or token-per-minute limit), shrink the budget and compact harder, then retry."""
        for attempt in range(4):
            self._compact(msgs, keep_recent=2 if attempt == 0 else 0, stub=250 if attempt < 2 else 120)
            try:
                return self.llm.complete(SYSTEM_PROMPT, msgs, tools)
            except LLMError as e:
                text = str(e).lower()
                if attempt == 3 or not ("413" in text or "too large" in text or "context length" in text
                                        or "maximum context" in text or "reduce" in text):
                    raise
                self.context_chars = max(3000, int(self.context_chars * 0.5))
                self.result_chars = min(self.result_chars, self.context_chars // 3)
                self.t.emit("recover", f"model said the request is too large; shrinking context budget to "
                                       f"{self.context_chars} chars and retrying")

    def _compact(self, msgs: list, keep_recent: int = 2, stub: int = 250) -> None:
        """Keep the conversation inside the context budget (small free-tier limits) by shortening the oldest tool
        outputs first, then long assistant messages. The newest `keep_recent` tool turns are touched last."""
        size = lambda: sum(len(json.dumps(m, default=str)) for m in msgs)
        if size() <= self.context_chars:
            return
        shrunk = 0
        tool_turns = [m for m in msgs if m.get("role") == "tool"]
        ordered = tool_turns[:-keep_recent] if keep_recent else tool_turns
        for m in ordered:
            for r in m["results"]:
                c = str(r.get("content", ""))
                if len(c) > stub + 60:
                    r["content"] = f"[compacted to save context] {c[:stub]}"
                    shrunk += 1
            if size() <= self.context_chars:
                break
        if size() > self.context_chars:
            for m in msgs[1:-1]:
                if m.get("role") == "assistant" and len(m.get("text") or "") > stub + 60:
                    m["text"] = m["text"][:stub] + " [compacted]"
                    shrunk += 1
        if shrunk:
            self.t.emit("recover", f"context over budget: compacted {shrunk} old message(s)")

    def _on_plan(self, c) -> dict:
        a = c.args
        steps = [str(s) for s in a.get("steps", [])] if isinstance(a.get("steps"), list) else []
        if not steps or not a.get("reasoning"):
            return {"id": c.id, "name": c.name, "is_error": True, "content": "make_plan needs 'reasoning' and a non-empty 'steps' list."}
        plan = {"reasoning": str(a["reasoning"]), "steps": steps, "assumptions": [str(x) for x in a.get("assumptions", [])],
                "revision_reason": a.get("revision_reason")}
        self.plans.append(plan)
        label = "REVISED PLAN" if len(self.plans) > 1 else "Reasoning"
        if plan["revision_reason"]:
            self.t.emit("plan", f"Revision reason: {plan['revision_reason']}")
        self.t.emit("plan", f"{label}: {plan['reasoning']}")
        for i, s in enumerate(steps, 1):
            self.t.emit("plan", f"  {i}. {s}")
        for x in plan["assumptions"]:
            self.t.emit("plan", f"  assumption: {x}")
        return {"id": c.id, "name": c.name, "content": "plan recorded; begin executing it", "is_error": False}


def run_agent(goal: str, llm: LLMClient, registry: ToolRegistry, tracer: Tracer, **kw) -> dict:
    return Agent(llm, registry, tracer, **kw).run(goal)
