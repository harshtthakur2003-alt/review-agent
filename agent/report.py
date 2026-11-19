"""The final report is written by the model through the `submit_report` tool. This module only defines the
schema it must satisfy, validates it, and renders the (model-authored) JSON as Markdown. No finding text,
score or advice is generated here."""
from __future__ import annotations

from typing import Any

SEVERITIES = ("high", "medium", "low")

REPORT_SCHEMA: dict = {
    "status": {"type": "string", "enum": ["success", "partial", "failed"],
               "description": "success = goal met; partial = met with gaps you must list; failed = could not be met"},
    "repository": {"type": "string", "description": "owner/name actually reviewed (note if it differs from the request)"},
    "summary": {"type": "string", "description": "3-6 sentence executive summary of code quality"},
    "quality_score": {"type": "number", "description": "0-100 overall score; justify it in score_rationale"},
    "score_rationale": {"type": "string"},
    "findings": {"type": "array", "description": "most important issues first",
                 "items": {"type": "object", "properties": {
                     "severity": {"type": "string", "enum": list(SEVERITIES)},
                     "title": {"type": "string"},
                     "location": {"type": "string", "description": "path:line"},
                     "evidence": {"type": "string", "description": "what you saw in the code or tool output"},
                     "suggested_fix": {"type": "string", "description": "concrete change, ideally with a short code sketch"}},
                     "required": ["severity", "title", "location", "evidence", "suggested_fix"]}},
    "what_i_did": {"type": "array", "items": {"type": "string"}, "description": "the steps you took and why"},
    "recoveries": {"type": "array", "description": "every failure/surprise you hit and how you handled it (empty if none)",
                   "items": {"type": "object", "properties": {"problem": {"type": "string"}, "action": {"type": "string"}},
                             "required": ["problem", "action"]}},
    "assumptions": {"type": "array", "items": {"type": "string"}},
    "limitations": {"type": "array", "items": {"type": "string"}},
}
REQUIRED = ["status", "repository", "summary", "quality_score", "score_rationale", "findings", "what_i_did",
            "recoveries", "assumptions", "limitations"]


def validate_report(r: Any) -> list[str]:
    """Return a list of problems (empty = valid)."""
    if not isinstance(r, dict):
        return ["report must be an object"]
    problems = [f"missing field '{k}'" for k in REQUIRED if k not in r]
    if problems:
        return problems
    if r["status"] not in ("success", "partial", "failed"):
        problems.append("status must be success|partial|failed")
    if not isinstance(r["quality_score"], (int, float)) or not 0 <= r["quality_score"] <= 100:
        problems.append("quality_score must be a number 0-100")
    for k in ("findings", "what_i_did", "recoveries", "assumptions", "limitations"):
        if not isinstance(r[k], list):
            problems.append(f"{k} must be a list")
    for i, f in enumerate(r["findings"] if isinstance(r["findings"], list) else []):
        if not isinstance(f, dict) or f.get("severity") not in SEVERITIES or not all(
                f.get(k) for k in ("title", "location", "evidence", "suggested_fix")):
            problems.append(f"findings[{i}] needs severity(high|medium|low), title, location, evidence, suggested_fix")
    if r["status"] != "failed" and not r["findings"]:
        problems.append("a non-failed review must contain at least one finding (or status must explain why not)")
    return problems


def to_markdown(rep: dict) -> str:
    run = rep.get("run", {})
    if rep.get("status") == "needs_clarification":
        return f"# Needs clarification\n\n**Goal:** {run.get('goal')}\n\n{rep['question']}\n"
    L = ["# Code review report", "", f"**Goal:** {run.get('goal')}  ", f"**Status:** `{rep['status']}`  ",
         f"**Model:** {run.get('llm')}  ", f"**Repository:** {rep.get('repository', '-')}", "", rep.get("summary", ""), ""]
    if "quality_score" in rep:
        L += [f"## Score: {rep['quality_score']} / 100", rep.get("score_rationale", ""), ""]
    L += ["## Findings"]
    for i, f in enumerate(rep.get("findings", []), 1):
        L += [f"### {i}. [{f['severity'].upper()}] {f['title']}", f"`{f['location']}`", "", f"**Evidence:** {f['evidence']}", "",
              f"**Fix:** {f['suggested_fix']}", ""]
    L += ["## What the agent did and why"] + [f"{i}. {s}" for i, s in enumerate(rep.get("what_i_did", []), 1)]
    if rep.get("recoveries"):
        L += ["", "## Failures and recoveries"] + [f"- **{r['problem']}** -> {r['action']}" for r in rep["recoveries"]]
    L += ["", "## Assumptions"] + [f"- {a}" for a in rep.get("assumptions", [])]
    L += ["", "## Limitations"] + [f"- {a}" for a in rep.get("limitations", [])]
    L += ["", "## Plans", ""]
    for n, p in enumerate(run.get("plans", []), 1):
        L += [f"**Plan v{n}** - {p['reasoning']}"] + [f"{i}. {s}" for i, s in enumerate(p["steps"], 1)] + [""]
    L += ["## Execution log", "| # | Tool | Result | Attempts |", "|---|---|---|---|"]
    L += [f"| {i} | `{c['tool']}` | {c['status']}{(': ' + c['error'][:90]) if c.get('error') else ''} | {c['attempts']} |"
          for i, c in enumerate(run.get("tool_log", []), 1)]
    u = run.get("usage", {})
    L += ["", f"_Turns: {run.get('turns')} - tool calls: {len(run.get('tool_log', []))} - tokens in/out: {u.get('in', 0)}/{u.get('out', 0)}_"]
    return "\n".join(L) + "\n"
