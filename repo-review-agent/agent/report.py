"""Builds the final structured result (dict -> JSON) and a Markdown rendering."""
from __future__ import annotations

from typing import Optional

from .models import Plan, StepResult

FIX_HINTS = {
    "bare_except": "Catch specific exception types (or `except Exception:`) and log/handle them.",
    "mutable_default_arg": "Use `None` as default and create the list/dict inside the function body.",
    "eval_exec": "Replace eval/exec with ast.literal_eval or an explicit dispatch table.",
    "swallowed_exception": "Log the exception or re-raise; silent `pass` hides real bugs.",
    "high_complexity": "Split into smaller functions / early returns to reduce branching.",
    "long_function": "Extract cohesive blocks into helpers; aim for functions that fit on one screen.",
    "too_many_args": "Group related parameters into a dataclass or config object.",
    "wildcard_import": "Import names explicitly to keep namespaces and static analysis clean.",
    "missing_docstring": "Add short docstrings to public modules/classes/functions.",
    "todo_comment": "Convert TODO/FIXME comments into tracked issues.",
    "syntax_error": "File failed to parse (possibly Python 2 / templated); confirm it is intended to be Python 3.",
}
LIMITATIONS = [
    "Static AST heuristics only: no type checking, no test execution, no data-flow analysis.",
    "Only a bounded sample of the largest non-test source files is reviewed, so the score is an estimate.",
    "Score = 100 - (weighted issues per 1,000 LOC); severity weights (3/2/1) are a judgement call, not a standard.",
    "Cyclomatic complexity is approximated by counting branch nodes.",
]


def build_report(goal: str, plan: Optional[Plan], results: list[StepResult], llm_name: str,
                 clarification: Optional[str] = None, narrative: Optional[str] = None) -> dict:
    if clarification:
        return {"goal": goal, "status": "needs_clarification", "message": clarification, "llm": llm_name}
    by_id = {r.step_id: r for r in results}
    summary = next((r.output for s in plan.steps if s.tool == "summarize_findings"
                    for r in [by_id[s.id]] if r.status in ("ok", "partial")), None)
    score_steps = [by_id[s.id] for s in plan.steps if s.tool == "calculator" and by_id[s.id].status == "ok"]
    score = score_steps[-1].output["value"] if score_steps else None
    info = next((r.output for s in plan.steps if s.tool == "repo_info" for r in [by_id[s.id]] if r.status == "ok"), None)
    sample = next((r.output for s in plan.steps if s.tool == "select_python_files" for r in [by_id[s.id]] if r.status == "ok"), None)

    required_bad = [s.id for s in plan.steps if not s.optional and by_id[s.id].status in ("failed", "skipped")]
    degraded = [r.step_id for r in results if r.status == "partial"]
    status = "failed" if required_bad else ("partial" if degraded else "success")

    suggestions = []
    if summary:
        for code, n in sorted(summary["by_code"].items(), key=lambda kv: -kv[1]):
            suggestions.append({"issue": code, "occurrences": n, "suggested_fix": FIX_HINTS.get(code, "Review manually.")})
    report = {
        "goal": goal, "status": status, "llm": llm_name,
        "repository": info, "quality_score": score,
        "files_reviewed": [f["path"] for f in sample["files"]] if sample else [],
        "metrics": {k: summary[k] for k in ("files_analyzed", "loc", "issue_count", "by_severity", "by_code")} if summary else None,
        "worst_files": summary["worst_files"] if summary else [],
        "top_issues": summary["top_issues"] if summary else [],
        "suggested_fixes": suggestions,
        "plan": {"reasoning": plan.reasoning,
                 "assumptions": plan.assumptions + [f"[{r.step_id} recovery] {x[len('fallback: '):]}" for r in results
                                                    for x in r.recoveries if x.startswith(("fallback:", "LLM repair:"))],
                 "steps": [{"id": s.id, "description": s.description, "tool": s.tool} for s in plan.steps]},
        "execution": [{"step": r.step_id, "status": r.status, "attempts": r.attempts, "note": r.note,
                       "errors": r.errors, "recoveries": r.recoveries} for r in results],
        "failed_or_skipped": [{"step": r.step_id, "status": r.status, "why": r.note} for r in results if r.status in ("failed", "skipped")],
        "limitations": LIMITATIONS,
        "summary": narrative,
    }
    return report


def to_markdown(rep: dict) -> str:
    if rep["status"] == "needs_clarification":
        return f"# Needs clarification\n\n**Goal:** {rep['goal']}\n\n{rep['message']}\n"
    L = [f"# Code review report", "", f"**Goal:** {rep['goal']}  ", f"**Status:** `{rep['status']}`  ",
         f"**Planner/LLM backend:** {rep['llm']}", ""]
    if rep["repository"]:
        r = rep["repository"]
        L += [f"**Repository:** {r['full_name']} (branch `{r['default_branch']}`)", ""]
    if rep["summary"]:
        L += [rep["summary"], ""]
    if rep["quality_score"] is not None and rep["metrics"]:
        m = rep["metrics"]
        L += ["## Result", f"- **Quality score:** {rep['quality_score']} / 100",
              f"- Files analysed: {m['files_analyzed']} ({m['loc']} LOC)",
              f"- Issues: {m['issue_count']} (high {m['by_severity']['high']}, medium {m['by_severity']['medium']}, low {m['by_severity']['low']})", ""]
        L += ["## Issues by type", "| Issue | Count | Suggested fix |", "|---|---|---|"]
        L += [f"| {s['issue']} | {s['occurrences']} | {s['suggested_fix']} |" for s in rep["suggested_fixes"]]
        L += ["", "## Top issues", "| Severity | Location | Detail |", "|---|---|---|"]
        L += [f"| {i['severity']} | `{i['path']}:{i['line']}` | {i['detail']} |" for i in rep["top_issues"]]
        L += ["", "## Hotspot files"] + [f"- `{w['path']}`: {w['issues']} issues in {w['loc']} LOC" for w in rep["worst_files"]] + [""]
    L += ["## What I did and why", rep["plan"]["reasoning"], ""]
    L += [f"{i}. {s['description']} (`{s['tool']}`)" for i, s in enumerate(rep["plan"]["steps"], 1)]
    L += ["", "## Execution log", "| Step | Status | Attempts | Recoveries / notes |", "|---|---|---|---|"]
    for e in rep["execution"]:
        notes = "; ".join(e["recoveries"][:3] + ([e["note"]] if e["note"] else [])) or "-"
        L.append(f"| {e['step']} | {e['status']} | {e['attempts']} | {notes} |")
    if rep["failed_or_skipped"]:
        L += ["", "## Failures that could not be recovered"] + [f"- {f['step']} ({f['status']}): {f['why']}" for f in rep["failed_or_skipped"]]
    L += ["", "## Assumptions"] + [f"- {a}" for a in rep["plan"]["assumptions"]]
    L += ["", "## Limitations"] + [f"- {x}" for x in rep["limitations"]]
    return "\n".join(L) + "\n"
