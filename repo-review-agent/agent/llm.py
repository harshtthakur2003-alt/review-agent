"""LLM backends.

* `AnthropicLLM`  - real model via the Messages API (needs ANTHROPIC_API_KEY).
* `OfflineLLM`    - deterministic stand-in so the agent, tests and demos run with no key/network.
                    It is *not* an intelligent model: it emits a fixed plan template for the
                    repo-review domain. This is documented in the README.

Both expose the same three capabilities, so the rest of the agent is backend-agnostic.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Protocol

import requests

from .models import Clarification
from .tools.github import _SLUG_RE, _URL_RE

PLAN_SCHEMA_HELP = """Return ONLY JSON: {"reasoning": str, "assumptions": [str], "steps": [
 {"id": "s1", "description": str, "tool": <tool name>, "args": {...}, "foreach": null | "$sN.results", "optional": bool, "rationale": str}]}
Args may reference earlier outputs as "$s1.field" (whole value) or embed "${s1.field}" in strings;
inside a foreach step the current element is "$item" / "${item.field}". Steps run in order."""


class LLM(Protocol):
    name: str

    def plan(self, goal: str, tools_desc: str) -> dict: ...
    def repair(self, step: dict, error: str, tools_desc: str) -> Optional[dict]: ...
    def narrate(self, facts: dict) -> Optional[str]: ...


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


class OfflineLLM:
    name = "offline-heuristic"

    def plan(self, goal: str, tools_desc: str) -> dict:
        m = re.search(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+/[\w.-]+", goal) or None
        ref = m.group(0) if m else None
        if not ref:
            slug = next((w for w in goal.replace(",", " ").split() if _SLUG_RE.match(w) and "/" in w), None)
            ref = slug
        if not ref:
            raise Clarification("No GitHub repository URL or owner/name found in the goal. "
                                "Please provide one, e.g. 'Review https://github.com/pallets/markupsafe'.")
        step = lambda i, d, t, a, **k: {"id": f"s{i}", "description": d, "tool": t, "args": a, **k}
        return {
            "reasoning": ("The goal is a code-quality review of a GitHub repository. I must (1) resolve the repo and its "
                          "default branch, (2) enumerate Python files, (3) pick a bounded, representative sample of source "
                          "files, (4) fetch and statically analyse each, (5) aggregate, (6) compute a score, then report."),
            "assumptions": ["Only Python files are reviewed.", "A sample of the largest non-test source files is analysed "
                            "(bounded cost) rather than the whole repository.",
                            "Analysis is static (AST); code is never executed."],
            "steps": [
                step(1, "Resolve repository and default branch", "repo_info", {"repo_url": ref},
                     rationale="Validates the URL early and discovers the default branch."),
                step(2, "List Python files on the default branch", "repo_list_files",
                     {"full_name": "$s1.full_name", "branch": "$s1.default_branch"}),
                step(3, "Select a representative sample of source files", "select_python_files",
                     {"files": "$s2.files", "max_files": 8}, rationale="Bound API calls and skip tests/vendored code."),
                step(4, "Fetch each selected file", "repo_read_file",
                     {"full_name": "$s1.full_name", "path": "$item.path", "ref": "$s2.branch"}, foreach="$s3.files"),
                step(5, "Statically analyse each file", "analyze_code",
                     {"path": "$item.path", "text": "$item.text"}, foreach="$s4.results"),
                step(6, "Aggregate findings", "summarize_findings", {"analyses": "$s5.results"}),
                step(7, "Compute weighted issues per 1,000 LOC", "calculator",
                     {"expression": "1000 * ${s6.weighted_issue_points} / max(${s6.loc}, 1)"}),
                step(8, "Derive 0-100 quality score", "calculator",
                     {"expression": "round(max(0, 100 - ${s7.value}), 1)"}, rationale="Simple, transparent heuristic."),
            ],
        }

    def repair(self, step: dict, error: str, tools_desc: str) -> Optional[dict]:
        return None  # the offline backend has no free-form reasoning; deterministic fallbacks cover its cases

    def narrate(self, facts: dict) -> Optional[str]:
        return None


class AnthropicLLM:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 60.0):
        self.name = f"anthropic:{model or os.environ.get('AGENT_MODEL', 'claude-sonnet-5-5')}"
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        self.timeout = timeout
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (use --llm offline to run without a key)")

    def _ask(self, system: str, user: str, max_tokens: int = 2000) -> str:
        r = requests.post(f"{self.base}/v1/messages", timeout=self.timeout,
                          headers={"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": self.model, "max_tokens": max_tokens, "system": system,
                                "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    def plan(self, goal: str, tools_desc: str) -> dict:
        system = ("You are the planning module of an autonomous agent. Decompose the user's goal into an ordered list of "
                  f"tool-using steps using ONLY these tools:\n{tools_desc}\n{PLAN_SCHEMA_HELP}\n"
                  "If the goal lacks a GitHub repo to review, return {\"clarification\": \"<question>\"}.")
        data = extract_json(self._ask(system, goal))
        if "clarification" in data:
            raise Clarification(data["clarification"])
        return data

    def repair(self, step: dict, error: str, tools_desc: str) -> Optional[dict]:
        system = ("A tool call failed. Propose corrected args, or give up. Return ONLY JSON: "
                  '{"args": {...}, "why": str} or {"give_up": true, "why": str}.\n' + tools_desc)
        data = extract_json(self._ask(system, json.dumps({"step": step, "error": error}), 500))
        return None if data.get("give_up") else data

    def narrate(self, facts: dict) -> Optional[str]:
        system = "Write a concise (<=120 words) plain-English executive summary of this code review. No markdown headers."
        return self._ask(system, json.dumps(facts, default=str), 400).strip()


def make_llm(kind: str) -> "LLM":
    if kind == "offline":
        return OfflineLLM()
    if kind == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"unknown llm backend {kind!r}")
