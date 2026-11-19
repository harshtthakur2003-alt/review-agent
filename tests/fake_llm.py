"""TEST DOUBLE ONLY (never imported by the agent package).

`PolicyLLM` lets a test script the *model's decisions* as a function of what it has observed, so the agent
loop, tool plumbing, retries, guard-rails and report handling can be verified deterministically and offline.
It is not an AI and is not available from the CLI."""
from __future__ import annotations

import itertools

from agent.llm import LLMClient, ToolCall, Turn

_ids = itertools.count(1)


def call(name, **args):
    return ToolCall(f"c{next(_ids)}", name, args)


GOOD_REPORT = {
    "status": "success", "repository": "acme/widgets", "summary": "Small repo with a few risky patterns.",
    "quality_score": 62, "score_rationale": "Several high severity patterns in bad.py.",
    "findings": [{"severity": "high", "title": "Bare except", "location": "src/bad.py:9",
                  "evidence": "except: pass swallows everything", "suggested_fix": "catch specific exceptions"}],
    "what_i_did": ["listed files", "analysed bad.py"], "recoveries": [], "assumptions": [], "limitations": ["sampled one file"],
}


class PolicyLLM(LLMClient):
    name = "policy-test-double"

    def __init__(self, policy):
        self.policy, self.turns, self.seen_tools = policy, 0, []

    def complete(self, system, messages, tools):
        self.turns += 1
        self.seen_tools.append([t["name"] for t in tools])
        decision = self.policy(self.turns, messages, tools)
        if isinstance(decision, Turn):
            return decision
        text, calls = decision if isinstance(decision, tuple) else ("", decision)
        return Turn(text, list(calls), usage={"in": 10, "out": 5})


def last_results(messages):
    for m in reversed(messages):
        if m["role"] == "tool":
            return m["results"]
    return []


def plan(reasoning="check repo"):
    return call("make_plan", reasoning=reasoning, steps=["resolve repo", "list files", "analyse", "report"])
