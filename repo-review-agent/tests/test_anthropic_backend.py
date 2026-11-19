"""The real-LLM backend can't be exercised without a key here, so its request/response handling is tested with a stub."""
import json

import pytest

from agent.llm import AnthropicLLM, extract_json
from agent.models import Clarification
from agent.planner import make_plan
from agent.tools import build_registry
from agent.trace import Tracer


class FakeResp:
    def __init__(self, text): self._t = text
    def raise_for_status(self): pass
    def json(self): return {"content": [{"type": "text", "text": self._t}]}


def llm_returning(monkeypatch, *texts):
    it = iter(texts)
    seen = []
    monkeypatch.setattr("agent.llm.requests.post", lambda *a, **k: (seen.append(k), FakeResp(next(it)))[1])
    return AnthropicLLM(model="m", api_key="k"), seen


def test_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AnthropicLLM()


def test_extract_json_tolerates_prose_and_fences():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_llm_plan_used_when_valid_and_self_corrected_when_not(monkeypatch):
    bad = json.dumps({"steps": [{"id": "s1", "tool": "made_up", "args": {}}]})
    good = json.dumps({"reasoning": "r", "steps": [{"id": "s1", "description": "d", "tool": "repo_info", "args": {"repo_url": "a/b"}}]})
    llm, seen = llm_returning(monkeypatch, bad, good)
    plan = make_plan("review a/b", llm, build_registry(), Tracer(quiet=True))
    assert plan.steps[0].tool == "repo_info" and len(seen) == 2
    assert "rejected" in seen[1]["json"]["messages"][0]["content"]  # validation error fed back to the model


def test_llm_clarification(monkeypatch):
    llm, _ = llm_returning(monkeypatch, json.dumps({"clarification": "which repo?"}))
    with pytest.raises(Clarification):
        make_plan("fix stuff", llm, build_registry(), Tracer(quiet=True))
