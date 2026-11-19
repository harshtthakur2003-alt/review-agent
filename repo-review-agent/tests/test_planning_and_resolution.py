import pytest

from agent.executor import resolve
from agent.llm import OfflineLLM
from agent.models import Clarification, ToolError
from agent.planner import make_plan, validate_plan
from agent.tools import build_registry
from agent.trace import Tracer


def test_resolve_whole_embedded_and_item():
    ctx = {"s1": {"a": {"b": [10, 20]}, "name": "x"}, "item": {"p": "f.py"}}
    assert resolve("$s1.a.b[1]", ctx) == 20
    assert resolve("n=${s1.name}!", ctx) == "n=x!"
    assert resolve({"k": ["$item.p"]}, ctx) == {"k": ["f.py"]}


def test_resolve_missing_reference_is_input_error():
    with pytest.raises(ToolError) as e:
        resolve("$s9.x", {})
    assert e.value.kind == "invalid_input"


def test_offline_plan_is_valid_and_ordered():
    reg = build_registry()
    plan = make_plan("review https://github.com/psf/requests", OfflineLLM(), reg, Tracer(quiet=True))
    assert [s.id for s in plan.steps] == [f"s{i}" for i in range(1, 9)]
    assert sum(1 for s in plan.steps if s.foreach) == 2


def test_ambiguous_goal_requests_clarification():
    with pytest.raises(Clarification):
        make_plan("make my code better", OfflineLLM(), build_registry(), Tracer(quiet=True))


@pytest.mark.parametrize("bad,msg", [
    ({"steps": [{"id": "s1", "tool": "nope", "args": {}}]}, "unknown tool"),
    ({"steps": [{"id": "s1", "tool": "calculator", "args": {"expression": "$s2.v"}}]}, "does not run before"),
    ({"steps": [{"id": "s1", "tool": "calculator", "args": {"expression": "$item.v"}}]}, "no foreach"),
    ({"steps": []}, "no steps"),
])
def test_validate_plan_rejects(bad, msg):
    with pytest.raises(ValueError, match=msg):
        validate_plan(bad, build_registry())


def test_invalid_llm_plan_is_retried_then_falls_back():
    class Bad(OfflineLLM):
        calls = 0

        def plan(self, goal, tools_desc):
            Bad.calls += 1
            return {"steps": [{"id": "s1", "tool": "ghost", "args": {}}]}

    plan = make_plan("review github.com/a/b", Bad(), build_registry(), Tracer(quiet=True))
    assert Bad.calls == 2 and plan.steps[0].tool == "repo_info"
