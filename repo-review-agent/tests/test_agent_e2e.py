"""End-to-end behaviour against a real local git remote (no network, no LLM key)."""
import pytest

from agent.core import run_agent
from agent.faults import inject_faults, parse_spec
from agent.llm import OfflineLLM
from agent.executor import Executor
from agent.models import Step, ToolError
from agent.tools.base import Tool, ToolRegistry
from agent.trace import Tracer


def go(make_env, goal, faults=(), known=("acme/widgets",)):
    reg, tr = make_env(known)
    if faults:
        inject_faults(reg, list(faults))
    return run_agent(goal, OfflineLLM(), reg, tr, backoff=0), tr


def rec(rep, step):
    return next(e for e in rep["execution"] if e["step"] == step)


def test_happy_path(make_env):
    rep, _ = go(make_env, "review acme/widgets")
    assert rep["status"] == "success"
    assert rep["files_reviewed"] and all("test" not in f for f in rep["files_reviewed"])
    assert rep["metrics"]["by_severity"]["high"] >= 3
    assert 0 <= rep["quality_score"] < 100
    assert "syntax_error" in rep["metrics"]["by_code"]  # broken.py is reported, not a crash


def test_transient_timeouts_are_retried(make_env):
    rep, _ = go(make_env, "review acme/widgets", ["repo_list_files:timeout:2"])
    assert rep["status"] == "success"
    r = rec(rep, "s2")
    assert r["attempts"] == 3 and len(r["recoveries"]) == 2 and "backoff" in r["recoveries"][0]


def test_malformed_response_detected_and_retried(make_env):
    rep, _ = go(make_env, "review acme/widgets", ["repo_info:malformed:1"])
    assert rep["status"] == "success" and rec(rep, "s1")["attempts"] == 2
    assert "missing keys" in rec(rep, "s1")["errors"][0]


def test_rate_limit_then_success(make_env):
    rep, _ = go(make_env, "review acme/widgets", ["repo_read_file:rate_limit:2"])
    assert rep["status"] == "success"


def test_persistent_failure_degrades_to_partial_not_crash(make_env):
    # every read of the first file fails permanently (not_found, no fallback) -> that item excluded
    rep, _ = go(make_env, "review acme/widgets", ["repo_read_file:not_found:1"])
    assert rep["status"] == "partial"
    assert rec(rep, "s4")["status"] == "partial"
    assert rep["files_reviewed"] and rep["quality_score"] is not None


def test_misspelled_repo_is_repaired_via_search(make_env):
    rep, _ = go(make_env, "review acme/widgetz")
    assert rep["status"] == "success" and rep["repository"]["full_name"] == "acme/widgets"
    assert "closest unambiguous match" in rec(rep, "s1")["recoveries"][0]


def test_unrepairable_repo_fails_loudly_and_skips_dependents(make_env):
    rep, _ = go(make_env, "review nobody/nothing-here", known=())
    assert rep["status"] == "failed"
    assert rec(rep, "s1")["status"] == "failed"
    assert all(rec(rep, f"s{i}")["status"] == "skipped" for i in range(2, 9))
    assert rep["failed_or_skipped"][0]["step"] == "s1"


def test_ambiguous_match_is_not_guessed(make_env):
    rep, _ = go(make_env, "review acme/widgetz", known=("acme/widgets", "acme/widgetx"))
    assert rep["status"] == "failed"  # two near-identical candidates -> refuse to guess


def test_wrong_branch_falls_back(make_env):
    from agent.executor import Executor
    reg, tr = make_env()
    out, attempts, errs, recs = Executor(reg, OfflineLLM(), tr, backoff=0).call_with_recovery(
        "repo_list_files", {"full_name": "acme/widgets", "branch": "develop"}, "t")
    assert out["branch"] == "main" and "trying 'main'" in recs[0]


def test_goal_without_repo_asks_for_clarification(make_env):
    rep, _ = go(make_env, "please improve my code")
    assert rep["status"] == "needs_clarification"


def test_executor_gives_up_after_bounded_retries():
    reg = ToolRegistry()
    n = {"c": 0}

    def flaky():
        n["c"] += 1
        raise ToolError("transient", "always down")

    reg.register(Tool("flaky", "x", flaky))
    ex = Executor(reg, OfflineLLM(), Tracer(quiet=True), max_attempts=3, backoff=0)
    with pytest.raises(ToolError):
        ex.call_with_recovery("flaky", {}, "t")
    assert n["c"] == 3  # bounded: no infinite retry loop


def test_real_timeout_enforced():
    import time
    reg = ToolRegistry(default_timeout=0.1)
    reg.register(Tool("slow", "x", lambda: time.sleep(1) or {}))
    with pytest.raises(ToolError) as e:
        reg.call("slow", {})
    assert e.value.kind == "transient" and "timed out" in e.value.message


def test_fault_spec_validation():
    assert parse_spec("a:timeout:3") == ("a", "timeout", 3)
    with pytest.raises(ValueError):
        parse_spec("a:explode")


def test_multiple_faults_on_one_tool_all_fire(make_env):
    rep, _ = go(make_env, "review acme/widgets", ["repo_read_file:malformed:1", "repo_read_file:rate_limit:1"])
    errs = rec(rep, "s4")["errors"]
    assert any("missing keys" in e for e in errs) and any("429" in e for e in errs)
    assert rep["status"] == "success"


def test_autonomous_repairs_are_surfaced_as_assumptions(make_env):
    rep, _ = go(make_env, "review acme/widgetz")
    assert any("closest unambiguous match" in a for a in rep["plan"]["assumptions"])


def test_failed_step_keeps_full_retry_history(make_env):
    rep, _ = go(make_env, "review acme/widgets", ["repo_info:timeout:10"])
    r = rec(rep, "s1")
    assert rep["status"] == "failed" and r["attempts"] >= 4 and len(r["errors"]) == r["attempts"] and r["recoveries"]
