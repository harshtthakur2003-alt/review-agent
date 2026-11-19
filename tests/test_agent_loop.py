"""Agent loop behaviour against a real local git remote, with the model's decisions scripted (offline)."""
import json

from agent.faults import inject_faults
from agent.loop import run_agent
from agent.report import to_markdown, validate_report

from fake_llm import GOOD_REPORT, PolicyLLM, call, last_results, plan


def run(make_env, policy, faults=(), **kw):
    reg, tr = make_env(kw.pop("known", ("acme/widgets",)))
    if faults:
        inject_faults(reg, list(faults))
    llm = PolicyLLM(policy)
    return run_agent("review acme/widgets", llm, reg, tr, backoff=0, **kw), llm, tr


def happy(turn, msgs, tools):
    return {1: [plan()], 2: [call("repo_info", repo_url="acme/widgets")],
            3: [call("repo_list_files", full_name="acme/widgets", glob="*.py")],
            4: [call("analyze_python_file", full_name="acme/widgets", path="src/bad.py")],
            5: [call("submit_report", **GOOD_REPORT)]}[turn]


def test_happy_path_and_report_rendering(make_env):
    rep, llm, _ = run(make_env, happy)
    assert rep["status"] == "success" and rep["quality_score"] == 62
    assert rep["run"]["plans"][0]["steps"] and rep["run"]["turns"] == 5
    assert [c["tool"] for c in rep["run"]["tool_log"]] == ["repo_info", "repo_list_files", "analyze_python_file"]
    md = to_markdown(rep)
    assert "Bare except" in md and "Execution log" in md and "Plan v1" in md
    json.dumps(rep, default=str)


def test_model_must_plan_before_acting(make_env):
    rep, llm, _ = run(make_env, happy)
    assert llm.seen_tools[0] == ["make_plan", "ask_user"]          # only planning tools offered on turn 1
    assert "repo_info" in llm.seen_tools[1]                          # real tools unlocked after the plan

    def eager(turn, msgs, tools):
        if turn == 1:
            return [call("repo_info", repo_url="acme/widgets")]      # tries to skip planning
        if turn == 2:
            assert "make_plan first" in last_results(msgs)[0]["content"]
            return [plan()]
        return [call("submit_report", **GOOD_REPORT)]
    rep, _, _ = run(make_env, eager)
    assert rep["status"] == "success" and not rep["run"]["tool_log"]


def test_transient_failures_auto_retried_with_backoff(make_env):
    rep, _, tr = run(make_env, happy, faults=["repo_list_files:timeout:2"])
    assert rep["status"] == "success"
    lf = [c for c in rep["run"]["tool_log"] if c["tool"] == "repo_list_files"][0]
    assert lf["attempts"] == 3 and lf["status"] == "ok"
    assert sum("automatic retry" in e["message"] for e in tr.events) == 2


def test_malformed_output_detected(make_env):
    rep, _, tr = run(make_env, happy, faults=["repo_info:malformed:1"])
    assert rep["status"] == "success"
    assert any("missing keys" in e["message"] for e in tr.events if e["kind"] == "error")


def test_model_repairs_misspelled_repo_after_not_found(make_env):
    """Self-correction is the model's job: it sees the error, searches, and retries with a fixed name."""
    def policy(turn, msgs, tools):
        if turn == 1:
            return [plan()]
        if turn == 2:
            return [call("repo_info", repo_url="acme/widgts")]
        r = last_results(msgs)[0]
        if turn == 3:
            assert r["is_error"] and "not_found" in r["content"]
            return [call("repo_search", query="user:acme")]
        if turn == 4:
            assert "acme/widgets" in r["content"]
            return [call("make_plan", reasoning="typo fixed", steps=["use acme/widgets"], revision_reason="repo name misspelled"),
                    call("repo_info", repo_url="acme/widgets")]
        return [call("submit_report", **{**GOOD_REPORT, "recoveries": [{"problem": "acme/widgts 404", "action": "searched owner"}],
                                          "assumptions": ["meant acme/widgets"]})]
    rep, _, tr = run(make_env, policy)
    assert rep["status"] == "success" and len(rep["run"]["plans"]) == 2
    assert any("REVISED PLAN" in e["message"] for e in tr.events)
    assert [c["status"] for c in rep["run"]["tool_log"]][:1] == ["not_found"]


def test_repeated_identical_failure_triggers_loop_guard(make_env):
    seen = []

    def policy(turn, msgs, tools):
        if turn == 1:
            return [plan()]
        if turn in (2, 3):
            if turn == 3:
                seen.append(last_results(msgs)[0]["content"])
            return [call("repo_read_file", full_name="acme/widgets", path="nope.py")]
        seen.append(last_results(msgs)[0]["content"])
        return [call("submit_report", **{**GOOD_REPORT, "status": "partial"})]
    rep, _, tr = run(make_env, policy)
    assert "repeated this exact failing call" in seen[-1]
    assert any("loop guard" in e["message"] for e in tr.events)


def test_invalid_report_is_rejected_then_corrected(make_env):
    def policy(turn, msgs, tools):
        if turn == 1:
            return [plan()]
        if turn == 2:
            return [call("submit_report", status="success", repository="x")]
        assert last_results(msgs)[0]["is_error"] and "missing field" in last_results(msgs)[0]["content"]
        return [call("submit_report", **GOOD_REPORT)]
    rep, _, _ = run(make_env, policy)
    assert rep["status"] == "success" and rep["run"]["turns"] == 3


def test_ambiguous_goal_asks_for_clarification(make_env):
    rep, _, _ = run(make_env, lambda t, m, tl: [call("ask_user", question="Which repo?")] if t == 2 else [plan()])
    assert rep["status"] == "needs_clarification" and rep["question"] == "Which repo?"
    assert "Needs clarification" in to_markdown(rep)


def test_turn_budget_forces_final_report(make_env):
    def policy(turn, msgs, tools):
        if turn == 1:
            return [plan()]
        if [t["name"] for t in tools] == ["submit_report"]:
            return [call("submit_report", **{**GOOD_REPORT, "status": "partial"})]
        return [call("repo_info", repo_url="acme/widgets")]
    rep, _, tr = run(make_env, policy, max_turns=4)
    assert rep["status"] == "partial"
    assert any("turn budget exhausted" in e["message"] for e in tr.events)


def test_model_that_never_finishes_fails_honestly(make_env):
    rep, _, _ = run(make_env, lambda t, m, tl: [plan()] if t == 1 else ("thinking...", []))
    assert rep["status"] == "failed" and "error" in rep


def test_model_api_outage_reported_not_crashed(make_env):
    from agent.models import LLMError

    def policy(turn, msgs, tools):
        raise LLMError("model API failed after 5 attempts (HTTP 529)")
    rep, _, _ = run(make_env, policy)
    assert rep["status"] == "failed" and "529" in rep["error"]


def test_validate_report():
    assert validate_report(GOOD_REPORT) == []
    assert validate_report({**GOOD_REPORT, "quality_score": 150})
    assert validate_report({**GOOD_REPORT, "findings": [{"severity": "huge"}]})
    assert validate_report({**GOOD_REPORT, "findings": []})


def test_context_compaction_keeps_recent_tool_outputs(make_env):
    from agent.loop import Agent
    reg, tr = make_env()
    agent = Agent(PolicyLLM(happy), reg, tr, context_chars=4000)
    big = "x" * 3000
    msgs = [{"role": "user", "content": "GOAL"}] + [
        {"role": "tool", "results": [{"id": str(i), "name": "repo_read_file", "content": big, "is_error": False}]}
        for i in range(4)]
    agent._compact(msgs)
    contents = [m["results"][0]["content"] for m in msgs[1:]]
    assert contents[0].startswith("[compacted") and contents[1].startswith("[compacted")
    assert contents[2] == big and contents[3] == big          # two most recent tool turns untouched


def test_request_too_large_shrinks_context_and_retries(make_env):
    from agent.models import LLMError
    reg, tr = make_env()
    inner = PolicyLLM(happy)
    sizes = []

    class TooLargeOnce:
        name = "fake-413"

        def __init__(self):
            self.failed = False

        def complete(self, system, messages, tools):
            sizes.append(sum(len(json.dumps(m, default=str)) for m in messages))
            if len(messages) > 3 and not self.failed:
                self.failed = True
                raise LLMError("HTTP 413: Request too large for model, please reduce your message size")
            return inner.complete(system, messages, tools)

    rep = run_agent("review acme/widgets", TooLargeOnce(), reg, tr, backoff=0, context_chars=60000)
    assert rep["status"] == "success"
    assert any("too large" in e["message"] for e in tr.events if e["kind"] == "recover")


def test_invalid_tool_call_from_provider_is_retried(make_env):
    from agent.models import LLMError
    reg, tr = make_env()
    inner = PolicyLLM(happy)

    class BadCallOnce:
        name = "fake-tool-use-failed"
        failed = False

        def complete(self, system, messages, tools):
            if not self.failed and len(messages) > 5:
                self.failed = True
                raise LLMError('HTTP 400: {"error":{"message":"Tool call validation failed: attempted to call tool '
                               '\'json\' which was not in request.tools","code":"tool_use_failed"}}')
            return inner.complete(system, messages, tools)

    rep = run_agent("review acme/widgets", BadCallOnce(), reg, tr, backoff=0)
    assert rep["status"] == "success"
    assert any("invalid tool call" in e["message"] for e in tr.events if e["kind"] == "recover")
