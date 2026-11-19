import json

import pytest

from agent.llm import AnthropicLLM, OpenAICompatLLM, make_llm
from agent.models import LLMError, ToolError


def test_registry_validates_inputs(make_env):
    reg, _ = make_env()
    with pytest.raises(ToolError) as e:
        reg.call("repo_info", {})
    assert e.value.kind == "invalid_input"
    with pytest.raises(ToolError) as e:
        reg.call("repo_info", {"repo_url": "acme/widgets", "bogus": 1})
    assert "unknown args" in str(e.value)
    with pytest.raises(ToolError):
        reg.call("no_such_tool", {})


def test_repo_tools(make_env):
    reg, _ = make_env()
    assert reg.call("repo_info", {"repo_url": "acme/widgets"})["default_branch"] == "main"
    with pytest.raises(ToolError) as e:
        reg.call("repo_info", {"repo_url": "acme/nothing"})
    assert e.value.kind == "not_found"
    files = reg.call("repo_list_files", {"full_name": "acme/widgets", "glob": "*.py"})
    assert files["total_files"] == 4
    f = reg.call("repo_read_file", {"full_name": "acme/widgets", "path": "src/bad.py", "start_line": 5, "max_lines": 3})
    assert f["text"].startswith("5: ") and f["has_more"]
    hits = reg.call("repo_grep", {"full_name": "acme/widgets", "pattern": r"except:"})["matches"]
    assert any(h["path"] == "src/bad.py" for h in hits)
    with pytest.raises(ToolError) as e:
        reg.call("repo_read_file", {"full_name": "acme/widgets", "path": "../../etc/passwd"})
    assert e.value.kind in ("invalid_input", "not_found")
    a = reg.call("analyze_python_file", {"full_name": "acme/widgets", "path": "src/bad.py"})
    assert {i["code"] for i in a["issues"]} >= {"bare_except", "mutable_default_arg", "eval_exec"}
    assert reg.call("analyze_python_file", {"full_name": "acme/widgets", "path": "src/broken.py"})["issues"][0]["code"] == "syntax_error"


def test_calculator_is_safe(make_env):
    reg, _ = make_env()
    assert reg.call("calculator", {"expression": "round(100 - 3*4/2, 1)"})["value"] == 94
    with pytest.raises(ToolError):
        reg.call("calculator", {"expression": "__import__('os').system('x')"})


class FakeResp:
    def __init__(self, code, body):
        self.status_code, self._b, self.text = code, body, json.dumps(body)

    def json(self):
        return self._b


MSGS = [{"role": "user", "content": "hi"},
        {"role": "assistant", "text": "plan", "tool_calls": [{"id": "t1", "name": "repo_info", "args": {"repo_url": "a/b"}}]},
        {"role": "tool", "results": [{"id": "t1", "name": "repo_info", "content": "{}", "is_error": True}]}]
TOOLS = [{"name": "repo_info", "description": "d", "input_schema": {"type": "object", "properties": {}}}]


def test_anthropic_wire_format(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, body=json, headers=headers)
        return FakeResp(200, {"content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "x", "name": "repo_info",
                                                                           "input": {"repo_url": "a/b"}}],
                              "stop_reason": "tool_use", "usage": {"input_tokens": 3, "output_tokens": 4}})
    monkeypatch.setattr("agent.llm.requests.post", post)
    t = AnthropicLLM(api_key="k", model="m").complete("sys", MSGS, TOOLS)
    assert t.text == "ok" and t.tool_calls[0].args == {"repo_url": "a/b"} and t.usage == {"in": 3, "out": 4}
    assert sent["body"]["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "{}", "is_error": True}
    assert sent["body"]["messages"][1]["content"][1]["type"] == "tool_use" and sent["headers"]["x-api-key"] == "k"


def test_openai_wire_format_and_bad_json_args(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent["body"] = json
        return FakeResp(200, {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "z", "type": "function", "function": {"name": "repo_info", "arguments": "{not json"}}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    monkeypatch.setattr("agent.llm.requests.post", post)
    t = OpenAICompatLLM(api_key="k", model="m").complete("sys", MSGS, TOOLS)
    assert "__malformed_json__" in t.tool_calls[0].args          # surfaced so the loop can tell the model
    w = sent["body"]["messages"]
    assert w[0]["role"] == "system" and w[2]["tool_calls"][0]["function"]["arguments"] == '{"repo_url": "a/b"}'
    assert w[3] == {"role": "tool", "tool_call_id": "t1", "content": "{}"}


def test_api_retry_then_success_and_fail_fast(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    calls = iter([FakeResp(529, {"e": 1}), FakeResp(429, {"e": 1}),
                  FakeResp(200, {"content": [{"type": "text", "text": "hi"}], "usage": {}})])
    monkeypatch.setattr("agent.llm.requests.post", lambda *a, **k: next(calls))
    assert AnthropicLLM(api_key="k").complete("s", MSGS[:1], TOOLS).text == "hi"
    monkeypatch.setattr("agent.llm.requests.post", lambda *a, **k: FakeResp(401, {"error": "bad key"}))
    with pytest.raises(LLMError):
        AnthropicLLM(api_key="k").complete("s", MSGS[:1], TOOLS)


def test_provider_selection(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(LLMError):
        make_llm(None)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert make_llm(None).name.startswith("openai-compatible")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert make_llm(None).name.startswith("anthropic")


def test_rate_limit_hint_is_honoured(monkeypatch):
    from agent.llm import _retry_after
    waits = []
    monkeypatch.setattr("agent.llm.time.sleep", waits.append)

    class R:  # Groq-style 429 with a hint in the body
        status_code, headers, text = 429, {}, "Rate limit reached ... Please try again in 7.5s. Need more tokens?"
        def json(self):
            return {}
    assert _retry_after(R()) == 7.5
    calls = iter([R(), FakeResp(200, {"content": [{"type": "text", "text": "ok"}], "usage": {}})])
    monkeypatch.setattr("agent.llm.requests.post", lambda *a, **k: next(calls))
    assert AnthropicLLM(api_key="k").complete("s", MSGS[:1], TOOLS).text == "ok"
    assert waits == [8.0]
