"""Model providers behind one small interface.

The agent talks to a model only through `LLMClient.complete(system, messages, tools) -> Turn`, using a
provider-neutral message format:

    {"role": "user",      "content": str}
    {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool",      "results": [{"id", "name", "content": str, "is_error": bool}]}

Adapters translate that to the Anthropic Messages API or any OpenAI-compatible Chat Completions API
(OpenAI, Gemini's OpenAI endpoint, Groq, OpenRouter, Ollama, ...). There is deliberately NO offline /
scripted backend here: every plan, tool choice, recovery and report is produced by a real model.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from .models import LLMError


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = ""
    usage: dict = field(default_factory=dict)


class LLMClient:
    name = "llm"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:  # pragma: no cover
        raise NotImplementedError


def _retry_after(r) -> Optional[float]:
    """Seconds the provider asked us to wait (Retry-After header or 'try again in 1.3s' in the body)."""
    h = (getattr(r, "headers", None) or {}).get("retry-after")
    try:
        if h:
            return float(h)
    except ValueError:
        pass
    m = re.search(r"try again in ([0-9.]+)\s*(ms|s)", r.text or "", re.I)
    if m:
        return float(m.group(1)) / (1000 if m.group(2).lower() == "ms" else 1)
    return None


def _post_with_retry(url: str, headers: dict, payload: dict, timeout: float, attempts: int = 8) -> dict:
    """Resilience for the model API itself: backoff on 429/5xx/network errors (honouring the provider's
    Retry-After hint, so free-tier tokens-per-minute limits are waited out), fail fast on other 4xx."""
    delay = 2.0
    last = ""
    for n in range(1, attempts + 1):
        wait = delay
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last = f"network error {type(e).__name__}"
        else:
            if r.status_code < 400:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 409, 425, 429, 500, 502, 503, 504, 529):
                raise LLMError(last)
            hint = _retry_after(r)
            if hint is not None:
                wait = max(wait, hint + 0.5)
        if n < attempts:
            time.sleep(min(wait, 60.0))
            delay = min(delay * 2, 30.0)
    raise LLMError(f"model API failed after {attempts} attempts ({last})")


class AnthropicLLM(LLMClient):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 120.0):
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        self.timeout = timeout
        self.name = f"anthropic:{self.model}"
        if not self.key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

    def complete(self, system, messages, tools) -> Turn:
        wire = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = ([{"type": "text", "text": m["text"]}] if m.get("text") else []) + [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]} for c in m.get("tool_calls", [])]
                wire.append({"role": "assistant", "content": blocks})
            else:
                wire.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"], "is_error": r["is_error"]}
                    for r in m["results"]]})
        data = _post_with_retry(
            f"{self.base}/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": self.model, "max_tokens": 4096, "system": system, "messages": wire,
             "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in tools]},
            self.timeout)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in data.get("content", []) if b.get("type") == "tool_use"]
        u = data.get("usage", {})
        return Turn(text, calls, data.get("stop_reason", ""), {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)})


class OpenAICompatLLM(LLMClient):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 120.0):
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4.1")
        self.key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout
        self.name = f"openai-compatible:{self.model}"
        if not self.key:
            raise LLMError("OPENAI_API_KEY is not set (for local servers such as Ollama set it to any non-empty value)")

    def complete(self, system, messages, tools) -> Turn:
        wire = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                         for c in m["tool_calls"]]
                wire.append(msg)
            else:
                wire += [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in m["results"]]
        data = _post_with_retry(
            f"{self.base}/chat/completions",
            {"Authorization": f"Bearer {self.key}", "content-type": "application/json"},
            {"model": self.model, "messages": wire,
             "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["input_schema"]}} for t in tools]},
            self.timeout)
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__malformed_json__": c["function"].get("arguments", "")}
            calls.append(ToolCall(c["id"], c["function"]["name"], args))
        u = data.get("usage", {})
        return Turn(msg.get("content") or "", calls, data["choices"][0].get("finish_reason", ""),
                    {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)})


def make_llm(provider: Optional[str], model: Optional[str] = None) -> LLMClient:
    """provider: anthropic | openai | None (auto-detect from whichever API key is set)."""
    if provider is None:
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai" if os.environ.get("OPENAI_API_KEY") else None
        if provider is None:
            raise LLMError("no model API key found: set ANTHROPIC_API_KEY or OPENAI_API_KEY (see README)")
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAICompatLLM(model)
    raise LLMError(f"unknown provider {provider!r}")
