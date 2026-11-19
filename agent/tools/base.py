"""Tool abstraction: every tool has a JSON schema (shown to the LLM), input validation, a hard timeout
and output-shape validation. Failures become typed ToolErrors that are fed back to the model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..models import ToolError


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    properties: dict[str, dict]                      # JSON-schema "properties"
    required: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)  # keys a valid output must contain

    def schema(self) -> dict:
        return {"type": "object", "properties": self.properties, "required": self.required}


class ToolRegistry:
    def __init__(self, default_timeout: float = 60.0):
        self._tools: dict[str, Tool] = {}
        self.default_timeout = default_timeout
        self._pool = ThreadPoolExecutor(max_workers=4)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError("invalid_input", f"unknown tool '{name}'. Available: {', '.join(self.names())}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict]:
        """Provider-neutral tool specs: name / description / JSON-schema input."""
        return [{"name": t.name, "description": t.description, "input_schema": t.schema()} for t in self._tools.values()]

    def call(self, name: str, args: dict, timeout: Optional[float] = None) -> Any:
        tool = self.get(name)
        if not isinstance(args, dict):
            raise ToolError("invalid_input", f"{name}: arguments must be an object")
        missing = [k for k in tool.required if args.get(k) in (None, "")]
        if missing:
            raise ToolError("invalid_input", f"{name}: missing/empty required args {missing}")
        unknown = [k for k in args if k not in tool.properties]
        if unknown:
            raise ToolError("invalid_input", f"{name}: unknown args {unknown}; allowed: {list(tool.properties)}")
        future = self._pool.submit(tool.fn, **args)
        limit = timeout or self.default_timeout
        try:
            out = future.result(timeout=limit)
        except FutureTimeout:
            raise ToolError("transient", f"{name}: timed out after {limit}s")
        except ToolError:
            raise
        except TypeError as e:
            raise ToolError("invalid_input", f"{name}: bad arguments ({e})")
        except Exception as e:  # tool bug / library error: surface it, never crash the agent
            raise ToolError("fatal", f"{name}: unexpected {type(e).__name__}: {e}")
        self._validate_output(tool, out)
        return out

    @staticmethod
    def _validate_output(tool: Tool, out: Any) -> None:
        if tool.output_keys:
            if not isinstance(out, dict) or any(k not in out or out[k] is None for k in tool.output_keys):
                raise ToolError("invalid_output", f"{tool.name}: response missing keys {tool.output_keys}")
