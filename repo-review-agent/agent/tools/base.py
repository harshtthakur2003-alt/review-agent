"""Tool abstraction + registry with input validation, output validation and timeouts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..models import ToolError

# A fallback inspects a failure and may propose corrected args (or None to give up).
Fallback = Callable[[dict, ToolError, "ToolRegistry"], Optional[tuple[dict, str]]]


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    required: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)  # keys a valid output must contain
    fallback: Optional[Fallback] = None


class ToolRegistry:
    def __init__(self, default_timeout: float = 30.0):
        self._tools: dict[str, Tool] = {}
        self.default_timeout = default_timeout
        self._pool = ThreadPoolExecutor(max_workers=4)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError("fatal", f"unknown tool '{name}'")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}({', '.join(t.required)}): {t.description}" for t in self._tools.values())

    def call(self, name: str, args: dict, timeout: Optional[float] = None) -> Any:
        tool = self.get(name)
        missing = [k for k in tool.required if args.get(k) in (None, "")]
        if missing:
            raise ToolError("invalid_input", f"{name}: missing/empty required args {missing}")
        future = self._pool.submit(tool.fn, **args)
        try:
            out = future.result(timeout=timeout or self.default_timeout)
        except FutureTimeout:
            raise ToolError("transient", f"{name}: timed out after {timeout or self.default_timeout}s")
        except ToolError:
            raise
        except TypeError as e:  # wrong/unexpected argument names produced by planner
            raise ToolError("invalid_input", f"{name}: bad arguments ({e})")
        except Exception as e:  # tool bug or unexpected library error: surface, don't crash the agent
            raise ToolError("fatal", f"{name}: unexpected {type(e).__name__}: {e}")
        self._validate_output(tool, out)
        return out

    @staticmethod
    def _validate_output(tool: Tool, out: Any) -> None:
        if tool.output_keys:
            if not isinstance(out, dict) or any(k not in out or out[k] is None for k in tool.output_keys):
                raise ToolError("invalid_output", f"{tool.name}: response missing keys {tool.output_keys}")
