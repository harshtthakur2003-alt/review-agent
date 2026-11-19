"""Core data structures shared by every component of the agent."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class ToolError(Exception):
    """Raised by tools. `kind` drives the recovery policy.

    kinds: transient | rate_limited | not_found | invalid_input | invalid_output | fatal
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.kind}] {self.message}"


class Clarification(Exception):
    """Raised by the planner when the goal is too ambiguous to act on."""


@dataclass
class Step:
    id: str
    description: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    # Optional reference (e.g. "$s4.results") to a list; the step runs once per item,
    # with the current element available as "$item".
    foreach: Optional[str] = None
    rationale: str = ""
    # Steps that may fail without invalidating the whole run (output is "nice to have").
    optional: bool = False


@dataclass
class Plan:
    goal: str
    reasoning: str
    steps: list[Step]
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepResult:
    step_id: str
    status: str  # ok | partial | failed | skipped
    output: Any = None
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    recoveries: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
