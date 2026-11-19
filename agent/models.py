"""Shared exception types."""
from __future__ import annotations


class ToolError(Exception):
    """Raised by tools. `kind` tells the agent (and the LLM) what sort of failure this was.

    kinds: transient | rate_limited | not_found | invalid_input | invalid_output | fatal
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


class LLMError(Exception):
    """The model API failed in a way we could not recover from (auth, bad request, repeated 5xx)."""
