from __future__ import annotations

from typing import Optional

from .analyzer import build_analyzer_tools
from .base import Tool, ToolRegistry
from .calculator import calculator
from .github import GitSource, build_github_tools


def build_registry(source: Optional[GitSource] = None, timeout: float = 90.0) -> ToolRegistry:
    reg = ToolRegistry(default_timeout=timeout)
    build_github_tools(reg, source or GitSource())
    build_analyzer_tools(reg)
    reg.register(Tool("calculator", "Safely evaluate an arithmetic expression (+,-,*,/,round,max,min).",
                      calculator, ["expression"], ["value"]))
    return reg


__all__ = ["build_registry", "Tool", "ToolRegistry", "GitSource"]
