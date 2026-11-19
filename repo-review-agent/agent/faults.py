"""Deliberate fault injection so recovery behaviour can be demonstrated and tested.

Spec format: "<tool>:<mode>[:<times>]"   e.g.  repo_list_files:timeout:2
modes: timeout (transient failure) | rate_limit | malformed (garbage response) | not_found
The first <times> calls (default 1) to that tool are sabotaged; later calls behave normally.
"""
from __future__ import annotations

from collections import defaultdict

from .models import ToolError
from .tools.base import ToolRegistry


def parse_spec(spec: str) -> tuple[str, str, int]:
    parts = spec.split(":")
    if len(parts) not in (2, 3) or parts[1] not in ("timeout", "rate_limit", "malformed", "not_found"):
        raise ValueError(f"bad fault spec {spec!r}; expected tool:mode[:times] with mode in timeout|rate_limit|malformed|not_found")
    return parts[0], parts[1], int(parts[2]) if len(parts) == 3 else 1


def inject_faults(registry: ToolRegistry, specs: list[str]) -> None:
    """Several specs may target the same tool; each keeps its own budget and they fire in the order given."""
    rules: dict[str, list[list]] = defaultdict(list)  # tool -> [[mode, remaining], ...]
    for s in specs:
        tool, mode, times = parse_spec(s)
        registry.get(tool)  # fail fast on typos
        rules[tool].append([mode, times])
    real_call = registry.call

    def faulty_call(name, args, timeout=None):
        for rule in rules.get(name, []):
            if rule[1] > 0:
                rule[1] -= 1
                mode = rule[0]
                if mode == "timeout":
                    raise ToolError("transient", f"{name}: timed out (INJECTED fault)")
                if mode == "rate_limit":
                    raise ToolError("rate_limited", f"{name}: HTTP 429 (INJECTED fault)")
                if mode == "not_found":
                    raise ToolError("not_found", f"{name}: HTTP 404 (INJECTED fault)")
                registry._validate_output(registry.get(name), {"garbage": True})  # malformed -> raises invalid_output
        return real_call(name, args, timeout)

    registry.call = faulty_call  # type: ignore[method-assign]
