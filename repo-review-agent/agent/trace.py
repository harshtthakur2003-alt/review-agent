"""Tracing: every decision/tool call becomes an event, rendered live and persisted as JSONL."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, TextIO


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False) if not isinstance(value, str) else value
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Tracer:
    ICONS = {
        "plan": "PLAN ", "step": "STEP ", "call": "CALL ", "result": " OK  ", "error": "FAIL ",
        "recover": "FIX  ", "skip": "SKIP ", "info": "INFO ", "report": "DONE ",
    }

    def __init__(self, jsonl_path: Optional[Path] = None, stream: Optional[TextIO] = None, quiet: bool = False):
        self.events: list[dict] = []
        self.lines: list[str] = []
        self.stream = stream or sys.stdout
        self.quiet = quiet
        self._fh = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None
        self._t0 = time.monotonic()

    def emit(self, kind: str, message: str, **data: Any) -> None:
        ev = {"t": round(time.monotonic() - self._t0, 2), "kind": kind, "message": message, **data}
        self.events.append(ev)
        line = f"[{ev['t']:6.2f}s] {self.ICONS.get(kind, kind.upper()):5} {message}"
        self.lines.append(line)
        if not self.quiet:
            print(line, file=self.stream, flush=True)
        if self._fh:
            self._fh.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    @staticmethod
    def short(value: Any, limit: int = 160) -> str:
        return _short(value, limit)
