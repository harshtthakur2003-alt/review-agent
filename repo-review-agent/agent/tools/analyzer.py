"""Static code-quality analysis (pure AST: the analysed code is never executed)."""
from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Any

from ..models import ToolError

SEVERITY = {
    "syntax_error": "high", "bare_except": "high", "mutable_default_arg": "high", "eval_exec": "high",
    "swallowed_exception": "medium", "high_complexity": "medium", "long_function": "medium",
    "too_many_args": "medium", "wildcard_import": "medium",
    "missing_docstring": "low", "todo_comment": "low",
}
LONG_FUNCTION_LINES = 60
COMPLEXITY_LIMIT = 10
MAX_ARGS = 7
_TODO = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b", re.I)
_EXCLUDED_PARTS = ("test", "tests", "docs", "doc", "examples", "example", "vendor", "vendored", "_vendor",
                   "build", "node_modules", "benchmarks", "scripts")


def _complexity(fn: ast.AST) -> int:
    score = 1
    for n in ast.walk(fn):
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp, ast.Assert)):
            score += 1
        elif isinstance(n, ast.BoolOp):
            score += len(n.values) - 1
        elif isinstance(n, ast.comprehension):
            score += 1 + len(n.ifs)
    return score


def analyze_code(path: str, text: str) -> dict:
    lines = text.splitlines()
    loc = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
    issues: list[dict] = []

    def add(code: str, line: int, detail: str) -> None:
        issues.append({"code": code, "severity": SEVERITY[code], "path": path, "line": line, "detail": detail})

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        add("syntax_error", e.lineno or 0, f"cannot parse: {e.msg}")
        return {"path": path, "loc": loc, "functions": 0, "issues": issues}

    if not ast.get_docstring(tree) and loc > 20:
        add("missing_docstring", 1, "module has no docstring")
    n_funcs = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_funcs += 1
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > LONG_FUNCTION_LINES:
                add("long_function", node.lineno, f"{node.name}() is {length} lines (limit {LONG_FUNCTION_LINES})")
            cx = _complexity(node)
            if cx > COMPLEXITY_LIMIT:
                add("high_complexity", node.lineno, f"{node.name}() cyclomatic complexity ~{cx} (limit {COMPLEXITY_LIMIT})")
            a = node.args
            n_args = len(a.args) + len(a.kwonlyargs) + len(getattr(a, "posonlyargs", []))
            if n_args > MAX_ARGS:
                add("too_many_args", node.lineno, f"{node.name}() takes {n_args} parameters")
            for d in a.defaults + [d for d in a.kw_defaults if d is not None]:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)) or (
                        isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in ("list", "dict", "set")):
                    add("mutable_default_arg", d.lineno, f"{node.name}() uses a mutable default argument")
            if not node.name.startswith("_") and not ast.get_docstring(node) and length > 8:
                add("missing_docstring", node.lineno, f"public function {node.name}() lacks a docstring")
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_") and not ast.get_docstring(node):
                add("missing_docstring", node.lineno, f"public class {node.name} lacks a docstring")
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                add("bare_except", node.lineno, "bare 'except:' also catches KeyboardInterrupt/SystemExit")
            elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                add("swallowed_exception", node.lineno, "exception caught and silently ignored (pass)")
        elif isinstance(node, ast.ImportFrom) and any(al.name == "*" for al in node.names):
            add("wildcard_import", node.lineno, f"'from {node.module} import *'")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
            add("eval_exec", node.lineno, f"call to {node.func.id}()")
    for i, l in enumerate(lines, 1):
        if _TODO.search(l):
            add("todo_comment", i, l.strip()[:80])
    return {"path": path, "loc": loc, "functions": n_funcs, "issues": issues}


def select_python_files(files: list[dict], max_files: int = 8, max_file_bytes: int = 60000) -> dict:
    """Pick the most substantial *source* files: skip tests/docs/vendored/tiny/huge files."""
    def is_source(p: str) -> bool:
        parts = [s.lower() for s in p.split("/")]
        base = parts[-1]
        return not (any(s in _EXCLUDED_PARTS for s in parts[:-1]) or base.startswith("test_")
                    or base.endswith("_test.py") or base in ("setup.py", "conftest.py", "__init__.py"))

    cands = [f for f in files if is_source(f["path"]) and 800 <= f.get("size", 0) <= max_file_bytes]
    cands.sort(key=lambda f: -f["size"])
    chosen = cands[:max_files]
    if not chosen:
        raise ToolError("invalid_output", "no analysable Python source files found")
    return {"files": chosen, "considered": len(files), "eligible": len(cands)}


def summarize_findings(analyses: list[dict]) -> dict:
    if not analyses:
        raise ToolError("invalid_input", "no analyses to summarise")
    issues = [i for a in analyses for i in a["issues"]]
    by_code = Counter(i["code"] for i in issues)
    by_sev = Counter(i["severity"] for i in issues)
    weight = {"high": 3, "medium": 2, "low": 1}
    per_file = sorted(analyses, key=lambda a: -sum(weight[i["severity"]] for i in a["issues"]))
    order = {"high": 0, "medium": 1, "low": 2}
    top = sorted(issues, key=lambda i: (order[i["severity"]], i["path"], i["line"]))[:10]
    return {
        "files_analyzed": len(analyses), "loc": sum(a["loc"] for a in analyses),
        "issue_count": len(issues), "weighted_issue_points": sum(weight[i["severity"]] for i in issues),
        "by_code": dict(by_code), "by_severity": {k: by_sev.get(k, 0) for k in ("high", "medium", "low")},
        "worst_files": [{"path": a["path"], "issues": len(a["issues"]), "loc": a["loc"]} for a in per_file[:3]],
        "top_issues": top,
    }


def build_analyzer_tools(registry) -> None:
    from .base import Tool
    registry.register(Tool("analyze_code", "Static AST analysis of one Python file's text (complexity, bare except, ...).",
                           analyze_code, ["path", "text"], ["issues", "loc"]))
    registry.register(Tool("select_python_files", "Choose the most substantial non-test source files to review.",
                           select_python_files, ["files"], ["files"]))
    registry.register(Tool("summarize_findings", "Aggregate per-file analyses into totals and rankings.",
                           summarize_findings, ["analyses"], ["issue_count", "loc"]))
