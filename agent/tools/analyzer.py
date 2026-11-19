"""Static code-quality analysis (pure AST: the analysed code is never executed)."""
from __future__ import annotations

import ast
import re

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
