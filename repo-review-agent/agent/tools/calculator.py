"""Safe arithmetic evaluator (AST-whitelist; no eval())."""
from __future__ import annotations

import ast
import operator as op

from ..models import ToolError

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {"round": round, "max": max, "min": min, "abs": abs}


def _ev(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        if isinstance(node.op, ast.Pow):
            right = _ev(node.right)
            if abs(right) > 10:
                raise ToolError("invalid_input", "exponent too large")
            return _BIN[ast.Pow](_ev(node.left), right)
        return _BIN[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_ev(a) for a in node.args])
    raise ToolError("invalid_input", f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> dict:
    try:
        tree = ast.parse(str(expression), mode="eval")
        value = _ev(tree)
    except ZeroDivisionError:
        raise ToolError("invalid_input", "division by zero")
    except SyntaxError as e:
        raise ToolError("invalid_input", f"cannot parse expression: {e.msg}")
    return {"expression": expression, "value": value}
