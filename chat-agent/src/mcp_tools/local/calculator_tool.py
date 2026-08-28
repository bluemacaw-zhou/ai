"""受限的确定性算术工具。"""

from __future__ import annotations

import ast
import operator
from decimal import Decimal, InvalidOperation
from typing import Annotated, Callable

from langchain_core.tools import tool

from mcp_tools.tool_result import build_tool_result, output_schema

_DATA_SCHEMA = {
    "type": "object",
    "description": "确定性算术结果。",
    "properties": {
        "expression": {"type": "string", "description": "原始算术表达式。"},
        "value": {"type": "number", "description": "计算结果。"},
        "decimal_value": {"type": "string", "description": "不使用科学计数法的十进制结果。"},
    },
}

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Decimal], Decimal]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@tool(
    "calculator_tool",
    description=(
        "执行确定性基本算术。支持数字、括号、加减乘除，不支持变量、函数或任意代码。"
        "入参 expression 必须先把上游结果替换成具体数值，例如 '2386.5 * 7.18'。"
        "比较、汇总等语言任务不使用本工具，但任何精确数值计算都必须使用本工具。"
    ),
    extras={"source": "local", "output_schema": output_schema(_DATA_SCHEMA)},
)
async def calculator_tool(
    expression: Annotated[str, "只含具体数值、括号和 + - * / 的算术表达式。"],
) -> list[dict]:
    value = _evaluate(ast.parse(expression, mode="eval").body)
    decimal_value = format(value.normalize(), "f")
    return build_tool_result(
        {
            "expression": expression,
            "value": float(value),
            "decimal_value": decimal_value,
        },
        _DATA_SCHEMA,
    )


def _evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        try:
            return Decimal(str(node.value))
        except InvalidOperation as exc:
            raise ValueError("表达式包含非法数值") from exc
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("表达式只能包含具体数值、括号和 + - * /")
