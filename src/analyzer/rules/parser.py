"""Expresiones de las reglas: de texto a función evaluable, sin ``eval()``.

La expresión se parsea con ``ast`` y solo se aceptan estos nodos: nombres
(columnas del panel), constantes, aritmética básica, comparaciones (también
encadenadas), ``and``/``or``/``not`` y llamadas por nombre a funciones del
registro. Cualquier otra cosa (atributos, índices, lambdas, argumentos con
nombre...) es un error al compilar, no al evaluar.

Los nombres se resuelven contra un ``Scope``: cada variable es una serie del
panel o un escalar. Los operadores de pandas y numpy hacen el trabajo
vectorizado; una comparación con NaN es falsa, así que un valor sin
histórico suficiente no cumple ninguna condición.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import numpy as np

from analyzer.rules.functions import FUNCTIONS, Function

Value = Any  # pd.Series o escalar
Scope = Mapping[str, Value]
Node = Callable[[Scope], Value]

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class ExpressionError(ValueError):
    """Expresión que no se puede compilar o evaluar; el mensaje cita la expresión."""


@dataclass(frozen=True)
class Expression:
    source: str
    variables: frozenset[str]  # nombres que deben existir en el scope
    functions: frozenset[str]
    lookback: int  # sesiones previas a la ventana que necesitan sus funciones
    _node: Node = field(repr=False, compare=False)

    def __call__(self, scope: Scope) -> Value:
        return self._node(scope)


def compile_expression(source: str, functions: Mapping[str, Function] = FUNCTIONS) -> Expression:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"expresión inválida {source!r}: {exc.msg}") from exc
    compiler = _Compiler(source, functions)
    node = compiler.compile(tree.body)
    return Expression(
        source=source,
        variables=frozenset(compiler.variables),
        functions=frozenset(compiler.calls),
        lookback=compiler.lookback,
        _node=node,
    )


class _Compiler:
    def __init__(self, source: str, functions: Mapping[str, Function]) -> None:
        self.source = source
        self.functions = functions
        self.variables: set[str] = set()
        self.calls: set[str] = set()
        self.lookback = 0

    def compile(self, node: ast.expr) -> Node:
        match node:
            case ast.Constant(value=value) if isinstance(value, int | float | str):
                return lambda scope: value
            case ast.Name(id=name):
                self.variables.add(name)
                return lambda scope: self._lookup(scope, name)
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                inner = self.compile(operand)
                return lambda scope: -inner(scope)
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                inner = self.compile(operand)
                return lambda scope: np.logical_not(inner(scope))
            case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
                apply, lhs, rhs = _BINARY[type(op)], self.compile(left), self.compile(right)
                return lambda scope: apply(lhs(scope), rhs(scope))
            case ast.BoolOp(op=op, values=values):
                join = np.logical_and if isinstance(op, ast.And) else np.logical_or
                parts = [self.compile(v) for v in values]
                return lambda scope: reduce(join, (p(scope) for p in parts))
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                return self._compare(left, ops, comparators)
            case ast.Call(func=ast.Name(id=name), args=args, keywords=[]):
                return self._call(name, args)
            case _:
                raise ExpressionError(
                    f"{self.source!r}: no se permite {type(node).__name__} en una regla"
                )

    def _compare(self, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr]) -> Node:
        operands = [self.compile(left), *(self.compile(c) for c in comparators)]
        applies = []
        for op in ops:
            if type(op) not in _COMPARE:
                raise ExpressionError(
                    f"{self.source!r}: comparación no permitida {type(op).__name__}"
                )
            applies.append(_COMPARE[type(op)])

        def evaluate(scope: Scope) -> Value:
            values = [o(scope) for o in operands]
            checks = [
                apply(a, b) for apply, a, b in zip(applies, values[:-1], values[1:], strict=True)
            ]
            return reduce(np.logical_and, checks)

        return evaluate

    def _call(self, name: str, args: list[ast.expr]) -> Node:
        function = self.functions.get(name)
        if function is None:
            raise ExpressionError(f"{self.source!r}: función desconocida {name!r}")
        if len(args) != function.arity:
            raise ExpressionError(
                f"{self.source!r}: {name} espera {function.arity} argumentos, recibe {len(args)}"
            )
        self.calls.add(name)
        self.lookback = max(self.lookback, function.lookback)
        compiled = [self.compile(a) for a in args]
        return lambda scope: function.call(*(c(scope) for c in compiled))

    def _lookup(self, scope: Scope, name: str) -> Value:
        try:
            return scope[name]
        except KeyError:
            raise ExpressionError(f"{self.source!r}: variable desconocida {name!r}") from None
