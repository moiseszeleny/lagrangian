"""SymPy → UFO python-expression strings.

UFO ``value`` strings are Python expressions evaluated with ``cmath``
semantics: ``cmath.sqrt``, ``cmath.pi``, ``complex(0,1)`` for ``I``,
``complexconjugate`` from the UFO function library.
"""

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

__all__ = ["ufo_expr"]


class _UFOPrinter(PythonCodePrinter):
    """Prints SymPy expressions in UFO conventions."""

    def _print_ImaginaryUnit(self, expr):
        return "complex(0,1)"

    def _print_Pow(self, expr, rational=False):
        # UFO uses ** and cmath.sqrt
        if expr.exp == sp.Rational(1, 2):
            return f"cmath.sqrt({self._print(expr.base)})"
        if expr.exp == -sp.Rational(1, 2):
            return f"1/cmath.sqrt({self._print(expr.base)})"
        return super()._print_Pow(expr, rational=rational)

    def _print_Function(self, expr):
        name = expr.func.__name__
        if name == "conjugate":
            return f"complexconjugate({self._print(expr.args[0])})"
        return super()._print_Function(expr)

    def _print_conjugate(self, expr):
        return f"complexconjugate({self._print(expr.args[0])})"

    def _print_Abs(self, expr):
        return f"abs({self._print(expr.args[0])})"

    def _print_Pi(self, expr):
        return "cmath.pi"

    def _print_exp(self, expr):
        return f"cmath.exp({self._print(expr.args[0])})"

    def _print_log(self, expr):
        return f"cmath.log({self._print(expr.args[0])})"

    def _print_atan(self, expr):
        return f"cmath.atan({self._print(expr.args[0])})"

    def _print_sin(self, expr):
        return f"cmath.sin({self._print(expr.args[0])})"

    def _print_cos(self, expr):
        return f"cmath.cos({self._print(expr.args[0])})"

    def _print_tan(self, expr):
        return f"cmath.tan({self._print(expr.args[0])})"

    def _print_asin(self, expr):
        return f"cmath.asin({self._print(expr.args[0])})"

    def _print_acos(self, expr):
        return f"cmath.acos({self._print(expr.args[0])})"


_printer = _UFOPrinter({"fully_qualified_modules": False})


def ufo_expr(expr):
    """UFO-compatible Python expression string for a SymPy expression."""
    return _printer.doprint(sp.sympify(expr))
