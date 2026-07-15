"""Fixed physics and SymPy conventions for feynlag.

See CONVENTIONS.md at the repository root for the full statement of each
convention; every entry there is pinned by a test.
"""

import sympy as sp

#: Metric signature used everywhere: diag(+1, -1, -1, -1).
METRIC_SIGNATURE = "+---"

#: Explicit sqrt(2) used in VEV expansions: phi0 -> (v + h + I*a)/sqrt(2).
SQRT2 = sp.sqrt(2)


def tidy(expr, collect_syms=None):
    """House simplification chain, cheapest first: expand -> collect -> factor.

    ``simplify`` is deliberately NOT called here — it is the last resort and
    must be invoked explicitly by the caller when needed.

    Args:
        expr: SymPy expression.
        collect_syms: optional iterable of symbols to ``collect`` on.

    Returns:
        The tidied expression.
    """
    expr = sp.expand(expr)
    if collect_syms:
        expr = sp.collect(expr, list(collect_syms))
    try:
        return sp.factor(expr)
    except sp.PolynomialError:
        return expr
