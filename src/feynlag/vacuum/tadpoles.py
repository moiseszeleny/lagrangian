"""Tadpole conditions: extraction and solving.

DLRSM1 pattern, systematized: evaluate the potential on the vacuum and
differentiate with respect to each VEV; by the chain rule this equals the
linear-fluctuation coefficient ``⟨∂V/∂h⟩``.
"""

import sympy as sp

__all__ = ["extract_tadpoles", "solve_tadpoles"]


def extract_tadpoles(potential, vacuum):
    """Tadpole conditions ``{vev: ∂V/∂vev |_vacuum}`` (all must vanish).

    Args:
        potential: the scalar potential ``V`` (NOT ``−V``; feynlag stores the
            Lagrangian with ``L ⊃ −V``).
        vacuum: :class:`~feynlag.vacuum.ewsb.Vacuum`.
    """
    V0 = vacuum.at_vacuum(potential)
    tadpoles = {}
    for vev in vacuum.vevs:
        if not isinstance(vev, sp.Symbol):
            raise ValueError(f"cannot differentiate with respect to composite "
                             f"VEV {vev}; declare it as a plain symbol")
        tadpoles[vev] = sp.factor(sp.diff(V0, vev))
    return tadpoles


def solve_tadpoles(potential, vacuum, for_params):
    """Solve the tadpole conditions for the given parameters.

    Args:
        potential: scalar potential ``V``.
        vacuum: :class:`Vacuum`.
        for_params: parameters to solve for — Parameter objects or plain
            symbols, one per independent tadpole condition.  Any
            ``InternalParameter`` among them gets its ``expr`` defined with
            the solution.

    Returns:
        dict ``{symbol: solution}``.

    Raises:
        ValueError: if the system has no (unique) solution.
    """
    tadpoles = extract_tadpoles(potential, vacuum)
    symbols = [getattr(p, "symbol", p) for p in for_params]

    equations = [sp.Eq(t, 0) for t in tadpoles.values()]
    solutions = sp.solve(equations, symbols, dict=True)
    if not solutions:
        raise ValueError(f"tadpole conditions have no solution for {symbols}")
    if len(solutions) > 1:
        raise ValueError(f"tadpole conditions have {len(solutions)} solution "
                         f"branches for {symbols}; solve manually and use "
                         f"InternalParameter.define")
    solution = {s: sp.factor(sp.expand(e)) for s, e in solutions[0].items()}

    for p in for_params:
        if hasattr(p, "define"):
            p.define(solution[p.symbol])
    return solution
