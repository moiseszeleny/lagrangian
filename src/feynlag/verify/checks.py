"""Executable physics invariants — the house verification toolkit.

Ported from ``bsm-calc/physics_utils.py``.  Philosophy: encode physical
invariants as executable checks; a failing invariant stops the pipeline before
the error propagates.  All functions operate on SymPy objects and are
model-agnostic.

House rule: every physical result gets BOTH a symbolic check and a
random-point numeric check (:func:`numeric_equal`) — the numeric check defends
against the canonical-form problem where two identical expressions differ
syntactically and defeat ``simplify``.
"""

import random

import sympy as sp

from ..vertices.extract import extract_interaction_coefficients

__all__ = [
    "is_hermitian", "is_symmetric", "check_dimension", "seesaw_light_mass",
    "decoupling_limit", "diagonalize_hermitian", "numeric_equal",
    "round_trip_reconstruct",
]


def is_hermitian(M):
    """Whether ``M = M†`` (simplifies the difference symbolically)."""
    M = sp.Matrix(M)
    return sp.simplify(M - M.conjugate().T) == sp.zeros(*M.shape)


def is_symmetric(M):
    """Whether ``M = Mᵀ`` — e.g. Majorana mass matrices."""
    M = sp.Matrix(M)
    return sp.simplify(M - M.T) == sp.zeros(*M.shape)


def check_dimension(expr, symbol_dims, target_dim):
    """Symbolic dimensional analysis in powers of mass.

    Args:
        expr: SymPy expression.
        symbol_dims: dict ``{symbol: mass_dimension}``, e.g. ``{m: 1, Y: 0}``.
        target_dim: expected mass dimension of ``expr``.

    Returns:
        ``(ok: bool, found_dim)``.
    """
    u = sp.Symbol("__mass_unit__", positive=True)
    subs = {s: u ** d for s, d in symbol_dims.items()}
    powered = sp.simplify(expr.subs(subs))
    poly = sp.together(powered)
    found = (sp.degree(sp.numer(poly), u) - sp.degree(sp.denom(poly), u)
             if poly.has(u) else 0)
    return (found == target_dim, found)


def seesaw_light_mass(M_D, M_R):
    """Type-I seesaw at leading order: ``m_ν ≈ −M_D M_R⁻¹ M_Dᵀ``."""
    M_D = sp.Matrix(M_D)
    M_R = sp.Matrix(M_R)
    return sp.simplify(-M_D * M_R.inv() * M_D.T)


def decoupling_limit(expr, heavy_scale, direction="oo"):
    """Limit of ``expr`` as ``heavy_scale → ∞`` (or ``→ 0``)."""
    point = sp.oo if direction == "oo" else 0
    return sp.simplify(sp.limit(expr, heavy_scale, point))


def diagonalize_hermitian(M):
    """Diagonalize a hermitian matrix: returns ``(P, D)`` with ``M = P D P†``.

    For symbolic use in small cases; prefer numeric methods for large ones.
    """
    M = sp.Matrix(M)
    P, D = M.diagonalize()
    return P, D


def numeric_equal(expr_a, expr_b, symbols, n_points=20, tol=1e-10,
                  sample_range=(0.1, 10.0), seed=None):
    """Random-point NUMERIC equality of two expressions.

    Defense against the canonical-form problem: physically identical
    expressions can differ syntactically and defeat ``simplify``.  Evaluates
    both at ``n_points`` random points and compares with relative tolerance.
    Complements (never replaces) the symbolic check.

    Returns:
        ``(ok: bool, max_observed_relative_diff)``.

    Note: samples positive reals by default; adjust ``sample_range`` for
    expressions with singularities.
    """
    rng = random.Random(seed)
    fa = sp.lambdify(symbols, expr_a, modules="mpmath")
    fb = sp.lambdify(symbols, expr_b, modules="mpmath")
    max_diff = 0.0
    for _ in range(n_points):
        vals = [rng.uniform(*sample_range) for _ in symbols]
        va, vb = complex(fa(*vals)), complex(fb(*vals))
        denom = max(abs(va), abs(vb), 1.0)
        max_diff = max(max_diff, abs(va - vb) / denom)
    return (max_diff < tol, max_diff)


def round_trip_reconstruct(lagrangian, fields):
    """Round-trip validation of the vertex extractor.

    Extracts interaction coefficients from ``lagrangian``, reconstructs the
    Lagrangian from the coefficient dict, and returns the residual
    ``expand(original − reconstructed)`` (zero on success), together with a
    boolean verdict.

    Returns:
        ``(ok: bool, residual)`` — ``ok`` is True when the residual vanishes
        or contains no field symbols (pure-parameter constant terms are not
        interactions and may legitimately differ).
    """
    interactions = extract_interaction_coefficients(lagrangian, fields)

    reconstructed = sp.Add(*[
        coeff * sp.Mul(*field_tuple)
        for terms in interactions.values()
        for field_tuple, coeff in terms.items()
    ])

    residual = (sp.expand(lagrangian) - sp.expand(reconstructed))
    residual = sp.expand(residual)
    fields_set = {f for f in fields if isinstance(f, sp.Symbol)}
    ok = residual == 0 or residual.free_symbols.isdisjoint(fields_set)
    return ok, residual
