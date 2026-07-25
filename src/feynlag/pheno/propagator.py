"""Internal propagators for off-shell (multi-body) decay amplitudes.

A 1→2 decay squares a single vertex; a 1→3 decay through a resonance is
*vertex × propagator × vertex*, and the propagator's momentum flows into the
trace algebra.  This module supplies the propagator pieces the assembler in
:mod:`~feynlag.pheno.offshell` contracts.

The **massive vector** numerator (for $h\to VV^*$) and the squared-modulus
:func:`breit_wigner` denominator were built for that 1→3 topology, where only
the already-squared line shape is ever needed.  :mod:`~feynlag.pheno.diagrams`
needs the propagator at *amplitude* level — before squaring, so its complex
phase survives into diagram interference — which :func:`breit_wigner` cannot
provide (it is $|1/(q^2-m^2+im\\Gamma)|^2$, not $1/(q^2-m^2+im\\Gamma)$).
:func:`propagator_denominator`, :func:`scalar_propagator` and
:func:`vector_propagator` fill that gap without touching the two squared-line
functions below; :func:`vector_propagator` reuses
:func:`vector_propagator_numerator` verbatim for its massive case.
"""

import sympy as sp

from .lorentz import LorentzIndex

__all__ = [
    "breit_wigner", "propagator_denominator", "scalar_propagator",
    "vector_propagator", "vector_propagator_numerator",
]


def breit_wigner(q2, mass, width):
    """The squared-modulus denominator ``1/((q²−m²)² + m²Γ²)``.

    This is $|1/(q^2 - m^2 + i m\\Gamma)|^2$ — the Breit–Wigner factor that
    turns a would-be on-shell pole into the finite off-shell line shape.  The
    resonance width ``Γ`` is itself a decay-calculator output (self-referential:
    the $W/Z$ width feeding the propagator comes from Tier-1).
    """
    q2, m, g = sp.sympify(q2), sp.sympify(mass), sp.sympify(width)
    return 1 / ((q2 - m**2)**2 + m**2 * g**2)


def vector_propagator_numerator(q, m, a, b):
    """The massive-vector numerator ``g_{ab} − q_a q_b/m²`` (both indices lower).

    Args:
        q: a callable ``q(index)`` giving the propagator four-momentum as a
            tensor expression (e.g. ``lambda i: p2(i) + p3(i)`` — the sum of the
            two fermion momenta).
        m: the vector mass.
        a, b: the two (upper) Lorentz ``TensorIndex`` objects; the numerator is
            returned with them **lowered** (``-a``, ``-b``) so it contracts with
            upper indices on the adjacent vertices.
    """
    metric = LorentzIndex.metric
    m = sp.sympify(m)
    return metric(-a, -b) - q(-a) * q(-b) / m**2


def propagator_denominator(q2, mass, width=0):
    """``q² − m² + i m Γ`` — the single source of truth for both the
    amplitude-level ``1/D`` propagators below and the squared
    :func:`breit_wigner` line shape (``breit_wigner(q2,m,Γ) ==
    1/(D·conjugate(D))`` for real ``q2, m, Γ``)."""
    q2, m, g = sp.sympify(q2), sp.sympify(mass), sp.sympify(width)
    return q2 - m**2 + sp.I * m * g


def scalar_propagator(q2, mass, width=0):
    """``i/(q² − m² + i m Γ)`` — the scalar Feynman propagator."""
    return sp.I / propagator_denominator(q2, mass, width)


def vector_propagator(q, q2, mass, a, b, width=0):
    """``−i(g_ab − q_a q_b/m²)/(q² − m² + i m Γ)``; massless (Feynman gauge)
    reduces to ``−i g_ab/q²``.

    Args:
        q: callable ``q(index) -> TensExpr``, as in
            :func:`vector_propagator_numerator`; unused (and may be ``None``)
            for a massless mediator.
        q2, mass, width: as in :func:`propagator_denominator`.
        a, b: the two upper ``TensorIndex`` objects; returned with them
            **lowered**, matching :func:`vector_propagator_numerator`.
    """
    metric = LorentzIndex.metric
    m = sp.sympify(mass)
    if m == 0:
        return -sp.I * metric(-a, -b) / sp.sympify(q2)
    num = vector_propagator_numerator(q, m, a, b)
    return -sp.I * num / propagator_denominator(q2, mass, width)
