"""Internal propagators for off-shell (multi-body) decay amplitudes.

A 1→2 decay squares a single vertex; a 1→3 decay through a resonance is
*vertex × propagator × vertex*, and the propagator's momentum flows into the
trace algebra.  This module supplies the propagator pieces the assembler in
:mod:`~feynlag.pheno.offshell` contracts.

Currently the **massive vector** propagator (for $h\to VV^*$); scalar and
fermion propagators are future entries (the general 1→3 framework).
"""

import sympy as sp

from .lorentz import LorentzIndex

__all__ = ["breit_wigner", "vector_propagator_numerator"]


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
