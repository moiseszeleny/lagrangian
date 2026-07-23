"""Off-shell (1→3) decay amplitudes: a scalar to one on-shell + one off-shell V.

Tier 2 of the decays roadmap.  On-shell ``VVS`` gives zero below threshold
(``2m_V > M``); the physical decay is ``S → V (V^* → f f̄')`` — a 1→3 process
with an internal Breit–Wigner propagator.  This module assembles the squared
amplitude with the covariant engine (the internal ``q = p₂+p₃`` flows into the
trace exactly like an external momentum) and integrates it over the Dalitz
region.

The current topology is ``SVV`` (scalar → vector + vector*→fermions); the
assembler is written so other 1→3 topologies (FFFF, ``t→bW^*``) can be added.

Two width routes, cross-checked against each other and the Keung–Marciano closed
form: the **numeric** Dalitz integral (:mod:`~feynlag.pheno.integrate`) and, for
massless fermions, an **analytic** SymPy integration.
"""

import numpy as np
import sympy as sp
from sympy.physics.hep.gamma_matrices import GammaMatrix, LorentzIndex

from .kinematics import ThreeBodyKinematics
from .lorentz import contract_to_dots, dirac_trace, index, slashed
from .propagator import vector_propagator_numerator

__all__ = ["scalar_vv_squared", "scalar_vv_s12_integral",
           "offshell_scalar_vv_width", "scalar_offshell_vv_width"]


def scalar_vv_squared(m_V, kin):
    """The reduced ``|M|²`` for ``S → V(p₁) [V^*→ f(p₂) f̄'(p₃)]``.

    Returns the spin-summed squared amplitude **stripped of** the coupling
    factor ``g_{SVV}² g_V² / |q²−m_V²+i m_V Γ_V|²`` (reinstated by
    :func:`offshell_scalar_vv_width`), as a symbolic function of the Dalitz
    invariants ``s₁₂, s₂₃`` (``s₂₃ = q²``).  Massless fermions, left-handed
    ``V f f̄'`` current (the parity-violating ε term drops against the symmetric
    propagator/polarisation structure — see the roadmap §16.2).

    Assembly (all contracted by the covariant engine):

    ``P^{μμ'}_{V}(p₁) · (g_{μα})(g_{μ'α'}) · N_{αβ}(q) N_{α'β'}(q) ·
    ½ Tr[p̸₂ γ^β p̸₃ γ^{β'}]``

    where ``P_V`` is the on-shell-V polarisation sum, ``N`` the propagator
    numerator, and the ``½`` is the left-handed projector (its ε piece dropped).
    """
    mV = sp.sympify(m_V)
    p1, p2, p3 = kin.p1, kin.p2, kin.p3
    a, ap, b, bp = index("a"), index("ap"), index("b"), index("bp")
    g = LorentzIndex.metric

    def q(i):
        return p2(i) + p3(i)

    # on-shell V(p₁) polarisation sum, upper (a, a')
    pol = -g(a, ap) + p1(a) * p1(ap) / mV**2
    n1 = vector_propagator_numerator(q, mV, a, b)      # lower (a, b)
    n2 = vector_propagator_numerator(q, mV, ap, bp)    # lower (a', b')
    chain = slashed(p2, index("m1")) * GammaMatrix(b) \
        * slashed(p3, index("m2")) * GammaMatrix(bp)
    fermion = sp.Rational(1, 2) * dirac_trace(chain)   # ½ = P_L; ε drops

    expr = (pol * n1 * n2 * fermion).contract_metric(g)
    return sp.expand(contract_to_dots(expr, kin.dot))


def scalar_vv_s12_integral(m_S, m_V):
    """The inner Dalitz integral ``∫ |M|² ds₁₂`` done **analytically**.

    ``|M|²`` is polynomial in ``s₁₂``, so the inner integral over the Dalitz
    limits is a closed form (a Källén square root times a polynomial in ``s₂₃``)
    — SymPy does it in a fraction of a second, where the full 2-D integral with
    the propagator does not.  Returns a symbolic expression in ``s₂₃`` (with
    ``m_S, m_V`` symbolic), the integrand of the remaining 1-D ``s₂₃`` integral.
    """
    kin = ThreeBodyKinematics(m_S, m_V, sp.S.Zero, sp.S.Zero)
    s12, s23 = kin.s12, kin.s23
    m2red = scalar_vv_squared(m_V, kin)
    lo, hi = kin.s12_bounds(s23)
    return sp.simplify(sp.integrate(m2red, (s12, lo, hi)))


# cache the symbolic inner integral (mass-symbol independent shape)
_S12_CACHE = {}


def offshell_scalar_vv_width(m_S, m_V, width_V, coupling, backend="auto",
                             n=400):
    """``Γ(S → V f f̄')`` for one fermion channel.

    Analytic inner ``s₁₂`` integral (:func:`scalar_vv_s12_integral`) then a 1-D
    numerical integral over ``s₂₃ = q²`` with the Breit–Wigner
    ``1/((q²−m_V²)² + m_V²Γ_V²)``.

    Args:
        m_S: parent scalar mass (number).
        m_V: vector mass (number).
        width_V: the vector total width ``Γ_V`` (feeds the Breit–Wigner; below
            threshold, ``q²`` never reaches ``m_V²`` so this is negligible).
        coupling: the product ``g_{SVV}² g_V²`` (number).
        backend: ``"scipy"``, ``"gauss"`` (numpy) or ``"auto"``.
        n: Gauss–Legendre node count for the 1-D integral.

    Returns:
        the partial width for **one** ``f f̄'`` channel (float).  The caller
        multiplies by the open ``V^*`` fermion channels and (for ``S→VV*``) the
        factor 2 for which ``V`` is on-shell.
    """
    m_S, m_V = float(m_S), float(m_V)
    mSs, mVs, s23 = sp.symbols("m_S m_V s23", positive=True)
    if "svv" not in _S12_CACHE:
        _S12_CACHE["svv"] = scalar_vv_s12_integral(mSs, mVs)
    inner = _S12_CACHE["svv"]
    inner_f = sp.lambdify((s23, mSs, mVs), inner, "numpy")

    coup = float(coupling)
    s23_hi = (m_S - m_V)**2

    def integrand(s23v):
        bw = 1.0 / ((s23v - m_V**2)**2 + m_V**2 * width_V**2)
        return coup * bw * float(inner_f(s23v, m_S, m_V))

    if backend == "auto":
        from .integrate import have_scipy
        backend = "scipy" if have_scipy() else "gauss"
    if backend == "scipy":
        from scipy.integrate import quad
        integral, _ = quad(integrand, 0.0, s23_hi)
    else:
        x, w = np.polynomial.legendre.leggauss(n)
        half, mid = 0.5 * s23_hi, 0.5 * s23_hi
        integral = float(sum(wk * integrand(half * xk + mid)
                             for xk, wk in zip(x, w)) * half)
    const = 1.0 / ((2 * np.pi)**3 * 32 * m_S**3)
    return const * integral


def scalar_offshell_vv_width(m_S, m_V, width_V, g_SVV, channels,
                             identical=False, backend="auto"):
    r"""``Γ(S → V V^* → V f f̄')`` summed over the ``V^*`` fermion channels.

    Args:
        m_S: parent scalar mass.
        m_V: vector mass.
        width_V: the vector total width $\Gamma_V$ (Breit–Wigner).
        g_SVV: the ``SVV`` coupling (the Feynman rule with its ``i`` stripped,
            e.g. ``g m_V`` for ``hWW``).
        channels: iterable of ``(g_L, g_R, multiplicity)`` for each distinct
            ``V^*→f f̄'`` fermion channel — ``g_L, g_R`` the chiral couplings of
            the ``V f f̄'`` current, ``multiplicity`` a colour/flavour count.
        identical: ``False`` for ``S→V₁V₂*`` with **distinct** vectors (e.g.
            ``WW``: either ``W⁺`` or ``W⁻`` is the on-shell one → an overall
            factor 2); ``True`` for identical vectors (``ZZ``: no factor 2).

    Returns:
        the total off-shell width (float).  Reproduces the Keung–Marciano
        closed form (``Γ(h→WW*)≈0.80`` MeV, ``Γ(h→ZZ*)≈0.089`` MeV).
    """
    unit = offshell_scalar_vv_width(m_S, m_V, width_V, 1.0, backend=backend)
    g2 = float(g_SVV)**2
    coupling_sum = sum((float(gL)**2 + float(gR)**2) * float(mult)
                       for gL, gR, mult in channels)
    config = 1.0 if identical else 2.0
    return unit * g2 * coupling_sum * config
