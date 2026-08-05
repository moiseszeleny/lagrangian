"""Spin-averaged squared amplitudes for the three-leg vertex catalog.

Each function returns ``⟨|M|²⟩`` — summed over final-state spins and
polarizations and **averaged** over the parent's — so that

``Γ = ⟨|M|²⟩ × √λ(M², m₁², m₂²)/(16 π M³)``

with the phase-space factor from :mod:`~feynlag.pheno.kinematics`.

Every Dirac trace goes through the covariant engine in
:mod:`~feynlag.pheno.lorentz`; nothing here is a hard-coded textbook formula.
The one place a result is written down directly is ``Tr[P_{L,R}] = 2``, which
is the *definition* of the projector reduction rather than an output of it.

Covered: the three-leg catalog subset a 1→2 decay can use — **SSS, FFS, FFV,
VVS, VSS**.  ``VSS`` is the one derivative coupling here: it carries feynlag
momentum tags (``p(φ)``, see
:func:`~feynlag.operators.to_momentum_space`), which :func:`vss_squared`
resolves into the physical momenta before contracting.  ``VVV`` still raises
rather than being silently skipped, matching
:func:`~feynlag.vertices.vertex.classify_spins`.
"""

import sympy as sp
from sympy.physics.hep.gamma_matrices import GammaMatrix, LorentzIndex

from ..operators import momentum as tag
from .lorentz import contract_to_dots, dirac_trace, index, reduce_projectors, slashed

__all__ = [
    "SUPPORTED_VERTEX_TYPES", "amplitude_squared", "ffs_squared",
    "ffv_squared", "polarization_sum", "spin_sum", "sss_squared",
    "vss_squared", "vss_vector_leg", "vvs_squared",
]

#: three-leg catalog entries this module can square
SUPPORTED_VERTEX_TYPES = ("SSS", "FFS", "FFV", "VVS", "VSS")


def spin_sum(p, mass, dummy, anti=False):
    """Fermion spin sum ``Σ u ū = p̸ + m`` (``Σ v v̄ = p̸ − m`` with ``anti``)."""
    return slashed(p, dummy) + (-1 if anti else 1) * sp.sympify(mass)


def polarization_sum(p, mass, a, b):
    """Vector polarization sum ``Σ ε_a ε*_b = −g_ab + p_a p_b/m²``.

    For ``mass == 0`` the longitudinal piece is absent and the sum reduces to
    ``−g_ab``: legitimate because every vertex feynlag extracts is contracted
    with a conserved current, so the gauge-dependent remainder cancels in
    ``|M|²``.
    """
    metric = LorentzIndex.metric
    if sp.sympify(mass) == 0:
        return -metric(a, b)
    return -metric(a, b) + p(a) * p(b) / sp.sympify(mass)**2


# ------------------------------------------------------------------ scalars

def sss_squared(coupling, kin):
    """``S → S S``: a contact scalar vertex has no Lorentz structure, so
    ``⟨|M|²⟩ = |c|²``."""
    return sp.Abs(sp.sympify(coupling))**2


# ----------------------------------------------------------------- fermions

def ffs_squared(g_left, g_right, kin):
    """``S(P) → f(p₁) f̄(p₂)`` for ``Γ = g_L P_L + g_R P_R``.

    ``Σ_spins |M|² = Tr[(p̸₁+m₁) Γ (p̸₂−m₂) Γ̄]`` with ``Γ̄ = ḡ_L P_R + ḡ_R P_L``
    (:func:`~feynlag.dirac.dirac_conjugate`: bare projectors swap).

    Pushing projectors right with ``P_{L,R} p̸ = p̸ P_{R,L}``:

    - **chirality-diagonal** (``|g_L|²``, ``|g_R|²``) — the two mass insertions
      are killed by ``P_L P_R = 0``, leaving ``½Tr[p̸₁p̸₂] = 2(p₁·p₂)``;
    - **chirality-mixing** (``g_L ḡ_R`` + c.c.) — the momentum terms die
      instead and ``m₁(−m₂)Tr[P_L] = −2m₁m₂`` survives.

    So ``Σ|M|² = 2(|g_L|²+|g_R|²)(p₁·p₂) − 4 m₁m₂ Re(g_L ḡ_R)``, and a scalar
    parent needs no spin average.
    """
    gL, gR = sp.sympify(g_left), sp.sympify(g_right)
    total = sp.S.Zero

    if gL != 0 or gR != 0:
        chain = slashed(kin.p1, index("mu_ffs")) * slashed(kin.p2, index("nu_ffs"))
        expr, factor = reduce_projectors(chain, "L", n_free_indices=0,
                                         n_momenta=2)
        diagonal = contract_to_dots(dirac_trace(expr), kin.dot) * factor
        total += (sp.Abs(gL)**2 + sp.Abs(gR)**2) * diagonal

    if gL != 0 and gR != 0:
        # m₁(−m₂)·Tr[P_L] = −2m₁m₂, once for g_L ḡ_R and once for its c.c.
        total += 2 * sp.re(gL * sp.conjugate(gR)) * (-2 * kin.m1 * kin.m2)

    return sp.expand(total)


def ffv_squared(g_left, g_right, kin, vector_mass=None):
    """``V(P) → f(p₁) f̄(p₂)`` for ``Γ^μ = γ^μ(g_L P_L + g_R P_R)``.

    ``Γ̄^μ = γ^μ(ḡ_L P_L + ḡ_R P_R)`` — a vector current is self-conjugate, the
    projector does **not** swap (see :func:`~feynlag.dirac.dirac_conjugate`).
    The same projector push-through as in :func:`ffs_squared` gives

    - chirality-diagonal: ``½Tr[p̸₁γ^a p̸₂γ^b]`` (mass terms killed),
    - chirality-mixing:  ``−m₁m₂ Tr[γ^aγ^b P_R] = −2m₁m₂ g^{ab}``,

    contracted with ``Σ ε_a ε*_b`` and averaged over the parent's three
    polarizations.
    """
    a, b = index("a_ffv"), index("b_ffv")
    gL, gR = sp.sympify(g_left), sp.sympify(g_right)
    M = kin.M if vector_mass is None else sp.sympify(vector_mass)

    tensor = sp.S.Zero
    if gL != 0 or gR != 0:
        chain = (slashed(kin.p1, index("mu_ffv")) * GammaMatrix(a)
                 * slashed(kin.p2, index("nu_ffv")) * GammaMatrix(b))
        expr, factor = reduce_projectors(chain, "L", n_free_indices=2,
                                         n_momenta=2)
        tensor += (sp.Abs(gL)**2 + sp.Abs(gR)**2) * factor * dirac_trace(expr)

    if gL != 0 and gR != 0:
        # ½Tr[γ^aγ^b] = 2g^{ab}, times m₁(−m₂), twice (term + c.c.)
        tensor += (2 * sp.re(gL * sp.conjugate(gR))
                   * (-2 * kin.m1 * kin.m2) * LorentzIndex.metric(a, b))

    P = lambda i: kin.p1(i) + kin.p2(i)
    contracted = (tensor * polarization_sum(P, M, -a, -b))
    contracted = contracted.contract_metric(LorentzIndex.metric)
    return sp.expand(contract_to_dots(contracted, kin.dot) / 3)


# ------------------------------------------------------------------ vectors

def vvs_squared(coupling, kin, m_v1=None, m_v2=None):
    """``S(P) → V(p₁) V(p₂)`` from a ``c·g^{μν}`` vertex (e.g. ``h W⁺W⁻``).

    With no spin average (scalar parent)::

        Σ|M|² = |c|² (−g_ab + p₁_a p₁_b/m₁²)(−g^{ab} + p₂^a p₂^b/m₂²)
              = |c|² [2 + (p₁·p₂)²/(m₁² m₂²)]
    """
    a, b = index("a_vvs"), index("b_vvs")
    m1 = kin.m1 if m_v1 is None else sp.sympify(m_v1)
    m2 = kin.m2 if m_v2 is None else sp.sympify(m_v2)
    expr = (polarization_sum(kin.p1, m1, a, b)
            * polarization_sum(kin.p2, m2, -a, -b))
    expr = expr.contract_metric(LorentzIndex.metric)
    return sp.expand(sp.Abs(sp.sympify(coupling))**2
                     * contract_to_dots(expr, kin.dot))


# --------------------------------------------------- vector + two scalars

def vss_vector_leg(vertex):
    """Which leg of a ``VSS`` vertex is the vector.

    Identified from the vertex itself rather than from leg ordering (which is
    alphabetical, not spin-sorted): the derivative in
    ``g V^μ(S ∂_μ S' − S' ∂_μ S)`` acts on the **scalars**, so exactly the two
    scalar legs carry a ``p(leg)`` momentum tag and the vector carries none.

    Raises:
        ValueError: if the tag pattern is not one untagged leg out of three —
            which means the coupling is not the derivative structure this
            module knows how to square.
    """
    legs = list(vertex.particles)
    untagged = [leg for leg in legs
                if sp.diff(vertex.coupling, tag(leg)) == 0]
    if len(untagged) != 1:
        raise ValueError(
            f"VSS vertex {tuple(legs)} has {len(untagged)} legs without a "
            f"p(leg) momentum tag; expected exactly one (the vector). "
            f"Coupling: {vertex.coupling}")
    return untagged[0]


def vss_squared(coupling, kin, legs, vector_leg, parent):
    """``⟨|M|²⟩`` for a derivative ``V S S'`` vertex, both leg assignments.

    The extracted coupling is linear in the momentum tags,
    ``c(p(S) − p(S'))``; feynlag's convention is ``∂_μφ → i p(φ) φ``, so with
    every leg taken **incoming** the amplitude is

        ``M = ε*_μ Σ_legs (∂coupling/∂p(leg)) q_leg^μ``,
        ``q_leg = +P`` for the parent, ``−p_i`` for a daughter.

    Nothing is hard-coded: the polarization sum contracts the momentum
    structure covariantly, exactly as :func:`vvs_squared` does. Both
    topologies are handled —

    * scalar parent, ``S → V S'``: no spin average;
    * vector parent, ``V → S S'``: averaged over the parent's 3 polarizations.

    ``p_V`` is annihilated by its own polarization sum
    (``p^μ(−g_{μν} + p_μp_ν/m²) = 0``), so the vector's own momentum drops out
    of ``M`` automatically rather than by hand.

    Args:
        coupling: the full Feynman rule, carrying its ``i`` and momentum tags.
        kin: :class:`~feynlag.pheno.kinematics.TwoBodyKinematics`.
        legs: ``(daughter1, daughter2)`` in the order matching ``kin.m1/m2``.
        vector_leg: the vector symbol (see :func:`vss_vector_leg`).
        parent: the decaying leg symbol.
    """
    a, b = index("a_vss"), index("b_vss")
    coupling = sp.sympify(coupling)

    # incoming-momentum head for each leg (parent in, daughters out)
    q = {parent: lambda i: kin.p1(i) + kin.p2(i),
         legs[0]: lambda i: -kin.p1(i),
         legs[1]: lambda i: -kin.p2(i)}
    mass = {legs[0]: kin.m1, legs[1]: kin.m2, parent: kin.M}

    def current(i, conjugate=False):
        """``Σ_legs (∂coupling/∂p(leg)) q_leg^i`` — the ``M^μ`` of the rule."""
        out = sp.S.Zero
        for leg, qhead in q.items():
            c = sp.diff(coupling, tag(leg))
            if c != 0:
                out += (sp.conjugate(c) if conjugate else c) * qhead(i)
        return out

    pol = polarization_sum(q[vector_leg], mass[vector_leg], -a, -b)
    expr = (current(a) * current(b, conjugate=True) * pol)
    expr = expr.contract_metric(LorentzIndex.metric)
    average = 3 if vector_leg == parent else 1     # average over V polarizations
    return sp.expand(contract_to_dots(expr, kin.dot) / average)


# ----------------------------------------------------------------- dispatch

def amplitude_squared(vertex, kin, children=None, parent=None):
    """``⟨|M|²⟩`` for a :class:`~feynlag.pheno.vertices.DecayVertex`.

    Args:
        children: ``(leg1, leg2)`` matching ``kin.m1``/``kin.m2``. Required
            for ``VSS``, whose two topologies (scalar parent vs vector parent)
            and whose vector mass cannot be read off ``kin`` alone.
        parent: the decaying leg symbol; required for ``VSS``.

    Raises:
        NotImplementedError: for a vertex type outside
            :data:`SUPPORTED_VERTEX_TYPES`.
    """
    vtype = vertex.vertex_type
    if vtype == "SSS":
        return sss_squared(vertex.coupling, kin)
    if vtype == "FFS":
        return ffs_squared(vertex.g_left, vertex.g_right, kin)
    if vtype == "FFV":
        return ffv_squared(vertex.g_left, vertex.g_right, kin,
                           vector_mass=kin.M)
    if vtype == "VVS":
        return vvs_squared(vertex.coupling, kin)
    if vtype == "VSS":
        if children is None or parent is None:
            raise ValueError(
                "VSS needs `children` and `parent` to fix which leg is the "
                "vector and which topology applies")
        return vss_squared(vertex.coupling, kin, tuple(children),
                           vss_vector_leg(vertex), parent)
    if vtype == "VVV":
        raise NotImplementedError(
            f"{vtype} decays carry momentum tags p(φ) from "
            f"to_momentum_space(); squaring them needs derivative-coupling "
            f"support that this stage does not implement")
    raise NotImplementedError(
        f"⟨|M|²⟩ for vertex type {vtype!r} is not implemented; supported "
        f"three-leg types are {SUPPORTED_VERTEX_TYPES}")
