"""Amplitude-level diagram objects: open spinor chains, propagators, ``|M|²``.

Tier 1 of ``docs/manual/scattering_roadmap.md``.  The 1→2 decay engine in
:mod:`~feynlag.pheno.amplitudes` never constructs an amplitude — each
squared-amplitude function writes down an already-squared closed form, with
the 1→2 mass-interference terms and the initial-state spin average baked in
as literals for that one topology.  There is nothing there to extend to a
diagram with a *propagator* in the middle, or to a sum over more than one
diagram.  This module builds the amplitude explicitly instead, as a sum of
:class:`Diagram`\\ s, each a product of open fermion :class:`SpinorChain`\\ s
and :class:`BosonPropagator`\\ s — the covariant engine
(:mod:`~feynlag.pheno.lorentz`) is reused unchanged; what is new is
*combining* its output across a propagator and (eventually) across diagrams.

Two hard guards, both load-bearing rather than incidental:

- **Interference.**  :meth:`Amplitude.squared` raises for more than one
  diagram.  Multi-diagram interference (relative fermion-flow signs,
  identical-particle exchange signs) is Tier 3.
- **The ε (γ₅) term.**  Each chain's own trace never needs it —
  :func:`~feynlag.pheno.lorentz.reduce_projectors` still proves it vanishes
  for a chain's own two legs.  But when *two* chiral chains meet through a
  propagator, the product of their would-be ε coefficients is a genuine,
  non-zero contribution to ``|M|²`` (the forward–backward-asymmetry term) —
  a chain-level fact `reduce_projectors` cannot see, since it only ever
  looks at one chain in isolation.  :meth:`Amplitude.squared` computes that
  product directly via :meth:`ChainVertex.epsilon_coefficient` and refuses
  to drop it.  ε·ε → Gram-determinant algebra is Tier 2.
"""

from dataclasses import dataclass

import sympy as sp
from sympy.physics.hep.gamma_matrices import GammaMatrix, LorentzIndex

from .lorentz import contract_to_dots, dirac_trace, index, reduce_projectors, slashed
from .propagator import breit_wigner, vector_propagator_numerator

__all__ = [
    "Amplitude", "BosonPropagator", "ChainVertex", "Diagram", "Leg",
    "SpinorChain",
]


def _chain_indices(tag):
    """Deterministic external ``(Γ, Γ̄)`` Lorentz-index pair for a chain
    tagged ``tag``.

    :func:`~feynlag.pheno.lorentz.contract_to_dots` requires every index name
    to appear exactly twice; two chains meeting at one propagator need four
    distinct external names (this chain's Γ/Γ̄ and the other chain's Γ/Γ̄), so
    callers must tag each chain distinctly (e.g. ``"in"``/``"out"``).
    """
    return index(f"mu_{tag}"), index(f"mup_{tag}")


@dataclass(frozen=True)
class Leg:
    """One external fermion leg of an open spinor chain.

    Args:
        momentum: the momentum ``TensorHead`` (e.g. ``kin.k1`` from a
            :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`).
        mass: the leg's mass.
        anti: ``True`` for an antiparticle leg (``Σ v v̄ = p̸ − m``),
            ``False`` for a particle leg (``Σ u ū = p̸ + m``) — this is a
            property of the *particle*, not of whether the leg is incoming or
            outgoing; the incoming/outgoing distinction is instead encoded by
            which of a chain's two legs the caller assigns to ``out``
            (the ψ̄/bar slot) vs. ``inn`` (the ψ/field slot), following the
            standard external-fermion-line table (incoming particle / outgoing
            antiparticle → field slot; outgoing particle / incoming
            antiparticle → bar slot).
    """

    momentum: object
    mass: sp.Expr
    anti: bool = False

    def __post_init__(self):
        object.__setattr__(self, "mass", sp.sympify(self.mass))


@dataclass(frozen=True)
class ChainVertex:
    """One fermion-line vertex: ``Γ = g_L P_L + g_R P_R`` (structure ``'S'``,
    a scalar/pseudoscalar current) or ``Γ = γ^μ(g_L P_L + g_R P_R)``
    (structure ``'V'``, a vector current) — the same two structures
    :func:`~feynlag.pheno.vertices.classify_gamma` classifies a 1→2 vertex
    into.
    """

    structure: str
    g_left: sp.Expr
    g_right: sp.Expr

    def __post_init__(self):
        if self.structure not in ("S", "V"):
            raise NotImplementedError(
                f"ChainVertex: unsupported structure {self.structure!r}; "
                f"only 'S' and 'V' are implemented")
        object.__setattr__(self, "g_left", sp.sympify(self.g_left))
        object.__setattr__(self, "g_right", sp.sympify(self.g_right))

    def bar(self):
        """``Γ̄ = γ⁰Γ†γ⁰``, matching :func:`feynlag.dirac.dirac_conjugate`.

        A ``'V'`` current is self-conjugate structure-wise (``γ^μP_L =
        P_Rγ^μ``): the two effects cancel, so the chirality labels do *not*
        swap, only the couplings conjugate.  A ``'S'`` current has bare
        projectors, which *do* swap (``P̄_L = P_R``), so the returned
        ``g_left``/``g_right`` are conjugated **and exchanged**.
        """
        if self.structure == "V":
            return ChainVertex("V", sp.conjugate(self.g_left),
                               sp.conjugate(self.g_right))
        return ChainVertex("S", sp.conjugate(self.g_right),
                           sp.conjugate(self.g_left))

    def epsilon_coefficient(self):
        """Coefficient of the ``Tr[…γ₅]`` (ε) piece of this vertex's chain.

        Only four-γ terms survive a γ₅ trace.  A ``'S'`` chain has at most
        two explicit γ's (from the two spin sums) and is always zero.  A
        ``'V'`` chain has four, and the chirality-diagonal split
        ``Tr[X P_{L,R}] = ½Tr[X] ∓ ½Tr[Xγ₅]`` leaves
        ``(|g_R|² − |g_L|²)/2`` as the coefficient that
        :func:`~feynlag.pheno.lorentz.reduce_projectors` drops.
        """
        if self.structure == "S":
            return sp.S.Zero
        return (sp.Abs(self.g_right)**2 - sp.Abs(self.g_left)**2) / 2


@dataclass(frozen=True)
class SpinorChain:
    """One open fermion line: ``(bar leg) Γ (field leg)``, spin-summed.

    ``out`` is the ψ̄ (bar) slot, ``inn`` the ψ (field) slot — see
    :class:`Leg`.  For a ``'V'`` vertex, ``indices``/``bar_indices`` must each
    hold exactly one external ``TensorIndex`` (the Γ-side and Γ̄-side
    Lorentz slots respectively, e.g. from :func:`_chain_indices`); for a
    ``'S'`` vertex both must be empty.
    """

    out: Leg
    inn: Leg
    vertices: tuple
    indices: tuple = ()
    bar_indices: tuple = ()

    def epsilon_coefficient(self):
        """This chain's :meth:`ChainVertex.epsilon_coefficient`.

        Raises:
            NotImplementedError: more than one vertex on the line — a
                derivative-coupling / multi-propagator chain, Tier 4.
        """
        if len(self.vertices) != 1:
            raise NotImplementedError(
                "SpinorChain.epsilon_coefficient: more than one vertex on a "
                "line is a derivative-coupling chain, Tier 4 of the "
                "scattering roadmap")
        return self.vertices[0].epsilon_coefficient()

    def trace(self):
        """The γ₅-free trace for this chain, open on ``indices``/
        ``bar_indices`` for a ``'V'`` vertex (a plain scalar for ``'S'``).

        ``Tr[spin_sum(out)·Γ·spin_sum(inn)·Γ̄]``, following the same
        diagonal/chirality-mixing split as
        :func:`~feynlag.pheno.amplitudes.ffs_squared`/``ffv_squared`` — the
        mass-interference term is a hand-derived literal there and here for
        the identical reason: :mod:`~feynlag.pheno.lorentz` has no explicit
        γ₅/projector object to build ``Γ``/``Γ̄`` from directly, only the
        machinery to reduce a γ₅-free remainder.  The result is left with
        its external indices **open** — reduction to on-shell dot products
        happens once, at the :class:`Amplitude` level, after combining with
        the other chain and the propagator.

        Raises:
            NotImplementedError: more than one vertex (Tier 4), or a
                structure outside ``{'S', 'V'}``.
        """
        if len(self.vertices) != 1:
            raise NotImplementedError(
                "SpinorChain.trace: more than one vertex on a line is a "
                "derivative-coupling / multi-propagator chain, Tier 4 of "
                "the scattering roadmap")
        vertex = self.vertices[0]
        gL, gR = vertex.g_left, vertex.g_right
        m_out, m_inn = self.out.mass, self.inn.mass
        tag = f"{id(self.out.momentum)}_{id(self.inn.momentum)}"
        d_out, d_inn = index(f"so_{tag}"), index(f"si_{tag}")
        # The DIAGONAL piece (weighted |g_L|²+|g_R|²) uses the bare momentum
        # slash, never the mass — P_L P_R = 0 kills any mass insertion there.
        # The mass-dependent piece survives only in the chirality-MIXING term
        # below (weighted 2Re(g_L ḡ_R)), exactly as in
        # amplitudes.ffs_squared/ffv_squared.  Folding the mass into a single
        # spin_sum() here and weighting the whole thing by |g_L|²+|g_R|² would
        # double-count it for a non-chiral (g_L=g_R) coupling.
        out_slash = slashed(self.out.momentum, d_out)
        inn_slash = slashed(self.inn.momentum, d_inn)
        sign_out = -1 if self.out.anti else 1
        sign_inn = -1 if self.inn.anti else 1
        mixing_mass = 2 * sign_out * m_out * sign_inn * m_inn

        if vertex.structure == "S":
            chain = out_slash * inn_slash
            expr, factor = reduce_projectors(chain, "L", n_free_indices=0,
                                             n_momenta=2)
            total = sp.S.Zero
            if gL != 0 or gR != 0:
                total += (sp.Abs(gL)**2 + sp.Abs(gR)**2) * factor * dirac_trace(expr)
            if gL != 0 and gR != 0:
                total += 2 * sp.re(gL * sp.conjugate(gR)) * mixing_mass
            return sp.expand(total)

        # 'V'
        if len(self.indices) != 1 or len(self.bar_indices) != 1:
            raise ValueError(
                "SpinorChain.trace: a 'V' vertex needs exactly one external "
                "index in each of `indices`/`bar_indices`")
        mu, mu_bar = self.indices[0], self.bar_indices[0]
        total = sp.S.Zero
        if gL != 0 or gR != 0:
            chain = out_slash * GammaMatrix(mu) * inn_slash * GammaMatrix(mu_bar)
            expr, factor = reduce_projectors(chain, "L", n_free_indices=2,
                                             n_momenta=2)
            total += (sp.Abs(gL)**2 + sp.Abs(gR)**2) * factor * dirac_trace(expr)
        if gL != 0 and gR != 0:
            total += (2 * sp.re(gL * sp.conjugate(gR)) * mixing_mass
                     * LorentzIndex.metric(mu, mu_bar))
        return total


@dataclass(frozen=True)
class BosonPropagator:
    """An internal boson line joining two chain vertices.

    Args:
        spin: ``0`` or ``1``.
        q2: the invariant ``q²`` (e.g. ``kin.s``).
        mass, width: as in
            :func:`~feynlag.pheno.propagator.propagator_denominator`.
        momentum: callable ``q(index) -> TensExpr``; required for a massive
            spin-1 propagator (unused for spin 0 or a massless spin-1).
    """

    spin: int
    q2: sp.Expr
    mass: sp.Expr
    width: sp.Expr = sp.S.Zero
    momentum: object = None

    def __post_init__(self):
        if self.spin not in (0, 1):
            raise NotImplementedError(
                f"BosonPropagator: unsupported spin {self.spin}; only 0 and "
                f"1 are implemented")
        object.__setattr__(self, "q2", sp.sympify(self.q2))
        object.__setattr__(self, "mass", sp.sympify(self.mass))
        object.__setattr__(self, "width", sp.sympify(self.width))
        if self.spin == 1 and self.mass != 0 and self.momentum is None:
            raise ValueError(
                "BosonPropagator: a massive spin-1 propagator needs "
                "`momentum` (a callable q(index) -> TensExpr)")

    def denominator_squared_inverse(self):
        """``1/|q²−m²+imΓ|²``, reusing
        :func:`~feynlag.pheno.propagator.breit_wigner` verbatim."""
        return breit_wigner(self.q2, self.mass, self.width)

    def numerator(self, a, b):
        """The real propagator numerator, both indices returned lowered:
        ``1`` for spin 0; ``g_ab`` (Feynman gauge) for a massless spin 1;
        :func:`~feynlag.pheno.propagator.vector_propagator_numerator`
        (reused verbatim) for a massive spin 1."""
        if self.spin == 0:
            return sp.S.One
        if self.mass == 0:
            return LorentzIndex.metric(-a, -b)
        return vector_propagator_numerator(self.momentum, self.mass, a, b)


@dataclass(frozen=True)
class Diagram:
    """One Feynman diagram: two spinor chains joined by one propagator.

    ``coefficient`` is an overall complex prefactor not already carried by
    either chain's couplings (default ``1`` — Tier 1's assemblers fold
    everything into the chain couplings and never need it).
    """

    chains: tuple
    propagators: tuple
    coefficient: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "coefficient", sp.sympify(self.coefficient))


@dataclass(frozen=True)
class Amplitude:
    """A sum of :class:`Diagram`\\ s."""

    diagrams: tuple

    def squared(self, kin):
        """The spin/colour-**summed** ``|M|²`` — no averaging (see
        :func:`~feynlag.pheno.scattering.average_factor`).

        Raises:
            NotImplementedError: more than one diagram (interference, Tier
                3); a diagram that is not the two-chain, one-propagator
                s-channel topology; or a non-zero ε·ε contribution
                (Tier 2) — see the module docstring.
        """
        if len(self.diagrams) != 1:
            raise NotImplementedError(
                "Amplitude.squared: interference between distinct diagrams "
                "is Tier 3 of the scattering roadmap — refusing to "
                "silently drop the cross terms")
        diagram = self.diagrams[0]
        if len(diagram.chains) != 2:
            raise NotImplementedError(
                "Amplitude.squared: Tier 1 only assembles the two-chain, "
                "single-propagator s-channel topology")
        if len(diagram.propagators) != 1:
            raise NotImplementedError(
                "Amplitude.squared: multi-propagator (t-channel) diagrams "
                "are not yet assembled")

        eps = sp.simplify(sp.Mul(*[c.epsilon_coefficient() for c in diagram.chains]))
        if eps != 0:
            raise NotImplementedError(
                f"the ε-tensor (γ₅) contribution to |M|² is non-zero for "
                f"these chiral couplings (coefficient {eps}); ε·ε → "
                f"Gram-determinant algebra is Tier 2 of the scattering "
                f"roadmap — refusing to silently drop it")

        chain_a, chain_b = diagram.chains
        tensor_a = chain_a.trace()
        tensor_b = chain_b.trace()
        prop = diagram.propagators[0]

        if prop.spin == 0:
            combined = tensor_a * tensor_b
        else:
            mu, mu_p = chain_a.indices[0], chain_a.bar_indices[0]
            nu, nu_p = chain_b.indices[0], chain_b.bar_indices[0]
            n1 = prop.numerator(mu, nu)
            n2 = prop.numerator(mu_p, nu_p)
            combined = (tensor_a * n1 * tensor_b * n2)
            combined = combined.contract_metric(LorentzIndex.metric)

        result = contract_to_dots(combined, kin.dot)
        result *= prop.denominator_squared_inverse()
        result *= sp.Abs(diagram.coefficient)**2
        return sp.expand(result)
