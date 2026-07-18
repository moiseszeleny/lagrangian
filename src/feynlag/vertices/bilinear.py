"""The fermion bilinear track.

Fermions never enter the commuting Poly extractor.  A Yukawa or gauge-current
term is written as::

    coefficient(bosons, params) × Bilinear(ψ̄_a[i], Γ, χ_b[j])

where ``Γ`` is a Dirac structure built from :mod:`feynlag.dirac` objects
(``diracPL``, ``diracPR``, ``DiracGamma(mu)*diracPL`` …) and ``[i]``, ``[j]``
are flavor indices on ``IndexedBase`` components.  The bilinear is an opaque
commuting atom (a spinor sandwich is a c-number).

A **single** bilinear per term is the FFS/FFV (Yukawa / gauge-current) track.
**Two** bilinears per term is the four-fermion (FFFF) track for dim-6 effective
operators like Fermi theory ``(ψ̄Γψ)(χ̄Γ′χ)`` — supported in the **as-written
bilinear basis** (no Fierz canonicalisation; a Fierz-rearranged operator is a
different input) and restricted to **four distinct fermion components**.  When
a fermion component repeats among the four legs (e.g. ``(ēΓe)(ēΓ′e)`` or a
squared bilinear ``B²``), extraction raises :class:`NotImplementedError`: the
repeated leg generates cross-chain Wick contractions whose relative signs need
genuine spinor-index Fierz algebra the opaque-``Bilinear`` design cannot
express.  With distinct legs there are no exchange contractions, so the vertex
is simply ``i × coefficient × ∏(boson multiplicity)!`` with the two Dirac
structures ``(Γ, Γ′)`` carried alongside (:func:`four_fermion_feynman_rule`).

:func:`extract_fermion_vertices` groups terms by bilinear and peels the boson
legs off each coefficient with the bosonic extractor — this systematizes the
DLRSM1 ``IndexedBase`` coupling pattern.
"""

import sympy as sp
from sympy import Function

from ..dirac import dirac_conjugate
from ..fields import bar_partner
from .extract import extract_interaction_coefficients, vertex_multiplicity

__all__ = ["Bilinear", "expand_bilinear", "extract_fermion_vertices",
           "fermion_gauge_current", "fermion_mass_matrix",
           "fermion_feynman_rule", "four_fermion_feynman_rule"]


class Bilinear(Function):
    """``Bilinear(ψ̄_a[i], Γ, χ_b[j])`` — an opaque fermion sandwich.

    Args:
        bar: ``Indexed`` (an ``IndexedBase`` element) for ψ̄ with its flavor
            index, e.g. ``nuLbar[i]``.
        gamma: Dirac structure (``diracPL``, ``diracPR``,
            ``DiracGamma(mu)*diracPL``, …).
        field: ``Indexed`` for χ with its flavor index.
    """

    nargs = 3

    @classmethod
    def eval(cls, bar, gamma, field):
        return None  # stay unevaluated

    @property
    def bar(self):
        return self.args[0]

    @property
    def gamma(self):
        return self.args[1]

    @property
    def field(self):
        return self.args[2]

    def _latex(self, printer):
        return (rf"\bar{{{printer.doprint(self.bar)}}}"
                rf"\,{printer.doprint(self.gamma)}\,"
                rf"{printer.doprint(self.field)}")

    def _eval_conjugate(self):
        """``(ψ̄₁[i] Γ ψ₂[j])^† = ψ̄₂[j] Γ̄ ψ₁[i]``.

        Flavor indices are carried over unchanged (each stays with its own
        field's partner); only the bar/field roles and the Dirac structure
        swap.  ``bar_partner`` (fields.py) resolves "the bar IndexedBase of
        this field" / "the field IndexedBase of this bar" via the registry
        every :class:`~feynlag.fields.Fermion` populates at construction.
        """
        new_bar_base = bar_partner(self.field.base)
        new_field_base = bar_partner(self.bar.base)
        new_bar = new_bar_base[self.field.indices]
        new_field = new_field_base[self.bar.indices]
        return Bilinear(new_bar, dirac_conjugate(self.gamma), new_field)


def _split_indexed_term(term):
    """Split ``coeff * Indexed(...)`` into ``(coeff, Indexed(...))``.

    Unlike :meth:`~sympy.core.expr.Expr.as_coeff_Mul` (which, with its
    default ``rational=True``, only pulls out a *Rational* coefficient and
    leaves any Dummy/Symbol factors — such as the infinitesimal ``alpha``
    from :func:`~feynlag.invariance._fermion_transform` — bundled into the
    "atom" side), this finds the single ``Indexed`` factor explicitly and
    treats everything else (numbers, ``I``, ``alpha``, generator matrix
    entries, …) as the coefficient.
    """
    if isinstance(term, sp.Indexed):
        return sp.S.One, term
    factors = term.as_ordered_factors()
    indexed = [f for f in factors if isinstance(f, sp.Indexed)]
    if len(indexed) != 1:
        raise ValueError(f"expected exactly one Indexed factor in {term!r}, "
                         f"found {len(indexed)}")
    atom = indexed[0]
    coeff = sp.Mul(*[f for f in factors if f is not atom])
    return coeff, atom


def expand_bilinear(expr):
    """Distribute ``Bilinear(bar, gamma, field)`` over ``Add``-valued
    ``bar``/``field`` legs.

    ``Bilinear`` is linear in each of those two slots (not in ``gamma``) but
    is an opaque custom ``Function``, so ``sp.expand()`` alone won't
    distribute it — the same "teach SymPy about a custom operator's
    linearity" need as :func:`~feynlag.operators.D_linear` for
    ``PartialMu``, here applied to two slots instead of one.

    This is what makes fermion mass-basis rotations work: a
    :class:`~feynlag.vacuum.Rotation` with ``Indexed`` old fields leaves
    ``cosθ·ψ₁[i] + sinθ·ψ₂[i]`` sums trapped inside ``Bilinear`` slots after
    ``xreplace``; without distributing them, extraction would group by the
    un-split composite key.  Both :func:`extract_fermion_vertices` and
    :func:`fermion_mass_matrix` apply it before grouping.
    """
    def _one(bar, gamma, field):
        bar_terms = sp.Add.make_args(sp.expand(bar))
        field_terms = sp.Add.make_args(sp.expand(field))
        total = sp.S.Zero
        for bt in bar_terms:
            bc, ba = _split_indexed_term(bt)
            for ft in field_terms:
                fc, fa = _split_indexed_term(ft)
                total += bc * fc * Bilinear(ba, gamma, fa)
        return total

    return expr.replace(Bilinear, _one)


def _bilinear_factors(term):
    """Multiset of :class:`Bilinear` factors in ``term`` (powers expanded).

    ``term.atoms(Bilinear)`` deduplicates, so it cannot tell ``B`` from ``B²``;
    this counts multiplicity by walking the factors, so a squared bilinear is
    correctly seen as *two* legs (which the distinct-legs check then rejects).
    """
    out = []
    for f in sp.Mul.make_args(term):
        if isinstance(f, Bilinear):
            out.append(f)
        elif f.is_Pow and isinstance(f.base, Bilinear) \
                and f.exp.is_Integer and f.exp > 0:
            out.extend([f.base] * int(f.exp))
        elif f.has(Bilinear):
            # a Bilinear buried inside a non-factor structure (e.g. an Add that
            # survived expansion) — shouldn't happen after expand_bilinear
            raise ValueError(f"bilinear appears non-multiplicatively in {term}")
    return out


def _four_fermion_key(b1, b2):
    """Canonically ordered two-bilinear key, distinct-legs enforced.

    Returns ``((bar,Γ,field), (bar',Γ′,field'))`` sorted so ``B₁B₂`` and
    ``B₂B₁`` map to the same key.  Raises :class:`NotImplementedError` if any
    fermion component (``IndexedBase``) repeats among the four legs — see the
    module docstring for why (Fierz).
    """
    bases = [b1.bar.base, b1.field.base, b2.bar.base, b2.field.base]
    if len(set(bases)) != 4:
        raise NotImplementedError(
            "four-fermion operator with a repeated fermion component "
            f"({[str(b) for b in bases]}): identical legs generate cross-chain "
            "Wick contractions whose relative signs need spinor-index Fierz "
            "algebra, which the opaque-Bilinear design does not implement. "
            "Only operators with four distinct fermion components are supported.")
    subkeys = [(b1.bar, b1.gamma, b1.field), (b2.bar, b2.gamma, b2.field)]
    subkeys.sort(key=lambda t: sp.default_sort_key(sp.Tuple(*t)))
    return tuple(subkeys)


def extract_fermion_vertices(L, boson_fields):
    """Group a fermionic Lagrangian by bilinear(s) and extract boson legs.

    Args:
        L: expanded Lagrangian sector; every term must contain **one** (FFS/FFV)
            or **two** (FFFF, four-fermion) :class:`Bilinear` factors — raises
            otherwise (no silent drops).  ``Add``-valued bilinear legs (e.g.
            from a mass-basis rotation) are distributed first via
            :func:`expand_bilinear`.
        boson_fields: boson symbols for the coefficient extraction.

    Returns:
        dict ``{key: {n_bosons: {boson-tuple: coeff}}}`` where ``key`` is a flat
        ``(bar, gamma, field)`` for a one-bilinear (FFS/FFV) term and a nested
        ``((bar,Γ,field), (bar',Γ′,field'))`` pair for a two-bilinear (FFFF)
        term.  Consumers distinguish the two by ``isinstance(key[0], tuple)``.
    """
    L = sp.expand(expand_bilinear(sp.expand(L)))
    grouped = {}
    terms = L.as_ordered_terms() if L.is_Add else ([L] if L != 0 else [])
    for term in terms:
        bils = _bilinear_factors(term)
        if not bils:
            raise ValueError(f"term without a fermion bilinear in the "
                             f"fermionic sector: {term}")
        if len(bils) == 1:
            bil = bils[0]
            key = (bil.bar, bil.gamma, bil.field)
            coeff = sp.cancel(term / bil)
        elif len(bils) == 2:
            b1, b2 = bils
            key = _four_fermion_key(b1, b2)
            coeff = sp.cancel(term / (b1 * b2))
        else:
            raise NotImplementedError(
                f"term with {len(bils)} fermion bilinears is outside the "
                f"supported FFS/FFV/FFFF catalog: {term}")
        if coeff.has(Bilinear):
            raise ValueError(f"bilinear appears non-linearly in {term}")
        grouped[key] = grouped.get(key, sp.S.Zero) + coeff

    return {key: extract_interaction_coefficients(coeff, boson_fields)
            for key, coeff in grouped.items()}


def fermion_feynman_rule(coefficient, gamma, boson_tuple):
    """FFS/FFV rule: ``i × coefficient × ∏(boson multiplicities)! × Γ``."""
    return sp.I * coefficient * vertex_multiplicity(boson_tuple) * gamma


def four_fermion_feynman_rule(coefficient, gammas=None, boson_tuple=()):
    """FFFF rule: the **scalar** ``i × coefficient × ∏(boson multiplicities)!``.

    Unlike :func:`fermion_feynman_rule`, the two Dirac structures ``gammas =
    (Γ, Γ′)`` are **not** folded into the returned value: they act on two
    independent spinor chains, so multiplying them would wrongly invoke the
    Clifford algebra *across* chains (e.g. contract a shared ``γ^μ``).  They are
    carried alongside the coupling (on the extraction key / a :class:`Vertex`'s
    ``meta``), exactly as UFO keeps the Lorentz structure separate from the
    coupling value.  ``gammas`` is accepted for API symmetry and validated only.
    """
    if gammas is not None and len(tuple(gammas)) != 2:
        raise ValueError("a four-fermion vertex has exactly two Dirac chains")
    return sp.I * coefficient * vertex_multiplicity(boson_tuple)


def fermion_gauge_current(fermion, flavor_index, gauge_groups=None,
                          projector=None):
    """Gauge-current terms of ``i ψ̄ γ^μ D_μ ψ`` (the interaction part).

    With ``D_μ = ∂_μ − i g T^a A^a_μ`` (CONVENTIONS.md), the interaction is
    ``+ g A^a_μ ψ̄_a γ^μ T^a_{ab} P ψ_b``.

    Args:
        fermion: a :class:`~feynlag.fields.Fermion`.
        flavor_index: flavor index symbol (the current is flavor diagonal).
        gauge_groups: groups to include (default: all in ``fermion.reps``).
        projector: chiral projector; default from ``fermion.chirality``.

    Returns:
        SymPy expression: sum of ``coupling × A × Bilinear(ψ̄_a, γ^μ P, ψ_b)``.
    """
    from ..dirac import DiracGamma, diracI, diracPL, diracPR

    if projector is None:
        projector = {"L": diracPL, "R": diracPR, None: diracI}[
            fermion.chirality]
    mu = sp.Symbol("mu", integer=True)
    gamma = DiracGamma(mu) * projector

    groups = list(gauge_groups) if gauge_groups is not None \
        else list(fermion.reps.keys())
    i = flavor_index
    total = sp.S.Zero
    for group in groups:
        if group not in fermion.reps:
            continue
        boson = group.bosons()
        for a, T in enumerate(fermion.generators(group)):
            for r in range(fermion.dim):
                for c in range(fermion.dim):
                    if T[r, c] == 0:
                        continue
                    total += (group.g * T[r, c] * boson.components[a]
                              * Bilinear(fermion.bar_components[r][i], gamma,
                                         fermion.components[c][i]))
    return total


def fermion_mass_matrix(L_fermionic, bar_base, field_base, vacuum, nflavors,
                        indices, gamma=None):
    """Fermion mass matrix from the vacuum-evaluated fermionic sector.

    Collects the coefficient of ``Bilinear(bar_base[i], Γ, field_base[j])``
    at the vacuum; the Lagrangian mass term is ``−ψ̄ M χ``, so
    ``M[i,j] = −coefficient``.

    Args:
        L_fermionic: the fermionic Lagrangian sector (weak basis).
        bar_base, field_base: the ``IndexedBase`` pair of the mass term.
        vacuum: :class:`~feynlag.vacuum.Vacuum`.
        nflavors: matrix dimension.
        indices: the ``(i, j)`` index symbols used in the Lagrangian.
        gamma: restrict to one Dirac structure (default: sum all — correct
            when P_L and P_R terms are conjugate halves of a Dirac mass).

    Returns:
        ``nflavors × nflavors`` Matrix.
    """
    L0 = sp.expand(expand_bilinear(vacuum.at_vacuum(sp.expand(L_fermionic))))
    i, j = indices
    coeff = sp.S.Zero
    terms = L0.as_ordered_terms() if L0.is_Add else ([L0] if L0 != 0 else [])
    for term in terms:
        bils = list(term.atoms(Bilinear))
        if len(bils) != 1:
            continue
        bil = bils[0]
        if not (getattr(bil.bar, "base", None) == bar_base
                and getattr(bil.field, "base", None) == field_base):
            continue
        if gamma is not None and bil.gamma != gamma:
            continue
        # normalize this term's indices to (i, j)
        term_idx = (bil.bar.indices[0], bil.field.indices[0])
        c = sp.cancel(term / bil)
        c = c.subs({term_idx[0]: i, term_idx[1]: j}, simultaneous=True)
        coeff += c

    return sp.Matrix(nflavors, nflavors,
                     lambda a, b: -coeff.subs({i: a, j: b},
                                              simultaneous=True))
