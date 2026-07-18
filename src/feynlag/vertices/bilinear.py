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

from ..dirac import dirac_conjugate, majorana_symmetry_sign
from ..fields import bar_partner
from .extract import extract_interaction_coefficients, vertex_multiplicity

__all__ = ["Bilinear", "MajoranaBilinear", "expand_bilinear",
           "extract_fermion_vertices", "extract_majorana_vertices",
           "fermion_gauge_current", "fermion_mass_matrix",
           "fermion_feynman_rule", "four_fermion_feynman_rule",
           "majorana_mass_matrix", "majorana_feynman_rule"]


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


_maj_sign_cache = {}


def _majorana_sign(gamma):
    """Cached ``ψ₁ᵀMψ₂ = s ψ₂ᵀMψ₁`` sign (see dirac.majorana_symmetry_sign)."""
    if gamma not in _maj_sign_cache:
        _maj_sign_cache[gamma] = majorana_symmetry_sign(gamma)
    return _maj_sign_cache[gamma]


class MajoranaBilinear(Function):
    """``MajoranaBilinear(ψ₁[i], C·Γ, ψ₂[j])`` — a Majorana sandwich ``ψ₁ᵀ C Γ ψ₂``.

    Unlike :class:`Bilinear` (a Dirac ``ψ̄ Γ ψ``), **both** legs are ordinary
    *field* components (no bar): it is the same-chirality, charge-conjugation-
    contracted bilinear of the dim-5 Weinberg operator ``ν_Lᵀ C ν_L`` and of
    Majorana masses ``½ M ν_Rᵀ C ν_R``.  The middle ``Γ`` **must** contain the
    charge-conjugation matrix :data:`~feynlag.dirac.diracC` (e.g.
    ``diracC*diracPL``).

    The Grassmann anticommutation symmetry ``ψ₁ᵀ(CΓ)ψ₂ = s·ψ₂ᵀ(CΓ)ψ₁`` is
    applied at construction: the two legs are put in canonical order with the
    sign ``s`` from :func:`~feynlag.dirac.majorana_symmetry_sign` (``+1`` for
    the ``C P_L`` mass structure ⟹ the mass matrix comes out symmetric; a
    self-contraction of an antisymmetric structure collapses to ``0``).
    """

    nargs = 3

    @classmethod
    def eval(cls, field1, gamma, field2):
        # Only canonicalise fully-resolved Indexed legs; during the invariance
        # transform the legs are transiently Add-valued (expand_bilinear then
        # re-distributes into Indexed-leg atoms, which canonicalise here).
        if not (isinstance(field1, sp.Indexed) and isinstance(field2, sp.Indexed)):
            return None
        k1 = sp.default_sort_key(field1)
        k2 = sp.default_sort_key(field2)
        if k1 == k2:
            if _majorana_sign(gamma) == -1:
                return sp.S.Zero        # ψᵀ(antisym)ψ = 0
            return None
        if k1 > k2:
            return _majorana_sign(gamma) * cls(field2, gamma, field1)
        return None

    @property
    def field1(self):
        return self.args[0]

    @property
    def gamma(self):
        return self.args[1]

    @property
    def field2(self):
        return self.args[2]

    def _latex(self, printer):
        return (rf"{printer.doprint(self.field1)}^{{T}}"
                rf"\,{printer.doprint(self.gamma)}\,"
                rf"{printer.doprint(self.field2)}")

    def _eval_conjugate(self):
        """``(ψ₁ᵀ C Γ ψ₂)† = ψ̄₁ C Γ̄ ψ̄₂ᵀ`` — the barred-field partner.

        The two field legs become their bar partners (``bar_partner`` registry)
        and the chiral projector flips (``C P_L → C P_R``): the ``C`` is
        invariant, so :func:`~feynlag.dirac.dirac_conjugate` on the
        ``C``-stripped remainder does the ``P_L↔P_R`` swap.  Flavor indices are
        carried over unchanged.  The result is again a (symmetric) Majorana
        bilinear, of bar fields — so ``O + h.c.`` is manifestly hermitian
        (pinned in ``tests/test_majorana.py``).
        """
        from ..dirac import diracC
        b1 = bar_partner(self.field1.base)[self.field1.indices]
        b2 = bar_partner(self.field2.base)[self.field2.indices]
        rest = sp.Mul(*[f for f in sp.Mul.make_args(self.gamma)
                        if not isinstance(f, type(diracC))])
        conj_gamma = diracC * dirac_conjugate(rest)
        return MajoranaBilinear(b1, conj_gamma, b2)


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
    """Distribute :class:`Bilinear` and :class:`MajoranaBilinear` over
    ``Add``-valued legs.

    Both atoms are linear in their two field slots (not in ``gamma``) but are
    opaque custom ``Function``\\ s, so ``sp.expand()`` alone won't distribute
    them — the same "teach SymPy about a custom operator's linearity" need as
    :func:`~feynlag.operators.D_linear` for ``PartialMu``, here applied to two
    slots instead of one.

    This is what makes fermion mass-basis rotations work: a
    :class:`~feynlag.vacuum.Rotation` with ``Indexed`` old fields leaves
    ``cosθ·ψ₁[i] + sinθ·ψ₂[i]`` sums trapped inside bilinear slots after
    ``xreplace``; without distributing them, extraction would group by the
    un-split composite key.  Both :func:`extract_fermion_vertices` and
    :func:`fermion_mass_matrix` / :func:`majorana_mass_matrix` apply it first.
    """
    def _make(atom):
        def _one(leg1, gamma, leg2):
            terms1 = sp.Add.make_args(sp.expand(leg1))
            terms2 = sp.Add.make_args(sp.expand(leg2))
            total = sp.S.Zero
            for t1 in terms1:
                c1, a1 = _split_indexed_term(t1)
                for t2 in terms2:
                    c2, a2 = _split_indexed_term(t2)
                    total += c1 * c2 * atom(a1, gamma, a2)
            return total
        return _one

    return expr.replace(Bilinear, _make(Bilinear)).replace(
        MajoranaBilinear, _make(MajoranaBilinear))


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


def extract_majorana_vertices(L, boson_fields):
    """Group a Majorana sector by :class:`MajoranaBilinear` and extract bosons.

    The Majorana analogue of :func:`extract_fermion_vertices`: every term must
    contain exactly one ``MajoranaBilinear`` factor (the Weinberg operator's
    ``ν̄νh`` / ``ν̄νhh`` couplings, or a bare Majorana mass with the ``h``-free
    ``()`` boson tuple).  ``Add``-valued legs are distributed first.

    Returns:
        dict ``{(field1, C·Γ, field2): {n_bosons: {boson-tuple: coeff}}}``.
    """
    L = sp.expand(expand_bilinear(sp.expand(L)))
    grouped = {}
    terms = L.as_ordered_terms() if L.is_Add else ([L] if L != 0 else [])
    for term in terms:
        mbs = list(term.atoms(MajoranaBilinear))
        if not mbs:
            raise ValueError(f"term without a Majorana bilinear in the "
                             f"Majorana sector: {term}")
        if len(mbs) != 1:
            raise NotImplementedError(
                f"term with {len(mbs)} Majorana bilinears is outside scope "
                f"(only single-MajoranaBilinear terms supported): {term}")
        mb = mbs[0]
        coeff = sp.cancel(term / mb)
        if coeff.has(MajoranaBilinear):
            raise ValueError(f"Majorana bilinear appears non-linearly in {term}")
        key = (mb.field1, mb.gamma, mb.field2)
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


def majorana_mass_matrix(L_fermionic, field_base, vacuum, nflavors, indices,
                         gamma=None):
    """Symmetric Majorana mass matrix from the vacuum-evaluated sector.

    Collects the coefficient of ``MajoranaBilinear(field_base[i], CΓ,
    field_base[j])`` at the vacuum for a Lagrangian mass term written in the
    convention ``L ⊃ −½ M_ij ψ_iᵀ C ψ_j`` (so ``M[i,j] = −2·coefficient``).
    Because :class:`MajoranaBilinear` canonicalises its legs (the ``i↔j``
    Grassmann symmetry), each unordered flavour pair contributes exactly one
    atom, and the returned matrix is **symmetric** by construction — ready for
    :func:`~feynlag.vacuum.diagonalize.diagonalize_takagi`.

    Args:
        L_fermionic: the (Weinberg / bare-Majorana) sector, weak basis.
        field_base: the ``IndexedBase`` of the Majorana field (both legs).
        vacuum: :class:`~feynlag.vacuum.Vacuum`.
        nflavors: matrix dimension.
        indices: the ``(i, j)`` index symbols used in the Lagrangian.
        gamma: restrict to one middle structure (default: sum all).

    Returns:
        ``nflavors × nflavors`` symmetric Matrix.
    """
    L0 = sp.expand(expand_bilinear(vacuum.at_vacuum(sp.expand(L_fermionic))))
    i, j = indices
    coeff = sp.S.Zero
    terms = L0.as_ordered_terms() if L0.is_Add else ([L0] if L0 != 0 else [])
    for term in terms:
        mbs = list(term.atoms(MajoranaBilinear))
        if len(mbs) != 1:
            continue
        mb = mbs[0]
        if not (getattr(mb.field1, "base", None) == field_base
                and getattr(mb.field2, "base", None) == field_base):
            continue
        if gamma is not None and mb.gamma != gamma:
            continue
        term_idx = (mb.field1.indices[0], mb.field2.indices[0])
        c = sp.cancel(term / mb)
        coeff += c.subs({term_idx[0]: i, term_idx[1]: j}, simultaneous=True)

    def entry(a, b):
        cab = coeff.subs({i: a, j: b}, simultaneous=True)
        cba = coeff.subs({i: b, j: a}, simultaneous=True)
        # off-diagonal (a≠b): the two orderings collapse to one canonical atom,
        # so only one of cab/cba carries it — sum picks it up once; diagonal
        # (a=b): cab==cba, and the −½ convention gives M = −2·coeff.
        return -(cab + cba) if a != b else -2 * cab

    return sp.Matrix(nflavors, nflavors, entry)


def majorana_feynman_rule(coefficient, gamma, boson_tuple=()):
    """Majorana FFS/FFSS rule: ``i × coefficient × ∏(boson mult)! × (CΓ)``.

    Mirrors :func:`fermion_feynman_rule` but the Dirac structure carries the
    charge-conjugation ``C`` (``gamma = diracC·P_L`` …).
    """
    return sp.I * coefficient * vertex_multiplicity(boson_tuple) * gamma
