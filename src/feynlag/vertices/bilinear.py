"""The fermion bilinear track.

Fermions never enter the commuting Poly extractor.  A Yukawa or gauge-current
term is written as::

    coefficient(bosons, params) × Bilinear(ψ̄_a[i], Γ, χ_b[j])

where ``Γ`` is a Dirac structure built from :mod:`feynlag.dirac` objects
(``diracPL``, ``diracPR``, ``DiracGamma(mu)*diracPL`` …) and ``[i]``, ``[j]``
are flavor indices on ``IndexedBase`` components.  The bilinear is an opaque
commuting atom (a spinor sandwich is a c-number); v1 forbids more than one
bilinear per term (no four-fermion operators — no ordering ambiguities).

:func:`extract_fermion_vertices` groups terms by bilinear and peels the boson
legs off each coefficient with the bosonic extractor — this systematizes the
DLRSM1 ``IndexedBase`` coupling pattern.
"""

import sympy as sp
from sympy import Function

from ..dirac import dirac_conjugate
from ..fields import bar_partner
from .extract import extract_interaction_coefficients, vertex_multiplicity

__all__ = ["Bilinear", "extract_fermion_vertices", "fermion_gauge_current",
           "fermion_mass_matrix", "fermion_feynman_rule"]


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


def extract_fermion_vertices(L, boson_fields):
    """Group a fermionic Lagrangian by bilinear and extract boson legs.

    Args:
        L: expanded Lagrangian sector; every term must contain exactly one
            :class:`Bilinear` factor (raises otherwise — no silent drops).
        boson_fields: boson symbols for the coefficient extraction.

    Returns:
        dict ``{(bar, gamma, field): {n_bosons: {boson-tuple: coeff}}}``.
    """
    L = sp.expand(L)
    grouped = {}
    terms = L.as_ordered_terms() if L.is_Add else ([L] if L != 0 else [])
    for term in terms:
        bils = list(term.atoms(Bilinear))
        if not bils:
            raise ValueError(f"term without a fermion bilinear in the "
                             f"fermionic sector: {term}")
        if len(bils) > 1:
            raise ValueError(f"more than one bilinear in a term (four-fermion "
                             f"operators are outside v1 scope): {term}")
        bil = bils[0]
        coeff = sp.cancel(term / bil)
        if coeff.has(Bilinear):
            raise ValueError(f"bilinear appears non-linearly in {term}")
        key = (bil.bar, bil.gamma, bil.field)
        grouped[key] = grouped.get(key, sp.S.Zero) + coeff

    return {key: extract_interaction_coefficients(coeff, boson_fields)
            for key, coeff in grouped.items()}


def fermion_feynman_rule(coefficient, gamma, boson_tuple):
    """FFS/FFV rule: ``i × coefficient × ∏(boson multiplicities)! × Γ``."""
    return sp.I * coefficient * vertex_multiplicity(boson_tuple) * gamma


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
    L0 = vacuum.at_vacuum(sp.expand(L_fermionic))
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
