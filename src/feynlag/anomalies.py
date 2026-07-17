"""Gauge-anomaly cancellation from the declared fermion content.

A chiral gauge theory is only consistent if its triangle anomalies cancel.
This module computes every anomaly coefficient symbolically from the fermion
representations already declared for a :class:`~feynlag.lagrangian.Model`, so
it works even when U(1) charges are kept symbolic (e.g. the ``X = aY + b(B−L)``
assignment of ``examples/sm_u1x.py``) — the coefficients come out as
polynomials in the charge symbols that must vanish, which can then be solved
for the anomaly-free charge assignments.

Conventions
-----------
Everything is reduced to **left-handed Weyl** fermions.  A field declared with
``chirality='R'`` is the conjugate of a left-handed field, so under the
reduction all of its U(1) charges flip sign and every non-abelian
representation goes to its conjugate.  Since the Dynkin index ``T(R)=T(R̄)`` is
conjugation-invariant, only the cubic ``[SU(N)]³`` anomaly (odd under
conjugation, ``A(R̄)=−A(R)``) and the charge signs actually feel the flip.

Coefficients returned (each must vanish for consistency):

- ``[U(1)_a U(1)_b U(1)_c]`` — every multiset of three abelian factors,
- ``grav^2-U(1)_a`` — the mixed gravitational–gauge anomaly,
- ``[G]^2-U(1)_a`` — non-abelian squared times each abelian factor,
- ``[G]^3`` — the pure cubic non-abelian anomaly (non-zero only for complex
  representations; among the supported reps that is the SU(3) fundamental),
- ``Witten-G`` — the SU(2) global anomaly: the number of doublets, which must
  be even (reported as an integer count, not a coefficient that vanishes).
"""

from itertools import combinations_with_replacement

import sympy as sp

from .fields import Fermion

__all__ = [
    "anomaly_coefficients", "AnomalyReport", "check_anomaly_free",
]


def _dynkin_index(group, rep):
    """Dynkin index ``T(R)`` from ``Tr(T^a T^b) = T(R) δ^{ab}``.

    Computed as ``T = (1/dim G) Σ_a Tr(T^a T^a)`` so it is robust for any
    representation whose generators the group can build (singlet → 0).
    """
    gens = group.generators(rep)
    n = len(gens)
    if n == 0:
        return sp.S.Zero
    total = sum(((T * T).trace() for T in gens), sp.S.Zero)
    return sp.simplify(total / n)


def _cubic_index(group, rep):
    """Cubic-anomaly index ``A(R)`` for a fundamental of SU(3).

    Real/pseudoreal representations (all SU(2) reps, every adjoint, singlets)
    have ``A=0``.  Among the representations this library supports the only
    complex one is the SU(3) fundamental, normalised to ``A(3)=1``; its
    conjugate ``3̄`` (a right-handed triplet reduced to left-handed) is ``-1``,
    handled by the chirality sign in :func:`anomaly_coefficients`.
    """
    if getattr(group, "n_generators", 0) == 8 and int(rep) == 3:
        return sp.S.One
    return sp.S.Zero


def _nonabelian_dim(field, groups):
    """Product of representation dimensions over ``groups`` for ``field``."""
    dim = sp.S.One
    for g in groups:
        if g in field.reps:
            dim *= g.rep_dim(field.reps[g])
    return dim


def anomaly_coefficients(model):
    """All gauge-anomaly coefficients of a model's fermion content.

    Args:
        model: a :class:`~feynlag.lagrangian.Model` (its ``gauge_groups`` and
            the :class:`~feynlag.fields.Fermion` members of ``fields`` are
            used).

    Returns:
        ``dict`` mapping an anomaly name to its (possibly symbolic)
        coefficient.  ``Witten-<group>`` entries hold the integer doublet
        count instead of a vanishing coefficient.
    """
    fermions = [f for f in model.fields if isinstance(f, Fermion)]
    abelian = [g for g in model.gauge_groups if g.abelian]
    nonabelian = [g for g in model.gauge_groups if not g.abelian]

    # Pre-reduce every fermion to left-handed data.
    reduced = []
    for f in fermions:
        sign = 1 if f.chirality != "R" else -1          # charge/cubic flip
        nf = getattr(f, "nflavors", 1)
        charges = {g: sign * sp.sympify(f.reps[g]) for g in abelian
                   if g in f.reps}
        full_dim = nf * _nonabelian_dim(f, nonabelian)
        reduced.append((f, sign, nf, charges, full_dim))

    coeffs = {}

    def q(entry, group):
        return entry[3].get(group, sp.S.Zero)

    # [U(1)_a U(1)_b U(1)_c] — cubic abelian (all multisets of three)
    for triple in combinations_with_replacement(abelian, 3):
        a, b, c = triple
        name = f"[{a.name}][{b.name}][{c.name}]"
        coeffs[name] = sp.expand(sum(
            e[4] * q(e, a) * q(e, b) * q(e, c) for e in reduced))

    # grav^2-U(1)_a
    for a in abelian:
        coeffs[f"grav^2-{a.name}"] = sp.expand(sum(
            e[4] * q(e, a) for e in reduced))

    # [G]^2-U(1)_a  and  [G]^3
    for G in nonabelian:
        spectators = [g for g in nonabelian if g is not G]
        for a in abelian:
            total = sp.S.Zero
            for f, sign, nf, charges, full_dim in reduced:
                if G not in f.reps:
                    continue
                spec_dim = nf * _nonabelian_dim(f, spectators)
                total += spec_dim * _dynkin_index(G, f.reps[G]) \
                    * charges.get(a, sp.S.Zero)
            coeffs[f"[{G.name}]^2-{a.name}"] = sp.expand(total)

        cubic = sp.S.Zero
        for f, sign, nf, charges, full_dim in reduced:
            if G not in f.reps:
                continue
            spec_dim = nf * _nonabelian_dim(f, spectators)
            cubic += spec_dim * sign * _cubic_index(G, f.reps[G])
        coeffs[f"[{G.name}]^3"] = sp.expand(cubic)

        # Witten global anomaly: doublet count for an SU(2)-type group
        if getattr(G, "n_generators", 0) == 3:
            doublets = sp.S.Zero
            for f, sign, nf, charges, full_dim in reduced:
                if f.reps.get(G) == 2:
                    doublets += nf * _nonabelian_dim(f, spectators)
            coeffs[f"Witten-{G.name}"] = sp.expand(doublets)

    return coeffs


class AnomalyReport:
    """Result of :func:`check_anomaly_free`.

    Attributes:
        coefficients: the full ``{name: coefficient}`` map.
        nonzero: the subset that does not vanish (Witten entries that are odd
            are reported here as ``"<count> (odd)"``).
    """

    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.nonzero = {}
        for name, value in coefficients.items():
            if name.startswith("Witten-"):
                if int(value) % 2 != 0:
                    self.nonzero[name] = f"{value} (odd)"
            elif sp.simplify(value) != 0:
                self.nonzero[name] = sp.simplify(value)

    @property
    def ok(self):
        return not self.nonzero

    def raise_on_failure(self):
        if not self.ok:
            lines = "\n".join(f"  {n} = {v}" for n, v in self.nonzero.items())
            raise ValueError(f"gauge anomalies do not cancel:\n{lines}")
        return self

    def __repr__(self):
        status = "anomaly-free" if self.ok else f"{len(self.nonzero)} nonzero"
        return f"AnomalyReport({len(self.coefficients)} coefficients, {status})"


def check_anomaly_free(model):
    """Convenience: build coefficients and wrap them in an :class:`AnomalyReport`."""
    return AnomalyReport(anomaly_coefficients(model))
