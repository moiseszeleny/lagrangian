"""Theory constraints on the S₃-3HDM quartics: boundedness-from-below + unitarity.

Numpy-vectorized over a block of λ points, so a scan evaluates millions of
candidates without a Python loop.

**These conditions are quoted from the literature, not derived here.**  They
live in [DasDey14]'s λ₁…λ₈ parametrization; [GomezBock21] uses a different
(a,…,h) parametrization of the same potential and explicitly delegates its
bounds to [DasDey14] (their §2.2: "This analysis has already been done in
[66], and we use their expressions for the unitarity and stability bounds in
here").

feynlag's λ's are **not** trivially identical to [DasDey14]'s: the doublet
irrep bases differ, and the λ₄ invariant transcribed literally from
[DasDey14] into feynlag's component labels fails `check_discrete_invariance`
against S₃.  The dictionary is therefore *derived*, not assumed — by requiring
feynlag's independently-computed pseudoscalar and charged masses to reproduce
[GomezBock21]'s published closed forms Eqs. (30)–(33).  They match exactly, and
only for

    a = 2λ₈,  b = λ₅,  c = 2λ₁,  d = 2λ₂,  e = −λ₄,  f = λ₆,  g = 2λ₃,  h = 2λ₇

Composing with the [GomezBock21]↔[DasDey14] correspondence read off the two
potentials (λᴰᴰ₁=c/2, λᴰᴰ₂=d/2, λᴰᴰ₃=g/2, λᴰᴰ₄=e, λᴰᴰ₅=b, λᴰᴰ₆=f, λᴰᴰ₇=h/2,
λᴰᴰ₈=a/2) gives what matters here:

    λ_k^DasDey = λ_k^feynlag for every k ≠ 4,   λ₄^DasDey = −λ₄^feynlag

and **λ₄ enters every condition below only as |λ₄| (Eq. 4g) or λ₄² (Eqs. 37a,
37c, 37d, 37f)**.  So the conditions transfer to feynlag's λ's verbatim, with
no sign correction — as a derived result rather than a transcription gamble.
The derivation is reproduced in `01_scalar_parameter_space.ipynb` §3 and pinned
by `tests/test_thdm_s3.py::test_masses_match_gomezbock_closed_forms`.

References
----------
[DasDey14] D. Das, U. K. Dey, "Analysis of an extended scalar sector with S₃
    symmetry", Phys. Rev. D 89, 095025 (2014), arXiv:1404.2491,
    doi:10.1103/PhysRevD.89.095025.  Eq. (4a)–(4g) boundedness-from-below;
    Eq. (36) + Eq. (37a)–(37l) the tree-unitarity eigenvalues.
[GomezBock21] M. Gómez-Bock, M. Mondragón, A. Pérez-Martínez, Eur. Phys. J. C
    81, 942 (2021), arXiv:2102.02800.  §2.2 delegates these bounds to
    [DasDey14]; Eq. (8) is the v = 246 GeV constraint.
"""

from __future__ import annotations

import numpy as np

#: Tree-unitarity bound on each S-matrix eigenvalue, [DasDey14] Eq. (36).
UNITARITY_BOUND = 16.0 * np.pi


def _unpack(lam):
    """(N, 8) array → the eight columns λ₁…λ₈ (1-indexed in the physics)."""
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    if lam.shape[-1] != 8:
        raise ValueError(f"expected 8 quartics per point, got {lam.shape[-1]}")
    return (lam[:, 0], lam[:, 1], lam[:, 2], lam[:, 3],
            lam[:, 4], lam[:, 5], lam[:, 6], lam[:, 7])


# --------------------------------------------------------------------------
# boundedness from below — [DasDey14] Eq. (4a)-(4g)
# --------------------------------------------------------------------------

def bfb_conditions(lam):
    """(N, 7) boolean array, one column per condition Eq. (4a)…(4g).

    Note Eqs. (4e)/(4f) contain √(λ₈(λ₁+λ₃)), which is real only where (4b)
    and (4c) already hold; the product is clipped at 0 so the array stays
    finite, and those rows are rejected by (4b)/(4c) anyway.
    """
    l1, l2, l3, l4, l5, l6, l7, l8 = _unpack(lam)
    root = 2.0 * np.sqrt(np.clip(l8 * (l1 + l3), 0.0, None))
    return np.column_stack([
        l1 > 0,                                                    # (4a)
        l8 > 0,                                                    # (4b)
        l1 + l3 > 0,                                               # (4c)
        2 * l1 + (l3 - l2) > np.abs(l2 + l3),                      # (4d)
        l5 + root > 0,                                             # (4e)
        l5 + l6 + root > 2 * np.abs(l7),                           # (4f)
        l1 + l3 + l5 + l6 + 2 * l7 + l8 > 2 * np.abs(l4),          # (4g)
    ])


def bfb_mask(lam):
    """(N,) boolean: the potential is bounded from below at this λ point."""
    return bfb_conditions(lam).all(axis=1)


# --------------------------------------------------------------------------
# tree unitarity — [DasDey14] Eq. (37a)-(37l), bounded by Eq. (36)
# --------------------------------------------------------------------------

def _pm(trace_half, disc):
    """The `X ± √(X² − 4Y)` pair of Eq. (37a)-(37f), as two columns.

    Every discriminant in Eq. (37) reduces algebraically to a sum of squares
    — e.g. Eq. (37b)'s is (λ₁+λ₂+2λ₃−λ₈)² + 8λ₇² — so it is never negative
    and no branch guard is needed.  The clip only absorbs float round-off at
    the exact-degeneracy boundary.
    """
    root = np.sqrt(np.clip(disc, 0.0, None))
    return trace_half + root, trace_half - root


def unitarity_eigenvalues(lam):
    """(N, 18) array of the S-matrix eigenvalues a±₁…a±₆, b₁…b₆.

    [DasDey14] Eq. (37a)–(37l), transcribed verbatim.
    """
    l1, l2, l3, l4, l5, l6, l7, l8 = _unpack(lam)
    cols = []

    # (37a)
    x = l1 - l2 + (l5 + l6) / 2
    cols += _pm(x, x**2 - 4 * ((l1 - l2) * (l5 + l6) / 2 - l4**2))
    # (37b)
    x = l1 + l2 + 2 * l3 + l8
    cols += _pm(x, x**2 - 4 * (l8 * (l1 + l2 + 2 * l3) - 2 * l7**2))
    # (37c)
    x = l1 - l2 + 2 * l3 + l8
    cols += _pm(x, x**2 - 4 * (l8 * (l1 - l2 + 2 * l3) - l4**2 / 2))
    # (37d)
    x = l1 + l2 + l5 / 2 + l7
    cols += _pm(x, x**2 - 4 * ((l1 + l2) * (l5 / 2 + l7) - l4**2))
    # (37e)
    x = 5 * l1 - l2 + 2 * l3 + 3 * l8
    cols += _pm(x, x**2 - 4 * (3 * l8 * (5 * l1 - l2 + 2 * l3)
                               - (2 * l5 + l6)**2 / 2))
    # (37f)
    x = l1 + l2 + 4 * l3 + l5 / 2 + l6 + 3 * l7
    cols += _pm(x, x**2 - 4 * ((l1 + l2 + 4 * l3) * (l5 / 2 + l6 + 3 * l7)
                               - 9 * l4**2))

    # (37g)-(37l)
    cols += [l5 + 2 * l6 - 6 * l7,          # b1
             l5 - 2 * l7,                   # b2
             2 * (l1 - 5 * l2 - 2 * l3),    # b3
             2 * (l1 - l2 - 2 * l3),        # b4
             2 * (l1 + l2 - 2 * l3),        # b5
             l5 - l6]                       # b6
    return np.column_stack(cols)


def unitarity_mask(lam, bound=UNITARITY_BOUND):
    """(N,) boolean: every S-matrix eigenvalue satisfies |a| ≤ 16π."""
    return (np.abs(unitarity_eigenvalues(lam)) <= bound).all(axis=1)


# --------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------

def theory_mask(lam, bound=UNITARITY_BOUND):
    """(N,) boolean: bounded from below AND tree-unitary."""
    return bfb_mask(lam) & unitarity_mask(lam, bound=bound)


def constraint_report(lam, bound=UNITARITY_BOUND):
    """Per-condition survival fractions — which cut is actually doing the work."""
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    bfb = bfb_conditions(lam)
    labels = [f"BFB Eq.(4{c})" for c in "abcdefg"]
    out = {lab: float(bfb[:, i].mean()) for i, lab in enumerate(labels)}
    out["unitarity Eq.(36)"] = float(unitarity_mask(lam, bound).mean())
    out["BFB (all)"] = float(bfb.all(axis=1).mean())
    out["combined"] = float(theory_mask(lam, bound).mean())
    return out
