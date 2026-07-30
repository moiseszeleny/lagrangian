"""Theory constraints on the S₃-3HDM quartics: boundedness-from-below + unitarity.

Numpy-vectorized over a block of λ points, so a scan evaluates millions of
candidates without a Python loop.

Two provenance warnings, both load-bearing — read them before trusting a scan
built on this module.

**(1) The [DasDey14] conditions are quoted, and their source is pre-erratum.**
They live in [DasDey14]'s λ₁…λ₈ parametrization; [GomezBock21] uses a different
(a,…,h) parametrization of the same potential and explicitly delegates its
bounds to [DasDey14] (their §2.2: "This analysis has already been done in
[66], and we use their expressions for the unitarity and stability bounds in
here").  [DasDey14] carries an **Erratum, Phys. Rev. D 91, 039905 (2015)**,
which was never posted to arXiv and is paywalled at APS; everything transcribed
below comes from the **pre-erratum arXiv v2**.

**(2) The boundedness conditions below are NOT sufficient.**  Tested directly
against the potential (`model.quartic_potential`), Eq. (4) accepts λ points that
have an explicit real neutral direction with V₄ < 0 — ≈10% of them.  Eq. (4g)
turns out to be a single point of a curve that must stay non-negative
everywhere; see `neutral_real_bfb_min`, which derives and solves the exact
condition on that slice, and `strict_bfb_mask`, which combines the two.
`01_scalar_parameter_space.ipynb` §5.3–§5.5 reproduces the whole argument with
an explicit counterexample.

The parameter dictionary
------------------------
feynlag's λ's are **not** trivially identical to [DasDey14]'s: the doublet
irrep bases differ, and the λ₄ invariant transcribed literally from
[DasDey14] into feynlag's component labels fails `check_discrete_invariance`
against S₃.  The dictionary is established in two steps, neither assumed:

* Under a general SO(2) rotation of the S₃ doublet, (s₁,s₂) rotates by α while
  d₂ rotates by 2α.  Checking all eight quartic structures symbolically, **seven
  are invariant** and only s·d₂ — the λ₄ structure — can change.  Invariant
  structures are the same object in either basis, so term-matching fixes
  a = 2λ₈, b = λ₅, c = 2λ₁, d = 2λ₂, f = λ₆, g = 2λ₃, h = 2λ₇.
* The remaining sign follows from physics: requiring feynlag's independently
  computed pseudoscalar and charged masses to reproduce [GomezBock21]'s closed
  forms Eqs. (30)–(33).  All four match exactly, and only for **e = −λ₄**.

Composing with the [GomezBock21]↔[DasDey14] correspondence read off the two
potentials (λᴰᴰ₁=c/2, λᴰᴰ₂=d/2, λᴰᴰ₃=g/2, λᴰᴰ₄=e, λᴰᴰ₅=b, λᴰᴰ₆=f, λᴰᴰ₇=h/2,
λᴰᴰ₈=a/2) gives what matters here:

    λ_k^DasDey = λ_k^feynlag for every k ≠ 4,   λ₄^DasDey = −λ₄^feynlag

and **λ₄ enters every condition below only as |λ₄| (Eq. 4g) or λ₄² (Eqs. 37a,
37c, 37d, 37f)**.  That is not luck: the only S₃-preserving transformation that
changes the λ₄ structure is the field redefinition H₁,H₂ → −H₁,−H₂, so no
physical condition *may* depend on its sign.  The conditions therefore transfer
to feynlag's λ's verbatim.  Reproduced in `01_scalar_parameter_space.ipynb` §4
and pinned by
`tests/test_thdm_s3.py::test_masses_match_gomezbock_closed_forms`.

References
----------
[DasDey14] D. Das, U. K. Dey, "Analysis of an extended scalar sector with S₃
    symmetry", Phys. Rev. D 89, 095025 (2014), arXiv:1404.2491 (v2, May 2014),
    doi:10.1103/PhysRevD.89.095025.  Eq. (4a)–(4g) boundedness-from-below;
    Eq. (36) + Eq. (37a)–(37l) the tree-unitarity eigenvalues.
    **Erratum: Phys. Rev. D 91, 039905 (2015)** — not on arXiv, not applied here.
[GomezBock21] M. Gómez-Bock, M. Mondragón, A. Pérez-Martínez, Eur. Phys. J. C
    81, 942 (2021), arXiv:2102.02800.  §2.2 delegates these bounds to
    [DasDey14]; Eq. (8) is the v = 246 GeV constraint.
[BentoRomaoSilva22] M. P. Bento, J. C. Romão, J. P. Silva, JHEP 08 (2022) 273,
    arXiv:2204.13130.  Independent post-erratum recomputation of 3HDM unitarity
    bounds — the cross-check target for Eq. (37).
"""

from __future__ import annotations

import numpy as np

#: Tree-unitarity bound on each S-matrix eigenvalue, [DasDey14] Eq. (36).
UNITARITY_BOUND = 16.0 * np.pi


#: Rows processed per block by `_chunked`.  A multi-million-point scan otherwise
#: allocates several hundred MB of temporaries at once (18 eigenvalue columns, or
#: an (N, 3, 3) stack of companion matrices) and can exhaust memory.
CHUNK = 250_000


def _unpack(lam):
    """(N, 8) array → the eight columns λ₁…λ₈ (1-indexed in the physics)."""
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    if lam.shape[-1] != 8:
        raise ValueError(f"expected 8 quartics per point, got {lam.shape[-1]}")
    return (lam[:, 0], lam[:, 1], lam[:, 2], lam[:, 3],
            lam[:, 4], lam[:, 5], lam[:, 6], lam[:, 7])


def _chunked(fn, lam, chunk=CHUNK):
    """Apply a row-wise array function in blocks, keeping peak memory bounded."""
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    if len(lam) <= chunk:
        return fn(lam)
    return np.concatenate([fn(lam[i:i + chunk]) for i in range(0, len(lam), chunk)])


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
    """(N,) boolean: the λ point satisfies the quoted [DasDey14] Eq. (4) conditions.

    **These are not sufficient** — see `neutral_real_bfb_min` below, which finds
    explicit counterexamples.  Use `strict_bfb_mask` for the corrected condition.
    """
    return _chunked(lambda block: bfb_conditions(block).all(axis=1), lam)


# --------------------------------------------------------------------------
# the corrected neutral-direction condition — DERIVED HERE, not quoted
# --------------------------------------------------------------------------

def neutral_real_bfb_min(lam):
    """min over t ≥ 0 of f(t) = λ₈t⁴ + (λ₅+λ₆+2λ₇)t² − 2|λ₄|t + (λ₁+λ₃), chunked.

    Negative ⟹ the quartic potential runs to −∞ along a real neutral direction,
    i.e. the point is **not** bounded from below whatever Eq. (4) says.

    Derivation (reproduced and verified symbolically in
    `01_scalar_parameter_space.ipynb` §5).  Restrict the potential to real
    neutral components (H₁⁰,H₂⁰,H_S⁰) = (r₁,r₂,r_S).  Every quartic invariant
    collapses onto three structures,

        V₄ = (λ₁+λ₃)x² + (λ₅+λ₆+2λ₇)x y + λ₈y² + 2λ₄ r_S r₁(r₁²−3r₂²),
        x ≡ r₁²+r₂²,  y ≡ r_S² ,

    and writing r₁=√x cosψ, r₂=√x sinψ the last term is exactly
    2λ₄√y·x^{3/2}cos3ψ — so its worst case over ψ and over the sign of r_S is
    −2|λ₄|x^{3/2}√y.  Setting x=1, t=√y (everything is degree 2 in (x,y)) gives
    f(t) above, which must be ≥ 0 for **every** t ≥ 0.

    [DasDey14] Eq. (4g), λ₁+λ₃+λ₅+λ₆+2λ₇+λ₈ > 2|λ₄|, is precisely ``f(1) > 0``
    — one point on that curve.  It is therefore necessary but not sufficient,
    and ≈10% of the λ points Eq. (4) accepts have f(t) < 0 somewhere.

    Still only a *necessary* condition overall: it covers the real neutral slice,
    not complex-neutral or charged directions.  `model.numeric_bfb_min` samples
    the full 12-dimensional field space as the empirical backstop.
    """
    return _chunked(_neutral_real_bfb_min_block, lam)


def _neutral_real_bfb_min_block(lam):
    A = lam[:, 0] + lam[:, 2]                      # λ₁+λ₃
    B = lam[:, 4] + lam[:, 5] + 2 * lam[:, 6]      # λ₅+λ₆+2λ₇
    C = 2 * np.abs(lam[:, 3])                      # 2|λ₄|
    l8 = lam[:, 7]

    out = np.full(len(lam), -np.inf)
    good = l8 > 0                                  # λ₈ ≤ 0 already fails Eq. (4b)
    if not good.any():
        return out
    A, B, C, l8 = A[good], B[good], C[good], l8[good]

    # stationary points: f'(t)/4λ₈ = t³ + (B/2λ₈)t − C/4λ₈ = 0, via companion matrices
    p, q = B / (2 * l8), -C / (4 * l8)
    comp = np.zeros((len(l8), 3, 3))
    comp[:, 0, 2], comp[:, 1, 2] = -q, -p
    comp[:, 1, 0] = comp[:, 2, 1] = 1.0
    roots = np.linalg.eigvals(comp)
    t = np.where((np.abs(roots.imag) < 1e-9) & (roots.real > 0), roots.real, 0.0)

    f = (l8[:, None] * t**4 + B[:, None] * t**2 - C[:, None] * t + A[:, None])
    out[good] = np.minimum(f.min(axis=1), A)       # t = 0 gives f(0) = A
    return out


def neutral_real_bfb_mask(lam, tol=0.0):
    """(N,) boolean: no real neutral direction sends the quartic potential to −∞."""
    return neutral_real_bfb_min(lam) >= tol


def strict_bfb_mask(lam):
    """(N,) boolean: [DasDey14] Eq. (4) **and** the corrected neutral condition."""
    return bfb_mask(lam) & neutral_real_bfb_mask(lam)


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
    """(N,) boolean: every S-matrix eigenvalue satisfies |a| ≤ 16π.

    Chunked: the 18 eigenvalue columns are ~150 MB per million points, and a
    scan calls this more than once.
    """
    return _chunked(
        lambda block: (np.abs(unitarity_eigenvalues(block)) <= bound).all(axis=1),
        lam)


# --------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------

def theory_mask(lam, bound=UNITARITY_BOUND, strict=True):
    """(N,) boolean: bounded from below AND tree-unitary.

    ``strict=True`` (default) uses `strict_bfb_mask`, i.e. the quoted Eq. (4)
    conditions *plus* the corrected neutral-direction condition derived in
    `neutral_real_bfb_min`.  ``strict=False`` reproduces the literature-only
    behaviour, for comparison.
    """
    bfb = strict_bfb_mask(lam) if strict else bfb_mask(lam)
    return bfb & unitarity_mask(lam, bound=bound)


def constraint_report(lam, bound=UNITARITY_BOUND):
    """Per-condition survival fractions — which cut is actually doing the work."""
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    bfb = bfb_conditions(lam)
    labels = [f"BFB Eq.(4{c})" for c in "abcdefg"]
    out = {lab: float(bfb[:, i].mean()) for i, lab in enumerate(labels)}
    out["BFB Eq.(4) all"] = float(bfb.all(axis=1).mean())
    out["neutral cond. (derived)"] = float(neutral_real_bfb_mask(lam).mean())
    out["BFB strict"] = float(strict_bfb_mask(lam).mean())
    out["unitarity Eq.(36)"] = float(unitarity_mask(lam, bound).mean())
    out["combined"] = float(theory_mask(lam, bound).mean())
    return out
