"""Reusable 3HDM-S₃ build: the algebra, lifted out of notebook scope.

`examples/thdm_s3.py` stops at the weak-basis CP-even mass matrix and returns
nothing; everything past it — the pseudoscalar and charged mass matrices, the
Gómez-Bock–Mondragón–Pérez-Martínez geometric rotation, the leftover CP-even
2×2 block, the numeric spectrum — lives only as local variables inside
`examples/THDM_S3_Tutorial.ipynb`.  This module is that physics made
importable, so research notebooks can build on it instead of re-deriving it.

Algebra only: no scanning, no plotting, no constraint evaluation.  The heavy
symbolic steps are cached per model instance, so re-running a notebook cell is
free after the first call.

References
----------
[GomezBock21] M. Gómez-Bock, M. Mondragón, A. Pérez-Martínez, "Scalar and
    gauge sectors in the 3-Higgs Doublet Model under the S₃-symmetry",
    Eur. Phys. J. C 81, 942 (2021), arXiv:2102.02800,
    doi:10.1140/epjc/s10052-021-09731-3.
    Eq. (8) is the v₁²+v₂²+v_S² = v² constraint used by `aligned_vevs`;
    Eq. (13) is the √3 alignment; Eq. (25)–(29) the rotation `rotation()`
    builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import Callable

import sympy as sp

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, S3, SU2,
    Scalar, U1, dag, diagonalize_orthogonal_2x2, solve_mixing_angle_2x2,
)

#: Electroweak scale, GeV — the [GomezBock21] Eq. (8) constraint
#: √(v₁²+v₂²+v_S²) = v that `aligned_vevs` enforces.
V_EW = 246.0


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

@dataclass(eq=False)          # eq=False keeps the default identity hash, so
class S3Model:                # the lru_caches below can key on the instance
    """Everything the 3HDM-S₃ algebra needs, in one bundle."""

    model: Model
    s3: S3
    SU2L: SU2
    U1Y: U1
    doublets: tuple            # (H1, H2, HS)
    vevs: dict                 # {"v1": ExternalParameter, "v2": ..., "vS": ...}
    lams: dict                 # {1..8: ExternalParameter}
    mus: tuple                 # (mu0sq, mu1sq)
    align: dict                # {v1.s: v2.s/√3}
    tadpole_sol: dict          # (mu0sq, mu1sq) on the aligned vacuum
    _cache: dict = _field(default_factory=dict, repr=False)

    # convenience accessors used constantly by the notebooks
    @property
    def v1(self):
        return self.vevs["v1"]

    @property
    def v2(self):
        return self.vevs["v2"]

    @property
    def vS(self):
        return self.vevs["vS"]

    @property
    def lam_symbols(self):
        """[λ₁.s, …, λ₈.s], the scan variables."""
        return [self.lams[k].s for k in range(1, 9)]

    def on_vacuum(self, expr):
        """Substitute the tadpole solution and the √3 alignment into `expr`."""
        return expr.subs(self.tadpole_sol).subs(self.align)


def build_model(vevs=(200.0, 115.0, 80.0), tex=False, check=False) -> S3Model:
    """Build the S₃-invariant 3HDM and solve its tadpoles on the alignment.

    Parameters
    ----------
    vevs : (v1, v2, vS) numeric defaults for the VEV parameters.  These are
        only the ``.value`` attached to each ``ExternalParameter`` — every
        result below is symbolic in them.  Note the defaults are the tutorial's
        *unaligned* numbers; use `aligned_vevs` for a vacuum that satisfies
        both the √3 alignment and the electroweak-scale constraint.
    tex : use LaTeX-styled parameter names (``v_1``, ``lambda_1``, ``H_1``) as
        the tutorial notebook does, instead of the plain ASCII names.
    check : also run ``model.check_invariance()`` and raise if it fails.
        Off by default because it costs a few seconds and is pinned in
        ``tests/test_thdm_s3.py``.
    """
    nm = (lambda plain, latex: latex if tex else plain)

    gw = ExternalParameter(nm("gw", "g_w"), 0.6535, positive=True)
    g1 = ExternalParameter(nm("g1", "g_1"), 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    s3 = S3()

    v1 = ExternalParameter(nm("v1", "v_1"), vevs[0], positive=True, unit_dim=1)
    v2 = ExternalParameter(nm("v2", "v_2"), vevs[1], positive=True, unit_dim=1)
    vS = ExternalParameter(nm("vS", "v_S"), vevs[2], positive=True, unit_dim=1)
    lams = {k: ExternalParameter(nm(f"lm{k}", f"lambda_{k}"), 0.05 * k)
            for k in range(1, 9)}
    mu0sq = InternalParameter("mu0sq", unit_dim=2)
    mu1sq = InternalParameter("mu1sq", unit_dim=2)

    def doublet(name):
        return Scalar(name, reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                      component_names=[f"{name}p", f"{name}0"])

    H1 = doublet(nm("H1", "H_1"))
    H2 = doublet(nm("H2", "H_2"))
    HS = doublet(nm("HS", "H_S"))
    s3.assign("2", H1, H2)
    s3.assign("1", HS)
    H1.expand_vev({H1.components[1]: v1})
    H2.expand_vev({H2.components[1]: v2})
    HS.expand_vev({HS.components[1]: vS})

    def bra(a, b):
        return (dag(a) * b.mat)[0]

    x11, x22 = bra(H1, H1), bra(H2, H2)
    x12, x21 = bra(H1, H2), bra(H2, H1)
    s1, s2 = bra(HS, H1), bra(HS, H2)
    sss = bra(HS, HS)

    # S₃ CG contractions of 2⊗2 = 1 ⊕ 1' ⊕ 2 in the real-orthogonal basis.
    # Written inline, matching examples/thdm_s3.py:52-60; the equivalent
    # s3.doublet_product() route is exercised in tests/test_thdm_s3.py.
    inv1 = x11 + x22
    inv1p = x12 - x21
    d2 = (x11 - x22, -(x12 + x21))

    lam4_term = s1 * d2[0] + s2 * d2[1]
    lam4_term += sp.conjugate(lam4_term)
    lam7_term = s1**2 + s2**2
    lam7_term += sp.conjugate(lam7_term)

    l = {k: p.s for k, p in lams.items()}
    V = (mu1sq.s * inv1 + mu0sq.s * sss
         + l[1] * inv1**2 + l[2] * inv1p**2
         + l[3] * (d2[0]**2 + d2[1]**2)
         + l[4] * lam4_term
         + l[5] * sss * inv1
         + l[6] * (s1 * bra(H1, HS) + s2 * bra(H2, HS))
         + l[7] * lam7_term
         + l[8] * sss**2)

    L = Lagrangian().add(-V, sector="potential")
    model = Model("3HDM-S3", gauge_groups=[SU2L, U1Y], discrete_groups=[s3],
                  fields=[H1, H2, HS],
                  parameters=[gw, g1, v1, v2, vS, mu0sq, mu1sq, *lams.values()],
                  lagrangian=L)

    if check:
        model.check_invariance(raise_on_failure=True)

    # The tadpole system is deliberately OVER-constrained — that is the point
    # of the model.  Model.solve_tadpoles() does not apply here (contrast
    # examples/thdm.py, where it does): two conditions fix (μ0², μ1²) and the
    # third is not an equation for a parameter but the vacuum-alignment
    # residual, [GomezBock21] Eq. (13).  So impose the alignment first, then
    # solve the remaining two.
    align = {v1.s: v2.s / sp.sqrt(3)}
    tadpoles = model.tadpoles()
    sol = sp.solve([sp.Eq(tadpoles[v2.s].subs(align), 0),
                    sp.Eq(tadpoles[vS.s].subs(align), 0)],
                   [mu0sq.s, mu1sq.s], dict=True)[0]

    return S3Model(model=model, s3=s3, SU2L=SU2L, U1Y=U1Y,
                   doublets=(H1, H2, HS),
                   vevs={"v1": v1, "v2": v2, "vS": vS},
                   lams=lams, mus=(mu0sq, mu1sq),
                   align=align, tadpole_sol=sol)


def alignment_residual(m: S3Model):
    """The third tadpole with the other two solved: its roots are the √3 alignment.

    Returned factored, as `examples/thdm_s3.py` prints it.  Kept separate from
    `build_model` (which imposes the alignment up front) because a notebook
    that wants to *show* the alignment being forced needs the residual itself.
    """
    tad = m.model.tadpoles()
    sol = sp.solve([sp.Eq(tad[m.v2.s], 0), sp.Eq(tad[m.vS.s], 0)],
                   [m.mus[0].s, m.mus[1].s], dict=True)[0]
    return sp.factor(sp.expand(tad[m.v1.s].subs(sol)))


# --------------------------------------------------------------------------
# the vacuum
# --------------------------------------------------------------------------

def aligned_vevs(theta, v=V_EW):
    """Numeric (v1, v2, vS) satisfying BOTH vacuum conditions at once.

    The two conditions are the S₃ alignment v₁ = v₂/√3 ([GomezBock21] Eq. 13,
    in feynlag's real-orthogonal irrep basis — the literature basis has the
    roles swapped) and the electroweak-scale constraint v₁²+v₂²+v_S² = v²
    ([GomezBock21] Eq. 8).  Together they leave a one-parameter family,
    parametrized here by the same angle θ the geometric rotation uses:

        v₁₂ ≡ √(v₁²+v₂²) = v sinθ,   v_S = v cosθ

    and the alignment then fixes v₁ = v₁₂/2, v₂ = √3 v₁₂/2 exactly (so the
    rotation's φ is always 60°, independent of θ).

    This is what `examples/THDM_S3_Tutorial.ipynb` gets wrong: it picks
    (200, 115, 80) so that √Σvᵢ² ≈ 246, but then imposes v₁ → v₂/√3, which
    *replaces* v₁ = 200 by 66.4 and drops the total to 155 GeV.  Every mass in
    its §9 benchmark is low by a factor ≈ 246/155 = 1.59.
    """
    theta = float(theta)
    v12, vS = v * math.sin(theta), v * math.cos(theta)
    return v12 / 2.0, v12 * math.sqrt(3.0) / 2.0, vS


def vacuum_scale(v1, v2, vS):
    """√(v₁²+v₂²+v_S²) — the quantity that must equal 246 GeV."""
    return (v1 * v1 + v2 * v2 + vS * vS) ** 0.5


# --------------------------------------------------------------------------
# mass matrices and the geometric rotation
# --------------------------------------------------------------------------

def _neutral_symbols(m: S3Model, part):
    """The real (`part='r'`) or imaginary (`'i'`) fluctuation of each neutral leg.

    Read off `Field.vev_expansions` rather than hardcoding `"H10_r"`, so the
    `tex=True` naming works unchanged.
    """
    out = []
    for H in m.doublets:
        vev, re, im = H.vev_expansions[H.components[1]]
        out.append(re if part == "r" else im)
    return out


def mass_matrices(m: S3Model):
    """(M_S, M_A, M_C): CP-even, CP-odd and charged 3×3 mass matrices.

    All three on the aligned vacuum with the tadpole solution substituted, and
    fully simplified.  Cached — this is the expensive call (tens of seconds).
    """
    if "M" in m._cache:
        return m._cache["M"]

    def build(fields, charged=False):
        M = m.model.mass_matrix(fields, charged=charged) if charged \
            else m.model.mass_matrix(fields)
        M = m.on_vacuum(M)
        return M.applyfunc(lambda e: sp.simplify(sp.expand(e)))

    M_S = build(_neutral_symbols(m, "r"))
    M_A = build(_neutral_symbols(m, "i"))
    M_C = build([H.components[0] for H in m.doublets], charged=True)

    m._cache["M"] = (M_S, M_A, M_C)
    return m._cache["M"]


def rotation(m: S3Model):
    """The Gómez-Bock–Mondragón–Pérez-Martínez geometric rotation R.

    [GomezBock21] Eq. (25)–(29): built purely from the VEV geometry, with
    cosφ = v₁/v₁₂, sinφ = v₂/v₁₂, cosθ = v_S/v, sinθ = v₁₂/v.  Its first
    column is the vacuum direction, so RᵀMR isolates the Goldstones at [0,0]
    in the CP-odd and charged sectors.
    """
    if "R" in m._cache:
        return m._cache["R"]

    v1, v2, vS = m.v1.s, m.v2.s, m.vS.s
    v12 = sp.sqrt(v1**2 + v2**2)
    vtot = sp.sqrt(v1**2 + v2**2 + vS**2)
    cphi, sphi = v1 / v12, v2 / v12
    cth, sth = vS / vtot, v12 / vtot

    R = sp.Matrix([
        [sth * cphi, -sphi, -cth * cphi],
        [sth * sphi,  cphi, -cth * sphi],
        [cth,          0,    sth],
    ])
    m._cache["R"] = sp.simplify(R.subs(m.align))
    return m._cache["R"]


def diagonal_blocks(m: S3Model):
    """(D_S, D_A, D_C) = RᵀMR for each sector.

    D_A[0,0] and D_C[0,0] are the Goldstones and must be *exactly* zero.
    D_A and D_C come out fully diagonal; D_S retains a 2×2 block in the
    (0, 2) entries, finished off by `cp_even_angle`.
    """
    if "D" in m._cache:
        return m._cache["D"]
    R = rotation(m)
    m._cache["D"] = tuple(sp.simplify(R.T * M * R) for M in mass_matrices(m))
    return m._cache["D"]


def cp_even_block(m: S3Model):
    """The leftover CP-even 2×2 block, in the (0, 2) subspace of D_S."""
    D_S = diagonal_blocks(m)[0]
    return sp.Matrix([[D_S[0, 0], D_S[0, 2]],
                      [D_S[2, 0], D_S[2, 2]]])


def cp_even_angle(m: S3Model):
    """(theta_expr, tan2theta, Rotation) for the leftover CP-even 2×2 block.

    Uses the library's own analytic 2×2 route — `solve_mixing_angle_2x2` and
    `diagonalize_orthogonal_2x2` — rather than diagonalizing the full 3×3,
    whose cubic roots have no usable closed form.
    """
    if "angle" in m._cache:
        return m._cache["angle"]
    block = cp_even_block(m)
    theta_expr, tan2theta = solve_mixing_angle_2x2(block)
    h_a, h_b, H_1, H_2 = sp.symbols("h_a h_b H_1 H_2", real=True)
    rot = diagonalize_orthogonal_2x2(block, [h_a, h_b], [H_1, H_2])
    m._cache["angle"] = (theta_expr, tan2theta, rot)
    return m._cache["angle"]


# --------------------------------------------------------------------------
# numeric spectrum
# --------------------------------------------------------------------------

#: Order of the masses returned by `spectrum_function`.
SPECTRUM_KEYS = ("h0", "H1", "H2", "A1", "A2", "Hpm1", "Hpm2")


def _block_eigs(a, b, c):
    """Eigenvalues of the real-symmetric 2×2 [[a,b],[b,c]] in closed form.

    The discriminant (a−c)²+4b² is never negative, so no branch guard is
    needed.  Closed form rather than a numeric eigensolve because this runs
    once per scan point.
    """
    tr, det = a + c, a * c - b * b
    root = (tr * tr - 4.0 * det) ** 0.5
    return (tr + root) / 2.0, (tr - root) / 2.0


def spectrum_function(m: S3Model, modules="math") -> Callable:
    """Lambdified (λ₁…λ₈, v₂, v_S) → {name: mass², …} over `SPECTRUM_KEYS`.

    v₁ is *not* an argument: the alignment has already eliminated it in favour
    of v₂.  Feed it VEVs from `aligned_vevs` and drop the v₁ entry.

    The CP-even 2×2 block is passed through as its three independent entries
    and diagonalized numerically by `_block_eigs`; everything else is already
    diagonal in the geometric basis.

    ``modules="numpy"`` returns the same callable evaluated elementwise, so a
    scan can pass eight λ *arrays* and get arrays of masses back — the whole
    parameter scan then runs without a Python loop.
    """
    key = f"spectrum:{modules}"
    if key in m._cache:
        return m._cache[key]

    D_S, D_A, D_C = diagonal_blocks(m)
    exprs = [D_S[1, 1],                          # h0: the geometric-basis state
             D_S[0, 0], D_S[0, 2], D_S[2, 2],    # the 2×2 block entries
             D_A[1, 1], D_A[2, 2],               # A1, A2
             D_C[1, 1], D_C[2, 2]]               # H±1, H±2
    fn = sp.lambdify(m.lam_symbols + [m.v2.s, m.vS.s], exprs, modules)

    def spectrum(lam_values, v2_val, vS_val):
        h0, a, b, c, A1, A2, Hp1, Hp2 = fn(*lam_values, v2_val, vS_val)
        H1m, H2m = _block_eigs(a, b, c)
        return dict(zip(SPECTRUM_KEYS, (h0, H1m, H2m, A1, A2, Hp1, Hp2)))

    m._cache[key] = spectrum
    return spectrum


def check_spectrum_against_eigenvals(m: S3Model, lam_values, v2_val, vS_val,
                                     rtol=1e-8):
    """Cross-check the rotation-ansatz CP-even masses against brute force.

    The rotation gives (h0, H1, H2) analytically; `M_S.eigenvals()` gives them
    by direct numeric diagonalization of the un-rotated matrix.  Agreement is
    an independent check that the geometric ansatz really diagonalizes what
    feynlag derived.  Returns (via_rotation, brute_force, agree).
    """
    spec = spectrum_function(m)(lam_values, v2_val, vS_val)
    subs = dict(zip(m.lam_symbols, lam_values))
    subs[m.v2.s] = v2_val
    subs[m.vS.s] = vS_val

    M_S = mass_matrices(m)[0]
    brute = sorted(complex(e).real for e in sp.N(M_S.subs(subs)).eigenvals())
    via = sorted(spec[k] for k in ("h0", "H1", "H2"))
    scale = max(abs(x) for x in brute + via) or 1.0
    agree = all(abs(a - b) <= rtol * scale for a, b in zip(via, brute))
    return via, brute, agree
