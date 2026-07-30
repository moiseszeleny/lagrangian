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
    align: dict                # {v1.s: v2.s/√3}; empty when soft-broken
    tadpole_sol: dict          # (mu0sq, mu1sq) on the aligned vacuum
    softs: dict = _field(default_factory=dict)   # soft S₃-breaking quadratics
    _cache: dict = _field(default_factory=dict, repr=False)

    @property
    def soft(self):
        """Whether the soft S₃-breaking quadratics are present."""
        return bool(self.softs)

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


def build_model(vevs=(200.0, 115.0, 80.0), tex=False, check=False,
                soft=False) -> S3Model:
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
    soft : add the four CP-conserving **soft S₃-breaking** quadratics
        (`SOFT_LABELS`).  Soft breaking is dimension-2 by definition, so it
        touches no quartic — every boundedness / unitarity result in
        `constraints.py` and `01_scalar_parameter_space.ipynb` is unaffected.
        What it *does* change is the vacuum: the tadpole system stops being
        over-constrained, so the √3 alignment is no longer forced and
        ``align`` comes back empty.  This is what lets the residual Z₂ break
        and, in turn, what generates a non-zero Cabibbo angle
        (`02_s3_fermion_sector.ipynb` §11, [DasDeyPal16]).
    """
    nm = (lambda plain, latex: latex if tex else plain)

    gw = ExternalParameter(nm("gw", "g_w"), 0.6535, positive=True)
    g1 = ExternalParameter(nm("g1", "g_1"), 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    s3 = S3()

    v1 = ExternalParameter(nm("v_1", "v_1"), vevs[0], positive=True, unit_dim=1)
    v2 = ExternalParameter(nm("v_2", "v_2"), vevs[1], positive=True, unit_dim=1)
    vS = ExternalParameter(nm("v_S", "v_S"), vevs[2], positive=True, unit_dim=1)
    lams = {k: ExternalParameter(nm(f"lambda_{k}", f"lambda_{k}"), 0.05 * k)
            for k in range(1, 9)}
    mu0sq = InternalParameter("mu0sq", unit_dim=2)
    mu1sq = InternalParameter("mu1sq", unit_dim=2)

    def doublet(name):
        return Scalar(name, reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                      component_names=[f"{name}p", f"{name}0"])

    H1 = doublet(nm("H_1", "H_1"))
    H2 = doublet(nm("H_2", "H_2"))
    HS = doublet(nm("H_S", "H_S"))
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

    # --- soft S₃ breaking: quadratic only, so no quartic is touched ---------
    softs = {}
    if soft:
        herm = lambda e: e + sp.conjugate(e)
        structures = {
            "mD1sq": d2[0],                 # x11 − x22        } the 2 of 2⊗2
            "mD2sq": d2[1],                 # −(x12 + x21)     }
            "mS1sq": herm(s1),              # H_S†H₁ + h.c.    } the S-doublet
            "mS2sq": herm(s2),              # H_S†H₂ + h.c.    }
        }
        for name, struct in structures.items():
            p = ExternalParameter(nm(name, name), 0.0, unit_dim=2)
            softs[name] = p
            V += p.s * struct

    L = Lagrangian().add(-V, sector="potential")
    model = Model("3HDM-S3", gauge_groups=[SU2L, U1Y], discrete_groups=[s3],
                  fields=[H1, H2, HS],
                  parameters=[gw, g1, v1, v2, vS, mu0sq, mu1sq, *lams.values(),
                              *softs.values()],
                  lagrangian=L)

    if check:
        model.check_invariance(raise_on_failure=True)

    tadpoles = model.tadpoles()
    if soft:
        # With four extra quadratic parameters the system is no longer
        # over-constrained: all three tadpoles become ordinary equations, and a
        # *general* (v₁, v₂, v_S) minimizes the potential.  Nothing forces the
        # alignment, so `align` is empty — that is the whole point.
        align = {}
        unknowns = [mu0sq.s, mu1sq.s, softs["mD2sq"].s]
        sol = sp.solve([sp.Eq(tadpoles[v.s], 0) for v in (v1, v2, vS)],
                       unknowns, dict=True)[0]
    else:
        # The tadpole system is deliberately OVER-constrained — that is the
        # point of the model.  Model.solve_tadpoles() does not apply here
        # (contrast examples/thdm.py, where it does): two conditions fix
        # (μ0², μ1²) and the third is not an equation for a parameter but the
        # vacuum-alignment residual, [GomezBock21] Eq. (13).  So impose the
        # alignment first, then solve the remaining two.
        align = {v1.s: v2.s / sp.sqrt(3)}
        sol = sp.solve([sp.Eq(tadpoles[v2.s].subs(align), 0),
                        sp.Eq(tadpoles[vS.s].subs(align), 0)],
                       [mu0sq.s, mu1sq.s], dict=True)[0]

    return S3Model(model=model, s3=s3, SU2L=SU2L, U1Y=U1Y,
                   doublets=(H1, H2, HS),
                   vevs={"v1": v1, "v2": v2, "vS": vS},
                   lams=lams, mus=(mu0sq, mu1sq),
                   align=align, tadpole_sol=sol, softs=softs)


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


# --------------------------------------------------------------------------
# the quartic potential, for boundedness checks
# --------------------------------------------------------------------------

#: Slot order of `quartic_potential`'s 12 real arguments: the 6 real parts, then
#: the 6 imaginary parts, of (H₁⁺, H₁⁰, H₂⁺, H₂⁰, H_S⁺, H_S⁰).  Indices 1, 3, 5
#: are the neutral legs.
NEUTRAL_INDICES = (1, 3, 5)
CHARGED_INDICES = (0, 2, 4)


def component_symbols(m: S3Model):
    """The six doublet-component symbols, in `NEUTRAL_INDICES` slot order.

    Read off the `Field` objects rather than reconstructed from hardcoded names:
    the parameter/field naming is a `build_model(tex=...)` option, so any name
    string baked in here would silently stop matching if it changed — and a
    ``subs`` keyed on a stale name is a **no-op**, not an error, so the failure
    would be invisible.
    """
    return [c for H in m.doublets for c in H.components]


def quartic_potential(m: S3Model):
    """Lambdified V₄(x₀…x₅, y₀…y₅, λ₁…λ₈) — the quartic part of the potential.

    Each doublet component is split as ``Hᵢ = xᵢ + i yᵢ`` over
    `component_symbols`, giving 12 real degrees of freedom.  Vectorized
    (``numpy``), so a whole block of field directions evaluates at once.

    Used to test boundedness from below *directly*, rather than trusting the
    analytic conditions in `constraints.py`: V₄ is homogeneous of degree 4, so
    the potential is bounded from below iff V₄ ≥ 0 on the unit 12-sphere.
    """
    if "V4" in m._cache:
        return m._cache["V4"]

    lam = m.lam_symbols
    V = -sum(t.expr for t in m.model.lagrangian.terms)
    V4 = sum(t for t in sp.expand(V).args if any(t.has(l) for l in lam))

    comps = component_symbols(m)
    xr = sp.symbols("x0:6", real=True)
    yi = sp.symbols("y0:6", real=True)
    split = {c: xr[i] + sp.I * yi[i] for i, c in enumerate(comps)}
    real_part = sp.expand(sp.expand(V4.subs(split)).as_real_imag()[0])

    leftover = real_part.free_symbols - set(xr) - set(yi) - set(lam)
    if leftover:
        raise ValueError(f"V4 still contains field symbols after the split: "
                         f"{sorted(map(str, leftover))} — component_symbols(m) "
                         f"does not match the fields in the Lagrangian")

    m._cache["V4"] = sp.lambdify(list(xr) + list(yi) + lam, real_part, "numpy")
    return m._cache["V4"]


def numeric_bfb_min(m: S3Model, lam_points, n_directions=20_000, seed=0,
                    subspace="all", chunk=64):
    """Smallest V₄ found over random unit field directions, per λ point.

    A negative entry is a **proof** that the point is not bounded from below —
    an explicit direction along which V₄ < 0, and V₄ being degree-4 homogeneous
    that direction runs to −∞.  A non-negative entry is only evidence, since the
    search is a finite sample of the unit 12-sphere.

    `subspace` selects which directions are sampled: ``"all"`` (12 real dof),
    ``"neutral"`` (charged legs zero), or ``"neutral-real"`` (charged legs and
    all phases zero) — the slice `constraints.neutral_real_bfb_min` solves in
    closed form, kept here as its cross-check.
    """
    import numpy as np

    fn = quartic_potential(m)
    lam_points = np.atleast_2d(np.asarray(lam_points, dtype=float))
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(12, n_directions))
    if subspace in ("neutral", "neutral-real"):
        for i in CHARGED_INDICES:
            X[i] = 0.0
            X[i + 6] = 0.0
    if subspace == "neutral-real":
        for i in NEUTRAL_INDICES:
            X[i + 6] = 0.0
    X /= np.linalg.norm(X, axis=0)

    out = np.empty(len(lam_points))
    for start in range(0, len(lam_points), chunk):
        block = lam_points[start:start + chunk]
        # broadcast: fields (12, 1, n_dir) against lambdas (n_block, 1)
        args = [X[i][None, :] for i in range(12)] + [block[:, k][:, None]
                                                     for k in range(8)]
        out[start:start + chunk] = fn(*args).min(axis=1)
    return out


def check_spectrum_against_eigenvals(m: S3Model, lam_values, v2_val, vS_val,
                                     rtol=1e-8):
    """Cross-check the rotation-ansatz CP-even masses against brute force.

    The rotation gives (h0, H1, H2) analytically; `M_S.eigenvals()` gives them
    by direct numeric diagonalization of the un-rotated matrix.  Agreement is
    an independent check that the geometric ansatz really diagonalizes what
    feynlag derived.  Returns (via_rotation, brute_force, agree).
    """
    spec = spectrum_function(m)(lam_values, v2_val, vS_val)
    subs = _numeric_subs(m, lam_values, v2_val, vS_val)

    M_S = mass_matrices(m)[0]
    brute = sorted(complex(e).real for e in sp.N(M_S.subs(subs)).eigenvals())
    via = sorted(spec[k] for k in ("h0", "H1", "H2"))
    return via, brute, _agree(via, brute, rtol)


def _numeric_subs(m: S3Model, lam_values, v2_val, vS_val):
    subs = dict(zip(m.lam_symbols, lam_values))
    subs[m.v2.s] = v2_val
    subs[m.vS.s] = vS_val
    return subs


def _agree(via, brute, rtol):
    scale = max(abs(x) for x in list(brute) + list(via)) or 1.0
    return all(abs(a - b) <= rtol * scale for a, b in zip(via, brute))


def check_sectors_against_eigenvals(m: S3Model, lam_values, v2_val, vS_val,
                                    rtol=1e-8):
    """The `check_spectrum_against_eigenvals` cross-check, for *all three* sectors.

    The rotation ansatz claims more than the CP-even spectrum: it claims the
    CP-odd and charged 3×3 matrices are diagonalized outright, with a Goldstone
    at [0,0].  Checking only the CP-even sector leaves that untested.  Here each
    sector's rotation-basis masses — including the Goldstone zeros — are compared
    against direct numeric diagonalization of the *un-rotated* mass matrix.

    Returns ``{sector: (via_rotation, brute_force, agree)}`` with sector in
    ``("CP-even", "CP-odd", "charged")``, each list sorted ascending.
    """
    spec = spectrum_function(m)(lam_values, v2_val, vS_val)
    subs = _numeric_subs(m, lam_values, v2_val, vS_val)
    M_S, M_A, M_C = mass_matrices(m)

    # the Goldstone is a genuine eigenvalue of M_A/M_C, so include the 0
    wanted = {"CP-even": (M_S, [spec[k] for k in ("h0", "H1", "H2")]),
              "CP-odd": (M_A, [0.0, spec["A1"], spec["A2"]]),
              "charged": (M_C, [0.0, spec["Hpm1"], spec["Hpm2"]])}

    out = {}
    for name, (M, via) in wanted.items():
        brute = sorted(complex(e).real for e in sp.N(M.subs(subs)).eigenvals())
        via = sorted(via)
        out[name] = (via, brute, _agree(via, brute, rtol))
    return out
