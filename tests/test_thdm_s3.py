"""Phase 6 stress test: 3HDM with S₃ symmetry.

The S₃-invariant three-Higgs-doublet potential, with (H1, H2) an S₃ doublet
and HS the S₃ singlet, built entirely from the library's own CG products:

V = μ1² (H1†H1 + H2†H2) + μ0² HS†HS
  + λ1 (H1†H1 + H2†H2)²          [1 of 2⊗2, squared]
  + λ2 (H1†H2 − H2†H1)²          [1' of 2⊗2, squared]
  + λ3 [(H1†H1 − H2†H2)² + (H1†H2 + H2†H1)²]   [2 of 2⊗2, squared]
  + λ4 [HS†H_CG-doublet contraction + h.c.]
  + λ5 (HS†HS)(H1†H1 + H2†H2)
  + λ6 [(HS†H1)(H1†HS) + (HS†H2)(H2†HS)]
  + λ7 [(HS†H1)² + (HS†H2)² + h.c.]
  + λ8 (HS†HS)²

Physics checks:
- every term passes gauge (SU2×U1) AND S₃ invariance; a forbidden term fails;
- with all three VEVs, the tadpole conditions over-constrain (μ0², μ1²) and
  force the alignment v1² = 3 v2² (or the equivalent branch) — the relation
  the user's 3HDM-S₃ papers rely on (v₁ = √3 v₂);
- CP-even 3×3 mass matrix is symmetric with the expected structure.
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, S3, SU2,
    Scalar, U1, dag,
)


@pytest.fixture(scope="module")
def s3_model():
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    s3 = S3()

    v1 = ExternalParameter("v1", 200.0, positive=True, unit_dim=1)
    v2 = ExternalParameter("v2", 115.0, positive=True, unit_dim=1)
    vS = ExternalParameter("vS", 80.0, positive=True, unit_dim=1)
    lams = {k: ExternalParameter(f"lm{k}", 0.05 * k) for k in range(1, 9)}
    mu0sq = InternalParameter("mu0sq", unit_dim=2)
    mu1sq = InternalParameter("mu1sq", unit_dim=2)

    def doublet(name):
        return Scalar(name, reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                      component_names=[f"{name}p", f"{name}0"])

    H1, H2, HS = doublet("H1"), doublet("H2"), doublet("HS")
    s3.assign("2", H1, H2)
    s3.assign("1", HS)

    H1.expand_vev({H1.components[1]: v1})
    H2.expand_vev({H2.components[1]: v2})
    HS.expand_vev({HS.components[1]: vS})

    def bra(a, b):
        return (dag(a) * b.mat)[0]

    x11, x22, x12, x21 = bra(H1, H1), bra(H2, H2), bra(H1, H2), bra(H2, H1)
    s11, s22 = bra(HS, H1), bra(HS, H2)          # HS†H_i : S3 doublet
    s11c, s22c = bra(H1, HS), bra(H2, HS)
    sss = bra(HS, HS)

    # CG contractions of the (H1,H2) doublet with itself
    # 1  : x11 + x22 ; 1' : x12 − x21 ; 2 : (x11 − x22, −(x12 + x21))
    cg = s3.doublet_product((sp.Symbol("_a1"), sp.Symbol("_a2")),
                            (sp.Symbol("_b1"), sp.Symbol("_b2")))
    # build with actual bilinears: bra-side (H1†, H2†) and ket-side (H1, H2)
    sub = {sp.Symbol("_a1") * sp.Symbol("_b1"): x11,
           sp.Symbol("_a1") * sp.Symbol("_b2"): x12,
           sp.Symbol("_a2") * sp.Symbol("_b1"): x21,
           sp.Symbol("_a2") * sp.Symbol("_b2"): x22}

    def cg_sub(expr):
        return sp.expand(expr).subs(sub, simultaneous=True)

    inv1 = cg_sub(cg["1"])            # x11 + x22
    inv1p = cg_sub(cg["1p"])          # x12 − x21
    d2_1, d2_2 = cg_sub(cg["2"][0]), cg_sub(cg["2"][1])

    # λ4 invariant: (HS†H)₂ ⊗ (H†H)₂ → 1 :  s11·d1 + s22·d2, + h.c.
    lam4_term = s11 * d2_1 + s22 * d2_2
    lam4_term = lam4_term + sp.conjugate(lam4_term)
    # λ7: (HS†H)₂ ⊗ (HS†H)₂ → 1 : s11² + s22², + h.c.
    lam7_term = s11**2 + s22**2
    lam7_term = lam7_term + sp.conjugate(lam7_term)

    l = {k: lams[k].s for k in lams}
    V = (mu1sq.s * inv1 + mu0sq.s * sss
         + l[1] * inv1**2
         + l[2] * inv1p**2
         + l[3] * (d2_1**2 + d2_2**2)
         + l[4] * lam4_term
         + l[5] * sss * inv1
         + l[6] * (s11 * s11c + s22 * s22c)
         + l[7] * lam7_term
         + l[8] * sss**2)

    L = Lagrangian().add(-V, sector="potential")
    model = Model("3HDM-S3", gauge_groups=[SU2L, U1Y], discrete_groups=[s3],
                  fields=[H1, H2, HS],
                  parameters=[gw, g1, v1, v2, vS, mu0sq, mu1sq,
                              *lams.values()],
                  lagrangian=L)
    return model, s3, (H1, H2, HS), (v1, v2, vS), (mu0sq, mu1sq), l


def test_invariance_full_potential(s3_model):
    model, s3, fields, vevs, mus, l = s3_model
    report = model.check_invariance()
    assert report.ok, report.failures


def test_forbidden_term_fails(s3_model):
    from feynlag import check_discrete_invariance
    model, s3, (H1, H2, HS), *_ = s3_model
    bad = (dag(HS) * H1.mat)[0] * (dag(H1) * H1.mat)[0]
    bad = bad + sp.conjugate(bad)
    ok, _ = check_discrete_invariance(bad, s3)
    assert not ok


def test_tadpole_alignment_sqrt3(s3_model):
    """Solving t2, tS for (μ0², μ1²) leaves a residual third condition whose
    non-trivial solution is the S₃ alignment  v² ratio = 3.

    Note on basis: the literature (Gómez-Bock et al.) quotes v₁ = √3 v₂; in
    feynlag's real-orthogonal S₃ basis the roles of the doublet components
    are swapped (an equivalent irrep, related by the reflection), so the
    alignment appears as v₂ = √3 v₁ — the ratio squared is 3 either way."""
    model, s3, fields, (v1, v2, vS), (mu0sq, mu1sq), l = s3_model

    tadpoles = model.tadpoles()
    t1, t2, tS = tadpoles[v1.s], tadpoles[v2.s], tadpoles[vS.s]

    # solve the v2 and vS conditions for the two mass parameters
    sol = sp.solve([sp.Eq(t2, 0), sp.Eq(tS, 0)], [mu0sq.s, mu1sq.s],
                   dict=True)
    assert len(sol) == 1
    residual = sp.factor(sp.expand(t1.subs(sol[0])))

    # the residual must vanish only on alignment: find its v1 solutions
    solutions = sp.solve(sp.Eq(residual, 0), v1.s)
    ratios = set()
    for s_v1 in solutions:
        r = sp.simplify((s_v1 / v2.s) ** 2)
        if not r.free_symbols:               # pure number
            ratios.add(sp.nsimplify(r))
    assert ratios & {sp.Integer(3), sp.Rational(1, 3)}, (solutions, ratios)


def test_cp_even_mass_matrix_structure(s3_model):
    model, s3, fields, (v1, v2, vS), (mu0sq, mu1sq), l = s3_model
    # impose the alignment (v2 = √3 v1 in this basis) and solve all
    # tadpoles consistently
    align = {v1.s: v2.s / sp.sqrt(3)}

    tadpoles = model.tadpoles()
    sol = sp.solve([sp.Eq(tadpoles[v2.s].subs(align), 0),
                    sp.Eq(tadpoles[vS.s].subs(align), 0)],
                   [mu0sq.s, mu1sq.s], dict=True)[0]
    # the v1 tadpole is then automatically satisfied
    assert sp.simplify(tadpoles[v1.s].subs(align).subs(sol)) == 0

    h1, h2, hS = (sp.Symbol("H10_r", real=True),
                  sp.Symbol("H20_r", real=True),
                  sp.Symbol("HS0_r", real=True))
    M = model.mass_matrix([h1, h2, hS])
    M = M.subs(sol).subs(align)
    M = M.applyfunc(lambda e: sp.simplify(sp.expand(e)))

    # symmetric, and no vanishing diagonal in general
    assert sp.simplify(M - M.T) == sp.zeros(3, 3)
    assert M[0, 0] != 0 and M[1, 1] != 0 and M[2, 2] != 0


# ---------------------------------------------------------------------------
# All three scalar sectors + the geometric rotation.
#
# These pin what previously lived only as print statements inside
# examples/THDM_S3_Tutorial.ipynb §6-§9.  Built here from this file's own
# fixture, deliberately NOT importing research/thdm_s3/model.py — research
# code churns, and the suite must not be hostage to it (see research/README.md).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s3_sectors(s3_model):
    """(M_S, M_A, M_C, R, v-symbols) on the aligned vacuum, tadpoles solved."""
    model, s3, (H1, H2, HS), (v1, v2, vS), (mu0sq, mu1sq), l = s3_model
    align = {v1.s: v2.s / sp.sqrt(3)}
    tad = model.tadpoles()
    sol = sp.solve([sp.Eq(tad[v2.s].subs(align), 0),
                    sp.Eq(tad[vS.s].subs(align), 0)],
                   [mu0sq.s, mu1sq.s], dict=True)[0]

    def build(fields, charged=False):
        M = (model.mass_matrix(fields, charged=True) if charged
             else model.mass_matrix(fields))
        return M.subs(sol).subs(align).applyfunc(
            lambda e: sp.simplify(sp.expand(e)))

    M_S = build([sp.Symbol(f"{n}0_r", real=True) for n in ("H1", "H2", "HS")])
    M_A = build([sp.Symbol(f"{n}0_i", real=True) for n in ("H1", "H2", "HS")])
    M_C = build([H1.components[0], H2.components[0], HS.components[0]],
                charged=True)

    # the Gomez-Bock-Mondragon-Perez-Martinez geometric ansatz, [GomezBock21]
    # Eq. (29): first column is the vacuum direction.
    v12 = sp.sqrt(v1.s**2 + v2.s**2)
    vtot = sp.sqrt(v1.s**2 + v2.s**2 + vS.s**2)
    cphi, sphi = v1.s / v12, v2.s / v12
    cth, sth = vS.s / vtot, v12 / vtot
    R = sp.simplify(sp.Matrix([
        [sth * cphi, -sphi, -cth * cphi],
        [sth * sphi, cphi, -cth * sphi],
        [cth, 0, sth],
    ]).subs(align))
    return M_S, M_A, M_C, R, (v1, v2, vS), l


def test_pseudoscalar_and_charged_mass_matrices(s3_sectors):
    """M_A and M_C are symmetric with non-vanishing diagonals, like M_S."""
    M_S, M_A, M_C, R, vevs, l = s3_sectors
    for M in (M_A, M_C):
        assert sp.simplify(M - M.T) == sp.zeros(3, 3)
        assert all(M[i, i] != 0 for i in range(3))


def test_geometric_rotation_isolates_goldstones(s3_sectors):
    """RᵀMR puts an EXACT zero at [0,0] in the CP-odd and charged sectors.

    One Goldstone each (eaten by Z and W±); the ansatz is purely geometric —
    built from the VEVs, with no reference to the quartics — so this vanishing
    is a nontrivial statement about the potential, not an identity.
    """
    M_S, M_A, M_C, R, vevs, l = s3_sectors
    assert sp.simplify(R.T * R) == sp.eye(3)          # orthogonal

    for M in (M_A, M_C):
        D = sp.simplify(R.T * M * R)
        assert D[0, 0] == 0
        # and fully diagonal in the remaining 2x2
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert sp.simplify(D[i, j]) == 0


def test_cp_even_block_diagonalizes_to_2x2(s3_sectors):
    """D_S = RᵀM_S R keeps exactly one off-diagonal pair, the (0,2) block."""
    M_S, M_A, M_C, R, vevs, l = s3_sectors
    D = sp.simplify(R.T * M_S * R)
    assert sp.simplify(D[0, 1]) == 0 and sp.simplify(D[1, 2]) == 0
    assert sp.simplify(D[0, 2]) != 0                   # the surviving mixing
    assert sp.simplify(D[0, 2] - D[2, 0]) == 0


def test_masses_match_gomezbock_closed_forms(s3_sectors):
    """feynlag's derived masses reproduce [GomezBock21] Eqs. (30)-(33) exactly.

    This is the end-to-end validation of the 3HDM-S₃ chain against published
    closed forms — potential, tadpoles, mass matrices and rotation at once —
    and it is what *fixes the parameter dictionary* to the literature:

        a=2λ₈, b=λ₅, c=2λ₁, d=2λ₂, e=−λ₄, f=λ₆, g=2λ₃, h=2λ₇

    The e = −λ₄ sign matters: feynlag's real-orthogonal S₃ doublet basis is not
    the literature's (the λ₄ invariant transcribed literally from [DasDey14]
    is not even S₃-invariant here).  Because the [DasDey14] boundedness and
    unitarity conditions involve λ₄ only as |λ₄| or λ₄², they nevertheless
    transfer to feynlag's λ's unchanged — which is what
    research/thdm_s3/constraints.py relies on.

    [GomezBock21] M. Gómez-Bock, M. Mondragón, A. Pérez-Martínez,
        Eur. Phys. J. C 81, 942 (2021), arXiv:2102.02800,
        doi:10.1140/epjc/s10052-021-09731-3.
    [DasDey14] D. Das, U. K. Dey, Phys. Rev. D 89, 095025 (2014),
        arXiv:1404.2491, doi:10.1103/PhysRevD.89.095025.
    """
    M_S, M_A, M_C, R, (v1, v2, vS), l = s3_sectors
    D_A = sp.simplify(R.T * M_A * R)
    D_C = sp.simplify(R.T * M_C * R)

    # the paper's vacuum parametrization: v12 = v sinθ, vS = v cosθ, with the
    # alignment fixing v2 = √3 v12/2 in feynlag's basis ([GomezBock21] Eq. 24)
    v, th = sp.symbols("v theta", positive=True)
    vac = {v2.s: v * sp.sin(th) * sp.sqrt(3) / 2, vS.s: v * sp.cos(th)}

    a, b, c, d, e, f, g, h = sp.symbols("a b c d e f g h")
    dictionary = {a: 2 * l[8], b: l[5], c: 2 * l[1], d: 2 * l[2],
                  e: -l[4], f: l[6], g: 2 * l[3], h: 2 * l[7]}

    published = {                                        # [GomezBock21]
        D_A[1, 1]: -v**2 * ((d + g) * sp.sin(th)**2
                            + sp.Rational(5, 4) * e * sp.sin(2 * th)
                            + h * sp.cos(th)**2),                     # Eq. (30)
        D_A[2, 2]: -v**2 * (e / 2 * sp.tan(th) + h),                  # Eq. (31)
        D_C[1, 1]: -v**2 / 4 * (5 * e * sp.sin(2 * th)
                                + 2 * (f + h) * sp.cos(th)**2
                                + 4 * g * sp.sin(th)**2),             # Eq. (32)
        D_C[2, 2]: -v**2 / 2 * (e * sp.tan(th) + (f + h)),            # Eq. (33)
    }

    for derived, closed_form in published.items():
        diff = sp.simplify(sp.expand_trig(sp.simplify(
            derived.subs(vac) - closed_form.subs(dictionary))))
        assert diff == 0, diff

    # and the opposite sign genuinely fails, so the test has teeth
    wrong = {**dictionary, e: l[4]}
    bad = sp.simplify(sp.expand_trig(sp.simplify(
        D_A[2, 2].subs(vac) - published[D_A[2, 2]].subs(wrong))))
    assert bad != 0


def test_aligned_vacuum_can_meet_the_electroweak_scale(s3_model):
    """The alignment and √Σvᵢ² = 246 GeV are simultaneously satisfiable.

    Regression for a real bug found in examples/THDM_S3_Tutorial.ipynb: it
    picks (v1,v2,vS) = (200,115,80) so that √Σvᵢ² ≈ 246 ([GomezBock21] Eq. 8),
    then imposes the alignment as the substitution v1 → v2/√3, which *replaces*
    v1 = 200 by 66.4 and silently drops the vacuum to 155 GeV.  Imposing both
    conditions at once leaves θ free: v12 = v sinθ, vS = v cosθ, v1 = v12/2,
    v2 = √3 v12/2.
    """
    import math
    v_ew = 246.0
    for theta in (0.3, 0.8, 1.0286, 1.4):
        v12 = v_ew * math.sin(theta)
        v1, v2, vS = v12 / 2, v12 * math.sqrt(3) / 2, v_ew * math.cos(theta)
        assert math.isclose(v2 / v1, math.sqrt(3), rel_tol=1e-12)     # alignment
        assert math.isclose(math.sqrt(v1**2 + v2**2 + vS**2), v_ew,
                            rel_tol=1e-12)                            # Eq. (8)

    # the tutorial's aligned point demonstrably misses the scale
    v2_t, vS_t = 115.0, 80.0
    v1_t = v2_t / math.sqrt(3)
    assert not math.isclose(math.sqrt(v1_t**2 + v2_t**2 + vS_t**2), v_ew,
                            rel_tol=1e-3)
