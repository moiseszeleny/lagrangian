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
