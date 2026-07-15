"""Phase 2 validation: softly-broken Z2 2HDM scalar sector.

Reference expressions (Gunion–Haber / Branco et al. review, real potential):

V = m11² H1†H1 + m22² H2†H2 − m12² (H1†H2 + h.c.)
    + λ1/2 (H1†H1)² + λ2/2 (H2†H2)² + λ3 (H1†H1)(H2†H2)
    + λ4 |H1†H2|² + λ5/2 [(H1†H2)² + h.c.]

with ⟨H_i⁰⟩ = v_i/√2 and λ₃₄₅ = λ3+λ4+λ5:

- tadpoles:  m11² = m12² v2/v1 − λ1 v1²/2 − λ₃₄₅ v2²/2   (and 1↔2)
- CP-even:   M11 = m12² v2/v1 + λ1 v1²,  M12 = −m12² + λ₃₄₅ v1 v2,
             M22 = m12² v1/v2 + λ2 v2²
- CP-odd:    M_odd = (m12²/(v1v2) − λ5) [[v2², −v1v2], [−v1v2, v1²]]
- charged:   M_ch  = (m12²/(v1v2) − (λ4+λ5)/2) [[v2², −v1v2], [−v1v2, v1²]]
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, SU2, Scalar, U1,
    ZN, dag, diagonalize_orthogonal_2x2, numeric_equal,
)


@pytest.fixture(scope="module")
def thdm():
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v1 = ExternalParameter("v1", 246.0 * 0.95, positive=True, unit_dim=1)
    v2 = ExternalParameter("v2", 246.0 * 0.31, positive=True, unit_dim=1)
    lams = {i: ExternalParameter(f"lam{i}", 0.1 * i) for i in range(1, 6)}
    m12sq = ExternalParameter("m12sq", 2000.0, unit_dim=2)
    m11sq = InternalParameter("m11sq", unit_dim=2)
    m22sq = InternalParameter("m22sq", unit_dim=2)

    H1 = Scalar("H1", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                component_names=["H1p", "H10"])
    H2 = Scalar("H2", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                component_names=["H2p", "H20"])
    H1.expand_vev({H1.components[1]: v1})
    H2.expand_vev({H2.components[1]: v2})

    # soft Z2: H1 -> H1, H2 -> -H2 broken only by the m12² term
    l1, l2, l3, l4, l5 = (lams[i].s for i in range(1, 6))
    H1dH1 = (dag(H1) * H1.mat)[0]
    H2dH2 = (dag(H2) * H2.mat)[0]
    H1dH2 = (dag(H1) * H2.mat)[0]

    V = (m11sq.s * H1dH1 + m22sq.s * H2dH2
         - m12sq.s * (H1dH2 + sp.conjugate(H1dH2))
         + l1 / 2 * H1dH1**2 + l2 / 2 * H2dH2**2
         + l3 * H1dH1 * H2dH2 + l4 * H1dH2 * sp.conjugate(H1dH2)
         + l5 / 2 * (H1dH2**2 + sp.conjugate(H1dH2)**2))

    L = Lagrangian().add(-V, sector="potential")
    model = Model("THDM", gauge_groups=[SU2L, U1Y], fields=[H1, H2],
                  parameters=[gw, g1, v1, v2, m12sq, m11sq, m22sq,
                              *lams.values()],
                  lagrangian=L)
    model.solve_tadpoles([m11sq, m22sq])
    params = dict(v1=v1.s, v2=v2.s, m12sq=m12sq.s,
                  **{f"lam{i}": lams[i].s for i in range(1, 6)})
    return model, H1, H2, params, (m11sq, m22sq)


def _sym(name):
    return sp.Symbol(name, real=True)


def test_gauge_invariance(thdm):
    model, *_ = thdm
    assert model.check_invariance().ok


def test_tadpole_solutions_match_literature(thdm):
    model, H1, H2, p, (m11sq, m22sq) = thdm
    v1, v2, m12sq = p["v1"], p["v2"], p["m12sq"]
    lam345 = p["lam3"] + p["lam4"] + p["lam5"]

    expected_m11 = m12sq * v2 / v1 - p["lam1"] * v1**2 / 2 - lam345 * v2**2 / 2
    expected_m22 = m12sq * v1 / v2 - p["lam2"] * v2**2 / 2 - lam345 * v1**2 / 2

    syms = list(p.values())
    for internal, expected in ((m11sq, expected_m11), (m22sq, expected_m22)):
        assert sp.simplify(internal.expr - expected) == 0
        ok, diff = numeric_equal(internal.expr, expected, syms, seed=1)
        assert ok, diff


def test_cp_even_mass_matrix(thdm):
    model, H1, H2, p, _ = thdm
    v1, v2, m12sq = p["v1"], p["v2"], p["m12sq"]
    lam345 = p["lam3"] + p["lam4"] + p["lam5"]

    M = model.mass_matrix([_sym("H10_r"), _sym("H20_r")])
    expected = sp.Matrix([
        [m12sq * v2 / v1 + p["lam1"] * v1**2, -m12sq + lam345 * v1 * v2],
        [-m12sq + lam345 * v1 * v2, m12sq * v1 / v2 + p["lam2"] * v2**2],
    ])
    syms = list(p.values())
    for i in range(2):
        for j in range(2):
            assert sp.simplify(M[i, j] - expected[i, j]) == 0, (i, j)
            ok, diff = numeric_equal(M[i, j], expected[i, j], syms, seed=2)
            assert ok, ((i, j), diff)


def test_cp_odd_mass_matrix_and_mA(thdm):
    model, H1, H2, p, _ = thdm
    v1, v2, m12sq, lam5 = p["v1"], p["v2"], p["m12sq"], p["lam5"]

    M = model.mass_matrix([_sym("H10_i"), _sym("H20_i")])
    pref = m12sq / (v1 * v2) - lam5
    expected = pref * sp.Matrix([[v2**2, -v1 * v2], [-v1 * v2, v1**2]])
    for i in range(2):
        for j in range(2):
            assert sp.simplify(M[i, j] - expected[i, j]) == 0, (i, j)

    # eigenvalues: one Goldstone (0) and m_A² = pref * (v1²+v2²)
    assert sp.simplify(M.det()) == 0
    assert sp.simplify(sp.trace(M) - pref * (v1**2 + v2**2)) == 0


def test_charged_mass_matrix_and_mHp(thdm):
    model, H1, H2, p, _ = thdm
    v1, v2, m12sq = p["v1"], p["v2"], p["m12sq"]
    lam4, lam5 = p["lam4"], p["lam5"]

    M = model.mass_matrix([H1.components[0], H2.components[0]], charged=True)
    pref = m12sq / (v1 * v2) - (lam4 + lam5) / 2
    expected = pref * sp.Matrix([[v2**2, -v1 * v2], [-v1 * v2, v1**2]])
    for i in range(2):
        for j in range(2):
            assert sp.simplify(M[i, j] - expected[i, j]) == 0, (i, j)
    assert sp.simplify(M.det()) == 0
    assert sp.simplify(sp.trace(M) - pref * (v1**2 + v2**2)) == 0


def test_alpha_rotation_diagonalizes_cp_even(thdm):
    """diagonalize_orthogonal_2x2 with symbolic angle: tan2α matches the
    literature and R M Rᵀ is numerically diagonal at a benchmark point."""
    model, H1, H2, p, _ = thdm
    M = model.mass_matrix([_sym("H10_r"), _sym("H20_r")])

    H_heavy, h_light = _sym("Hheavy"), _sym("hlight")
    alpha = sp.Symbol("alpha", real=True)
    rot = diagonalize_orthogonal_2x2(
        M, [_sym("H10_r"), _sym("H20_r")], [H_heavy, h_light], angle=alpha)

    # the tan(2α) defining relation (CONVENTIONS.md verification rule)
    expected_tan2a = 2 * M[0, 1] / (M[0, 0] - M[1, 1])
    assert sp.simplify(rot.angle_relation.rhs - expected_tan2a) == 0

    # substitute the explicit solution and check numeric diagonalization
    ok, residuals = rot.check(
        M, simplifier=lambda e: sp.nsimplify(
            sp.simplify(e.subs(alpha, rot.angle_solution)), rational=False))
    assert ok, residuals


def test_goldstone_beta_rotation(thdm):
    """The CP-odd block is diagonalized by the beta rotation
    (G0, A0) with tan(beta) = v2/v1; masses (0, m_A²)."""
    model, H1, H2, p, _ = thdm
    v1, v2 = p["v1"], p["v2"]
    M = model.mass_matrix([_sym("H10_i"), _sym("H20_i")])

    G0, A0 = _sym("G0"), _sym("A0")
    beta = sp.atan(v2 / v1)
    from feynlag import Rotation, rotation_2x2
    rot = Rotation([_sym("H10_i"), _sym("H20_i")], [G0, A0],
                   rotation_2x2(beta))
    ok, residuals = rot.check(M)
    assert ok, residuals
    masses = rot.masses_squared(M)
    assert sp.simplify(masses[0]) == 0  # Goldstone
    mA2 = (p["m12sq"] / (v1 * v2) - p["lam5"]) * (v1**2 + v2**2)
    assert sp.simplify(masses[1] - mA2) == 0
