"""Phase 2 validation: SM Higgs sector.

Pinned physics (CONVENTIONS.md / standard results):
- tadpole:  mu2 = lam * v**2
- m_h^2  =  2 lam v**2
- hhh vertex = -3 i m_h^2 / v
- hhhh vertex = -3 i m_h^2 / v**2
- Goldstones massless; h G0 G0 vertex = -i m_h^2 / v; h Gp Gm = -i m_h^2 / v
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, SU2, Scalar, U1,
    conjugate_pair, dag,
)


@pytest.fixture
def sm():
    """SM Higgs-sector model, VEV registered, nothing solved yet."""
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})

    HdH = (dag(H) * H.mat)[0]
    V = -mu2.s * HdH + lam.s * HdH**2

    L = Lagrangian().add(-V, sector="potential")
    model = Model("SM-Higgs", gauge_groups=[SU2L, U1Y], fields=[H],
                  parameters=[gw, g1, v, lam, mu2], lagrangian=L)
    return model, H, v, lam, mu2


def test_invariance(sm):
    model, H, v, lam, mu2 = sm
    assert model.check_invariance().ok


def test_tadpole_solution(sm):
    model, H, v, lam, mu2 = sm
    sol = model.solve_tadpoles([mu2])
    assert sp.simplify(sol[mu2.s] - lam.s * v.s**2) == 0
    # solution registered on the InternalParameter
    assert sp.simplify(mu2.expr - lam.s * v.s**2) == 0


def test_higgs_and_goldstone_masses(sm):
    model, H, v, lam, mu2 = sm
    model.solve_tadpoles([mu2])
    h, G0 = sp.Symbol("H0_r", real=True), sp.Symbol("H0_i", real=True)

    M2 = model.mass_matrix([h, G0])
    assert sp.simplify(M2[0, 0] - 2 * lam.s * v.s**2) == 0   # m_h^2 = 2 lam v^2
    assert M2[0, 1] == 0 and M2[1, 0] == 0
    assert sp.simplify(M2[1, 1]) == 0                        # G0 massless

    # charged Goldstone massless
    Gp = H.components[0]
    M2c = model.mass_matrix([Gp], charged=True)
    assert sp.simplify(M2c[0, 0]) == 0


def test_higgs_self_couplings_pinned(sm):
    """The Phase 2 pinned validation: h^3 and h^4 vertices."""
    model, H, v, lam, mu2 = sm
    model.solve_tadpoles([mu2])
    h, G0 = sp.Symbol("H0_r", real=True), sp.Symbol("H0_i", real=True)
    Gp = H.components[0]
    Gm, cmap = conjugate_pair(Gp, "Gm")

    rules = model.feynman_rules([h, G0, Gp, Gm], conjugate_map=cmap,
                                simplifier=sp.simplify)
    mh2 = 2 * lam.s * v.s**2

    def rule(*fields):
        key = tuple(sorted(fields, key=lambda s: s.sort_key()))
        return rules[key]

    assert sp.simplify(rule(h, h, h) + 3 * sp.I * mh2 / v.s) == 0
    assert sp.simplify(rule(h, h, h, h) + 3 * sp.I * mh2 / v.s**2) == 0
    assert sp.simplify(rule(h, G0, G0) + sp.I * mh2 / v.s) == 0
    assert sp.simplify(rule(h, Gp, Gm) + sp.I * mh2 / v.s) == 0


def test_no_two_point_interactions_returned(sm):
    model, H, v, lam, mu2 = sm
    model.solve_tadpoles([mu2])
    h = sp.Symbol("H0_r", real=True)
    table = model.interactions([h])
    assert all(n >= 3 for n in table)


def test_lazy_pipeline_no_state_leak(sm):
    """physical_lagrangian is cached but invalidated by state changes."""
    model, H, v, lam, mu2 = sm
    L_before = model.physical_lagrangian()
    assert mu2.s in L_before.free_symbols
    model.solve_tadpoles([mu2])
    L_after = model.physical_lagrangian()
    assert mu2.s not in L_after.free_symbols
