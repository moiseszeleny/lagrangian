"""Softly-broken Z2 2HDM scalar sector with feynlag.

Run:  python examples/thdm.py
"""

import sympy as sp

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, SU2, Scalar,
    U1, ZN, dag, diagonalize_orthogonal_2x2,
)


def main():
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v1 = ExternalParameter("v1", 246 * 0.95, positive=True, unit_dim=1)
    v2 = ExternalParameter("v2", 246 * 0.31, positive=True, unit_dim=1)
    m12sq = ExternalParameter("m12sq", 2000.0, unit_dim=2)
    lams = [ExternalParameter(f"lam{i}", 0.1 * i) for i in range(1, 6)]
    m11sq = InternalParameter("m11sq", unit_dim=2)
    m22sq = InternalParameter("m22sq", unit_dim=2)

    H1 = Scalar("H1", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                component_names=["H1p", "H10"])
    H2 = Scalar("H2", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                component_names=["H2p", "H20"])
    H1.expand_vev({H1.components[1]: v1})
    H2.expand_vev({H2.components[1]: v2})

    # soft Z2 (bookkeeping: broken only by m12²)
    Z2 = ZN("Z2", 2)
    Z2.assign(0, H1)
    Z2.assign(1, H2)

    l1, l2, l3, l4, l5 = (p.s for p in lams)
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
                  parameters=[gw, g1, v1, v2, m12sq, m11sq, m22sq, *lams],
                  lagrangian=L)

    print("invariance:", model.check_invariance())
    print("tadpoles solved for m11², m22²:")
    for k, s in model.solve_tadpoles([m11sq, m22sq]).items():
        print("  ", k, "=", s)

    # CP-even mass matrix and the alpha rotation
    h1r, h2r = sp.Symbol("H10_r", real=True), sp.Symbol("H20_r", real=True)
    M = model.mass_matrix([h1r, h2r])
    print("\nCP-even M²:")
    sp.pprint(M)

    Hh, hl = sp.symbols("Hheavy hlight", real=True)
    alpha = sp.Symbol("alpha", real=True)
    rot = diagonalize_orthogonal_2x2(M, [h1r, h2r], [Hh, hl], angle=alpha)
    print("\ntan(2α) =", sp.simplify(rot.angle_relation.rhs))
    model.rotate(rot)

    rules = model.feynman_rules([Hh, hl], simplifier=sp.simplify)
    print(f"\n{len(rules)} neutral-CP-even vertices, e.g.:")
    for key in list(rules)[:4]:
        print(" ", key, "->", sp.simplify(rules[key]))


if __name__ == "__main__":
    main()
