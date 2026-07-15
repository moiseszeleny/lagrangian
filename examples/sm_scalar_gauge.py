"""SM Higgs + electroweak gauge sector + lepton sector with feynlag, end to end.

Run:  python examples/sm_scalar_gauge.py
"""

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, Dmu, ExternalParameter, InternalParameter,
    Lagrangian, Model, Rotation, SU2, Scalar, U1, WeylFermion, conjugate_pair,
    dag, diracPL, diracPR, extract_fermion_vertices, fermion_feynman_rule,
    fermion_gauge_current, fermion_mass_matrix, latex_feynman_table,
    rotation_2x2,
)


def main():
    # --- symmetries and parameters -------------------------------------
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)

    # --- fields -----------------------------------------------------------
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=3, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                     component_names=["eR"])
    W, B = SU2L.bosons("W"), U1Y.bosons("B")

    # --- Lagrangian (user-written, FeynRules style) ------------------------
    HdH = (dag(H) * H.mat)[0]
    V = -mu2.s * HdH + lam.s * HdH**2
    DH = Dmu(H)

    i, j = sp.symbols("i j", integer=True)
    Ye = sp.IndexedBase("Ye")
    Gp, H0 = H.components
    nuLbar, eLbar = Ll.bar_components
    eRc, eRbar = eR.components[0], eR.bar_components[0]

    # L_Yuk = - Ye[i,j] Lbar_i H e_R[j] + h.c.  (P_R sandwich; h.c. carries P_L)
    LYuk = -(Ye[i, j] * Gp * Bilinear(nuLbar[i], diracPR, eRc[j])
             + Ye[i, j] * H0 * Bilinear(eLbar[i], diracPR, eRc[j]))
    LYuk += -(sp.conjugate(Ye[i, j]) * sp.conjugate(Gp)
              * Bilinear(eRbar[j], diracPL, Ll.components[0][i])
              + sp.conjugate(Ye[i, j]) * sp.conjugate(H0)
              * Bilinear(eRbar[j], diracPL, Ll.components[1][i]))

    # i psibar gamma^mu D_mu psi, interaction part: g A^a psibar gamma^mu T^a P psi
    current = fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i)

    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-V, sector="potential")
    L.add(LYuk, sector="yukawa")
    L.add(current, sector="yukawa")

    model = Model("SM-EW", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, W, B],
                  parameters=[gw, g1, v, lam, mu2], lagrangian=L)

    # --- pipeline -----------------------------------------------------------
    print("invariance:", model.check_invariance())
    print("tadpole:   ", model.solve_tadpoles([mu2]))

    h, G0 = sp.Symbol("H0_r", real=True), sp.Symbol("H0_i", real=True)
    print("m_h² =", model.mass_matrix([h])[0, 0])

    # gauge masses and Weinberg rotation
    W1, W2, W3 = W.components
    B0 = B.components[0]
    print("gauge M² =")
    sp.pprint(model.gauge_mass_matrix([W1, W2, W3, B0]))

    Z, A = sp.symbols("Z A", real=True)
    thetaW = sp.atan(g1.s / gw.s)
    model.rotate(Rotation([W3, B0], [Z, A], rotation_2x2(-thetaW)))
    Wp, Wm = sp.symbols("Wp Wm")
    Umix = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], Umix, kind="unitary"))

    # --- scalar Feynman rules -----------------------------------------------
    Gm, cmap = conjugate_pair(Gp, "Gm")
    boson_fields = [h, G0, Gp, Gm, Z, A, Wp, Wm]

    rules = model.feynman_rules(boson_fields, conjugate_map=cmap,
                                simplifier=sp.simplify)
    print(f"\n{len(rules)} scalar/gauge vertices; a few highlights:")
    for key in list(rules)[:6]:
        print(" ", key, "->", rules[key])

    # --- fermion mass matrix -------------------------------------------------
    M_e = fermion_mass_matrix(LYuk, eLbar, eRc, model.vacuum, 3, (i, j),
                              gamma=diracPR)
    print("\nlepton mass matrix M_e =")
    sp.pprint(M_e)

    # --- fermion Feynman rules (Yukawa + gauge currents, physical basis) ---
    # physical_lagrangian applies the vacuum shift (H0 -> (v+h+iG0)/√2), the
    # tadpole solution, and the registered Z/A/W± rotations to EVERY sector,
    # including "yukawa" — the h/G0-lepton couplings and the gauge currents
    # come out of the same extraction pass.
    LYuk_phys = model.physical_lagrangian(sector="yukawa").xreplace(cmap)
    table = extract_fermion_vertices(LYuk_phys, boson_fields)

    mu = sp.Symbol("mu", integer=True)
    gammaL = DiracGamma(mu) * diracPL
    gammaR = DiracGamma(mu) * diracPR
    nuL, eL = Ll.components

    def show(name, bar, gamma, field, boson):
        key = (bar, gamma, field)
        coeff = table[key][1][(boson,)]
        rule = sp.simplify(fermion_feynman_rule(coeff, gamma, (boson,)))
        print(f"  {name:10s} ->", rule)

    # note: the Yukawa term carries TWO flavor indices (Ye[i,j]) since it
    # connects the doublet's flavor i to eR's flavor j; the gauge currents
    # above are flavor-diagonal (same index on both legs) instead.
    print("\nh/G0-lepton Yukawa vertices:")
    show("h eL eR", eLbar[i], diracPR, eRc[j], h)

    m_ell = sp.Symbol("m_ell", positive=True)
    hll_coeff = table[(eLbar[i], diracPR, eRc[j])][1][(h,)]
    hll_rule = fermion_feynman_rule(hll_coeff, diracPR, (h,))
    hll_pinned = hll_rule.subs(Ye[i, j], sp.sqrt(2) * m_ell / v.s)
    print("  single flavor, m_ell = Ye v/√2 ->", sp.simplify(hll_pinned),
         " (+ h.c.)")

    print("\ngauge-current vertices:")
    show("W+ nu e", nuLbar[i], gammaL, eL[i], Wp)
    show("Z eL eL", eLbar[i], gammaL, eL[i], Z)
    show("Z eR eR", eRbar[i], gammaR, eRc[i], Z)
    show("A eL eL", eLbar[i], gammaL, eL[i], A)
    show("A eR eR", eRbar[i], gammaR, eRc[i], A)

    # --- LaTeX preview --------------------------------------------------------
    table_preview = {k: v for k, v in list(rules.items())[:5]}
    latex = latex_feynman_table(table_preview)
    print("\nLaTeX preview:\n", latex[:400], "...")


if __name__ == "__main__":
    main()
