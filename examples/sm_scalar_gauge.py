"""SM Higgs + electroweak gauge sector + lepton + quark/QCD sector with
feynlag, end to end.

Run:  python examples/sm_scalar_gauge.py
"""

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, Dmu, ExternalParameter, InternalParameter,
    Lagrangian, Model, Rotation, SU2, SU3, Scalar, U1, WeylFermion,
    conjugate_pair, cubic_couplings, dag, diracPL, diracPR,
    extract_fermion_vertices, fermion_feynman_rule, fermion_gauge_current,
    fermion_mass_matrix, latex_feynman_table, quartic_couplings, rotation_2x2,
)
from feynlag.export.ufo.vvvv import assemble_vvvv


def main():
    # --- symmetries and parameters -------------------------------------
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    gs = ExternalParameter("gs", 1.22, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    SU3c = SU3("SU3c", coupling=gs)

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

    # quark sector: SU(2) doublet Q_L + color-triplet singlets u_R, d_R.
    # flavor structure mirrors the lepton sector exactly — 3 generic,
    # flavor-indexed generations, undiagonalized Yukawas (Yu[i,j]/Yd[i,j]),
    # no CKM: CKM mixing is physically orthogonal to SU(3) vertex dynamics
    # (gluon vertices are flavor-diagonal/color-universal regardless of it),
    # and a fully generic 3-generation complex-Yukawa SVD doesn't resolve in
    # closed symbolic form via diagonalize_svd — deferred, not attempted here.
    QL = WeylFermion("QL", reps={SU2L: 2, U1Y: sp.Rational(1, 6), SU3c: 3},
                     chirality="L", nflavors=3,
                     component_names=["uL_1", "uL_2", "uL_3",
                                      "dL_1", "dL_2", "dL_3"])
    uR = WeylFermion("uR", reps={U1Y: sp.Rational(2, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=["uR_1", "uR_2", "uR_3"])
    dR = WeylFermion("dR", reps={U1Y: -sp.Rational(1, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=["dR_1", "dR_2", "dR_3"])

    W, B = SU2L.bosons("W"), U1Y.bosons("B")
    G = SU3c.bosons("G")

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

    # --- quark Yukawas: same L̄·H·(e_R-like)-field pattern as the lepton
    # sector, with an explicit color sum (the color index isn't a symbolic
    # IndexedBase axis here — it's baked into which Kronecker-product
    # component of QL/uR/dR you pick, so it's summed in Python, not left as
    # a free index like the flavor i,j).
    Yu, Yd = sp.IndexedBase("Yu"), sp.IndexedBase("Yd")
    qL_u, qL_d = QL.components[:3], QL.components[3:]
    qLbar_u, qLbar_d = QL.bar_components[:3], QL.bar_components[3:]
    uRc, uRbar = uR.components, uR.bar_components
    dRc, dRbar = dR.components, dR.bar_components

    def qcolor(bar_list, gamma, field_list, idx1, idx2):
        return sum(Bilinear(bar_list[c][idx1], gamma, field_list[c][idx2])
                   for c in range(3))

    # down-type: L_Yuk_d = -Yd[i,j] Qbar_i . H d_R[j] + h.c.
    LYuk_d = -(Yd[i, j] * Gp * qcolor(qLbar_u, diracPR, dRc, i, j)
               + Yd[i, j] * H0 * qcolor(qLbar_d, diracPR, dRc, i, j))
    LYuk_d += -(sp.conjugate(Yd[i, j]) * sp.conjugate(Gp)
                * qcolor(dRbar, diracPL, qL_u, j, i)
                + sp.conjugate(Yd[i, j]) * sp.conjugate(H0)
                * qcolor(dRbar, diracPL, qL_d, j, i))

    # up-type: uses the conjugate ("tilde") Higgs doublet H̃=(H0*, -Gp*);
    # no Htilde Field/abstraction exists (or is needed) anywhere in the
    # codebase — built inline exactly like the lepton h.c. terms already do.
    LYuk_u = -(Yu[i, j] * sp.conjugate(H0) * qcolor(qLbar_u, diracPR, uRc, i, j)
               + Yu[i, j] * (-sp.conjugate(Gp))
               * qcolor(qLbar_d, diracPR, uRc, i, j))
    LYuk_u += -(sp.conjugate(Yu[i, j]) * H0
                * qcolor(uRbar, diracPL, qL_u, j, i)
                + sp.conjugate(Yu[i, j]) * (-Gp)
                * qcolor(uRbar, diracPL, qL_d, j, i))

    # i psibar gamma^mu D_mu psi, interaction part: g A^a psibar gamma^mu T^a P psi
    current = fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i)
    current_quarks = (fermion_gauge_current(QL, i)
                      + fermion_gauge_current(uR, i)
                      + fermion_gauge_current(dR, i))

    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-V, sector="potential")
    L.add(LYuk, sector="yukawa")
    L.add(current, sector="yukawa")
    L.add(LYuk_d + LYuk_u, sector="yukawa")
    L.add(current_quarks, sector="yukawa")

    model = Model("SM-EW+QCD", gauge_groups=[SU2L, U1Y, SU3c],
                  fields=[H, Ll, eR, QL, uR, dR, W, B, G],
                  parameters=[gw, g1, gs, v, lam, mu2], lagrangian=L)

    # --- pipeline -----------------------------------------------------------
    # validate() is the umbrella: symmetry/hermiticity/dimension + gauge-anomaly
    # cancellation in one report (the SM fermion content below is anomaly-free).
    print("validation:")
    print(model.validate().summary())
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

    # quark mass matrices: color-diagonal, so any single color slot's
    # (bar_base, field_base) pair recovers the full flavor structure — no
    # diagonalization/CKM here (see the quark-sector comment above).
    M_u = fermion_mass_matrix(LYuk_u, qLbar_u[0], uRc[0], model.vacuum, 3,
                              (i, j), gamma=diracPR)
    M_d = fermion_mass_matrix(LYuk_d, qLbar_d[0], dRc[0], model.vacuum, 3,
                              (i, j), gamma=diracPR)
    print("\nup-quark mass matrix M_u =")
    sp.pprint(M_u)
    print("\ndown-quark mass matrix M_d =")
    sp.pprint(M_d)

    # --- QCD self-coupling: internal verification only, not UFO input ------
    # (gluons are ONE UFO particle "g" with the color-adjoint index summed
    # via a color-tensor string, not 8 separate weak-basis components — see
    # export/ufo/writer.py for the color-export side of this).
    G1, G2, G3 = G.components[0], G.components[1], G.components[2]
    ggg = cubic_couplings(SU3c)[(G1, G2, G3)]
    print("\nggg coupling (pinned, f^123=1) g_1 g_2 g_3 ->", ggg,
         " == -gs ?", sp.simplify(ggg + gs.s) == 0)

    G4, G5 = G.components[3], G.components[4]
    gggg = assemble_vvvv(quartic_couplings(SU3c), (G1, G2, G4, G5))
    print("gggg coupling (g_1 g_2 g_4 g_5, VVVV1/2/3) ->", gggg)

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
