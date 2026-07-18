"""SM × U(1)_X with a Z′ boson, end to end with feynlag.

The full Standard Model (Higgs + electroweak + lepton + quark/QCD sectors,
as in examples/sm_scalar_gauge.py) extended by an extra abelian gauge factor
U(1)_X with coupling g_X.  The X-charges are the general family-universal
anomaly-free assignment — a linear combination of the two anomaly-free
abelian charges the SM fermion content admits (once three right-handed
neutrinos are added):

    X = a·Y + b·(B−L)

    q_H = a/2          q_L  = −a/2 − b     q_Q  = a/6 + b/3
    q_S = free          q_eR = −a − b       q_uR = 2a/3 + b/3
                        q_νR = −b           q_dR = −a/3 + b/3

Because the Higgs carries X-charge a/2, the Z and Z′ mix at tree level.
New states and the pinned physics (validated in tests/test_u1x.py):

- an SM-singlet complex scalar S (X-charge q_S, VEV v_X) breaks U(1)_X;
  its CP-odd part is the Goldstone eaten by the Z′ (both pseudoscalars come
  out exactly massless: two Goldstones for two massive neutral bosons);
- Higgs-portal coupling λ_HS|H|²|S|² mixes h and s with
      tan 2θ_s = λ_HS v v_X / (λ v² − λ_S v_X²)   [(h, s) ordering];
- the 3×3 neutral gauge mass matrix over (W3, B, X) is diagonalized by TWO
  chained 2×2 rotations — Weinberg first, then Z–Z′ with
      tan 2θ′ = 2a g_Z g_X v² / (a²g_X²v² + 4g_X²q_S²v_X² − g_Z²v²),
  m_Z²·m_Z′² = g_Z²g_X²q_S²v²v_X²/4;  a → 0 is the pure B−L limit (θ′ → 0);
- three right-handed neutrinos (required for the B−L anomaly to cancel)
  with a Dirac Yukawa via H̃, giving M_ν = Y_ν v/√2;
- Z′ff couplings = −sinθ′·g_Z(T³ − Q sin²θ_W) + cosθ′·g_X·q_f; the photon
  couplings are exactly the SM ones (the second rotation never touches A).

The X-charges are kept SYMBOLIC (a, b, q_S) — every pipeline stage handles
that (U(1) generators, invariance checking, gauge currents, extraction);
the one exception in the library is UFO export, whose particle table calls
float() on charges, so no UFO directory is written here.

Run:  python examples/sm_u1x.py
"""

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, Dmu, ExternalParameter, InternalParameter,
    Lagrangian, Model, Rotation, SU2, SU3, Scalar, U1, WeylFermion,
    conjugate_pair, dag, diagonalize_orthogonal_2x2, diracPL, diracPR,
    extract_fermion_vertices, fermion_feynman_rule, fermion_gauge_current,
    fermion_mass_matrix, rotation_2x2,
)


def main():
    # --- symmetries and parameters -------------------------------------
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    gX = ExternalParameter("gX", 0.5, positive=True)
    gs = ExternalParameter("gs", 1.22, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    U1X = U1("U1X", coupling=gX)
    SU3c = SU3("SU3c", coupling=gs)

    # the two X-charge parameters (symbolic everywhere) and the singlet charge
    aX = ExternalParameter("aX", 0.4)
    bX = ExternalParameter("bX", -0.7)
    qS = ExternalParameter("qS", 1.0)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    vX = ExternalParameter("vX", 2000.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    lamS = ExternalParameter("lamS", 0.2)
    lamHS = ExternalParameter("lamHS", 0.05)
    mu2 = InternalParameter("mu2", unit_dim=2)
    muS2 = InternalParameter("muS2", unit_dim=2)

    def qcharge(Y, BmL):
        """The anomaly-free charge map X = a·Y + b·(B−L)."""
        return aX.s * Y + bX.s * BmL

    # --- fields -----------------------------------------------------------
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2),
                          U1X: qcharge(sp.Rational(1, 2), 0)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})

    # SM-singlet complex scalar: breaks U(1)_X at v_X, gives the Z′ its mass
    S = Scalar("S", reps={U1X: qS.s}, component_names=["S0"])
    S.expand_vev({S.components[0]: vX})

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2),
                                 U1X: qcharge(-sp.Rational(1, 2), -1)},
                     chirality="L", nflavors=3, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1, U1X: qcharge(-1, -1)},
                     chirality="R", nflavors=3, component_names=["eR"])
    # right-handed neutrinos: Y = 0, B−L = −1 — they carry ONLY the X charge
    # (needed for the B−L gauge anomaly of the SM fermion content to cancel)
    nuR = WeylFermion("nuR", reps={U1X: qcharge(0, -1)},
                      chirality="R", nflavors=3, component_names=["nuR"])

    QL = WeylFermion("QL", reps={SU2L: 2, U1Y: sp.Rational(1, 6), SU3c: 3,
                                 U1X: qcharge(sp.Rational(1, 6),
                                              sp.Rational(1, 3))},
                     chirality="L", nflavors=3,
                     component_names=["uL_1", "uL_2", "uL_3",
                                      "dL_1", "dL_2", "dL_3"])
    uR = WeylFermion("uR", reps={U1Y: sp.Rational(2, 3), SU3c: 3,
                                 U1X: qcharge(sp.Rational(2, 3),
                                              sp.Rational(1, 3))},
                     chirality="R", nflavors=3,
                     component_names=["uR_1", "uR_2", "uR_3"])
    dR = WeylFermion("dR", reps={U1Y: -sp.Rational(1, 3), SU3c: 3,
                                 U1X: qcharge(-sp.Rational(1, 3),
                                              sp.Rational(1, 3))},
                     chirality="R", nflavors=3,
                     component_names=["dR_1", "dR_2", "dR_3"])

    W, B = SU2L.bosons("W"), U1Y.bosons("B")
    X = U1X.bosons("X")
    G = SU3c.bosons("G")

    # --- Lagrangian (user-written, FeynRules style) ------------------------
    HdH = (dag(H) * H.mat)[0]
    SdS = (dag(S) * S.mat)[0]
    V = (-mu2.s * HdH + lam.s * HdH**2
         - muS2.s * SdS + lamS.s * SdS**2
         + lamHS.s * HdH * SdS)
    DH = Dmu(H)      # includes W, B AND X — the Higgs carries all three
    DS = Dmu(S)      # X only

    i, j = sp.symbols("i j", integer=True)
    Ye, Yv = sp.IndexedBase("Ye"), sp.IndexedBase("Yv")
    Gp, H0 = H.components
    nuL, eL = Ll.components
    nuLbar, eLbar = Ll.bar_components
    eRc, eRbar = eR.components[0], eR.bar_components[0]
    nuRc, nuRbar = nuR.components[0], nuR.bar_components[0]

    # charged-lepton Yukawa: -Ye[i,j] Lbar_i H e_R[j] + h.c.
    LYuk = -(Ye[i, j] * Gp * Bilinear(nuLbar[i], diracPR, eRc[j])
             + Ye[i, j] * H0 * Bilinear(eLbar[i], diracPR, eRc[j]))
    LYuk += -(sp.conjugate(Ye[i, j]) * sp.conjugate(Gp)
              * Bilinear(eRbar[j], diracPL, nuL[i])
              + sp.conjugate(Ye[i, j]) * sp.conjugate(H0)
              * Bilinear(eRbar[j], diracPL, eL[i]))

    # neutrino Yukawa via the conjugate doublet H̃ = (H0*, −Gp*), the same
    # inline pattern the up-quark Yukawa uses: -Yv[i,j] Lbar_i H̃ nu_R[j] + h.c.
    LYukN = -(Yv[i, j] * sp.conjugate(H0)
              * Bilinear(nuLbar[i], diracPR, nuRc[j])
              + Yv[i, j] * (-sp.conjugate(Gp))
              * Bilinear(eLbar[i], diracPR, nuRc[j]))
    LYukN += -(sp.conjugate(Yv[i, j]) * H0
               * Bilinear(nuRbar[j], diracPL, nuL[i])
               + sp.conjugate(Yv[i, j]) * (-Gp)
               * Bilinear(nuRbar[j], diracPL, eL[i]))

    # quark Yukawas: verbatim from sm_scalar_gauge.py (color summed in
    # Python; flavor-generic, undiagonalized — no CKM here; the CKM
    # insertion route is in examples/sm_ckm.py)
    Yu, Yd = sp.IndexedBase("Yu"), sp.IndexedBase("Yd")
    qL_u, qL_d = QL.components[:3], QL.components[3:]
    qLbar_u, qLbar_d = QL.bar_components[:3], QL.bar_components[3:]
    uRc, uRbar = uR.components, uR.bar_components
    dRc, dRbar = dR.components, dR.bar_components

    def qcolor(bar_list, gamma, field_list, idx1, idx2):
        return sum(Bilinear(bar_list[c][idx1], gamma, field_list[c][idx2])
                   for c in range(3))

    LYuk_d = -(Yd[i, j] * Gp * qcolor(qLbar_u, diracPR, dRc, i, j)
               + Yd[i, j] * H0 * qcolor(qLbar_d, diracPR, dRc, i, j))
    LYuk_d += -(sp.conjugate(Yd[i, j]) * sp.conjugate(Gp)
                * qcolor(dRbar, diracPL, qL_u, j, i)
                + sp.conjugate(Yd[i, j]) * sp.conjugate(H0)
                * qcolor(dRbar, diracPL, qL_d, j, i))

    LYuk_u = -(Yu[i, j] * sp.conjugate(H0) * qcolor(qLbar_u, diracPR, uRc, i, j)
               + Yu[i, j] * (-sp.conjugate(Gp))
               * qcolor(qLbar_d, diracPR, uRc, i, j))
    LYuk_u += -(sp.conjugate(Yu[i, j]) * H0
                * qcolor(uRbar, diracPL, qL_u, j, i)
                + sp.conjugate(Yu[i, j]) * (-Gp)
                * qcolor(uRbar, diracPL, qL_d, j, i))

    # gauge currents — Dmu/fermion_gauge_current pick up the X boson
    # automatically for every field with U1X in its reps
    current = (fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i)
               + fermion_gauge_current(nuR, i))
    current_quarks = (fermion_gauge_current(QL, i)
                      + fermion_gauge_current(uR, i)
                      + fermion_gauge_current(dR, i))

    L = Lagrangian()
    L.add((dag(DH) * DH)[0] + (dag(DS) * DS)[0], sector="kinetic")
    L.add(-V, sector="potential")
    L.add(LYuk + LYukN, sector="yukawa")
    L.add(current, sector="yukawa")
    L.add(LYuk_d + LYuk_u, sector="yukawa")
    L.add(current_quarks, sector="yukawa")

    model = Model("SM-U1X", gauge_groups=[SU2L, U1Y, U1X, SU3c],
                  fields=[H, S, Ll, eR, nuR, QL, uR, dR, W, B, X, G],
                  parameters=[gw, g1, gX, gs, aX, bX, qS, v, vX,
                              lam, lamS, lamHS, mu2, muS2], lagrangian=L)

    # --- pipeline -----------------------------------------------------------
    report = model.check_invariance()
    print("invariance:", report)
    report.raise_on_failure()
    print("tadpoles:  ", model.solve_tadpoles([mu2, muS2]))

    # --- scalar sector: portal h–s mixing, Goldstone pseudoscalars ----------
    h0, G0 = sp.Symbol("H0_r", real=True), sp.Symbol("H0_i", real=True)
    s0, aX0 = sp.Symbol("S0_r", real=True), sp.Symbol("S0_i", real=True)

    Meven = model.mass_matrix([h0, s0])
    print("\nCP-even scalar M² (h0, s0):")
    sp.pprint(Meven)
    Modd = model.mass_matrix([G0, aX0])
    assert Modd == sp.zeros(2, 2), Modd
    print("CP-odd (G0, S0_i) block ≡ 0 — two Goldstones, eaten by Z and Z′")

    h1, h2 = sp.symbols("h1 h2", real=True)
    thetas = sp.Symbol("thetas", real=True)
    rot_s = diagonalize_orthogonal_2x2(Meven, [h0, s0], [h1, h2],
                                       angle=thetas)
    print("scalar mixing:", rot_s.angle_relation)
    model.rotate(rot_s)

    # --- gauge sector: 3×3 neutral block, chained Weinberg + Z–Z′ rotations -
    W1, W2, W3 = W.components
    B0 = B.components[0]
    X0 = X.components[0]

    M3 = model.gauge_mass_matrix([W3, B0, X0])
    print("\nneutral gauge M² (W3, B, X):")
    sp.pprint(M3)
    assert sp.simplify(M3.det()) == 0     # exactly one massless state

    # rotation 1 — Weinberg, exactly as in the SM: leaves the massless
    # photon and an intermediate Z0 that still mixes with X
    Z0, A, Z, Zp = sp.symbols("Z0 A Z Zp", real=True)
    thetaW = sp.atan(g1.s / gw.s)
    model.rotate(Rotation([W3, B0], [Z0, A], rotation_2x2(-thetaW)))

    cw, sw = sp.cos(thetaW), sp.sin(thetaW)
    R1e = sp.Matrix([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    M3r = sp.simplify(R1e * M3 * R1e.T)          # rows/cols: (Z0, A, X)
    assert M3r[1, :] == sp.zeros(1, 3)           # photon decoupled exactly

    # rotation 2 — Z–Z′ mixing of the surviving (Z0, X) block.  Chained
    # rotations work because physical_lagrangian applies them sequentially
    # in registration order: the second xreplace consumes the Z0 the first
    # one produced.
    blockZX = M3r.extract([0, 2], [0, 2])
    print("\n(Z0, X) mass block:")
    sp.pprint(blockZX)
    thetap = sp.Symbol("thetap", real=True)
    rot_zzp = diagonalize_orthogonal_2x2(blockZX, [Z0, X0], [Z, Zp],
                                         angle=thetap)
    print("Z–Z′ mixing:", rot_zzp.angle_relation)
    model.rotate(rot_zzp)

    # dual verification (CONVENTIONS.md): exact check with the explicit
    # atan angle, and physical masses at the benchmark point
    check_rot = Rotation([Z0, X0], [Z, Zp],
                         rotation_2x2(rot_zzp.angle_solution))
    ok, residuals = check_rot.check(blockZX)
    assert ok, residuals
    bench = {gw.s: 0.6535, g1.s: 0.3580, gX.s: 0.5, aX.s: 0.4,
             qS.s: 1.0, v.s: 246.0, vX.s: 2000.0}
    mZ2, mZp2 = [complex(m.subs(bench)).real
                 for m in check_rot.masses_squared(blockZX.subs(bench),
                                                   simplifier=sp.nsimplify)]
    print(f"benchmark: m_Z = {mZ2**0.5:.2f} GeV, m_Z' = {mZp2**0.5:.2f} GeV")

    Wp, Wm = sp.symbols("Wp Wm")
    Umix = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], Umix, kind="unitary"))

    # --- bosonic Feynman rules -----------------------------------------------
    Gm, cmap = conjugate_pair(Gp, "Gm")
    boson_fields = [h1, h2, G0, aX0, Gp, Gm, Z, A, Zp, Wp, Wm]

    rules = model.feynman_rules(boson_fields, sector="kinetic",
                                conjugate_map=cmap, simplifier=sp.simplify)
    print(f"\n{len(rules)} kinetic-sector vertices; Z′ highlights:")
    for key in [(h1, Zp, Zp), (h1, Z, Zp), (h1, Z, Z)]:
        skey = tuple(sorted(key, key=lambda f: f.sort_key()))
        if skey in rules:
            print(" ", skey, "->", rules[skey])

    # --- neutrino Dirac mass ---------------------------------------------------
    M_nu = fermion_mass_matrix(LYukN, nuLbar, nuRc, model.vacuum, 3, (i, j),
                               gamma=diracPR)
    print("\nDirac neutrino mass matrix M_ν =")
    sp.pprint(M_nu)

    # --- fermion Feynman rules (Z′, shifted Z, unchanged photon, portal h) --
    LYuk_phys = model.physical_lagrangian(sector="yukawa").xreplace(cmap)
    table = extract_fermion_vertices(LYuk_phys, boson_fields)

    mu = sp.Symbol("mu", integer=True)
    gammaL = DiracGamma(mu) * diracPL
    gammaR = DiracGamma(mu) * diracPR

    def show(name, bar, gamma, field, boson):
        entry = table.get((bar, gamma, field), {})
        coeff = entry.get(1, {}).get((boson,), sp.S.Zero)
        rule = sp.simplify(fermion_feynman_rule(coeff, gamma, (boson,)))
        print(f"  {name:12s} ->", rule)

    print("\nZ′ couplings (−sinθ′·g_Z(T³−Q s_w²) + cosθ′·g_X·q_f):")
    show("Z' eL eL", eLbar[i], gammaL, eL[i], Zp)
    show("Z' eR eR", eRbar[i], gammaR, eRc[i], Zp)
    show("Z' nuL nuL", nuLbar[i], gammaL, nuL[i], Zp)
    show("Z' nuR nuR", nuRbar[i], gammaR, nuRc[i], Zp)
    show("Z' uL uL", qLbar_u[0][i], gammaL, qL_u[0][i], Zp)

    print("\nmodified Z couplings (SM recovered at θ′ → 0):")
    show("Z eR eR", eRbar[i], gammaR, eRc[i], Z)
    show("Z nuR nuR", nuRbar[i], gammaR, nuRc[i], Z)

    print("\nphoton couplings (exactly SM — A never mixes with X):")
    show("A eL eL", eLbar[i], gammaL, eL[i], A)
    show("A nuR nuR", nuRbar[i], gammaR, nuRc[i], A)

    print("\nportal-mixed Higgs couplings (cosθ_s/sinθ_s split):")
    show("h1 eL eR", eLbar[i], diracPR, eRc[j], h1)
    show("h2 eL eR", eLbar[i], diracPR, eRc[j], h2)
    show("h1 nuL nuR", nuLbar[i], diracPR, nuRc[j], h1)


if __name__ == "__main__":
    main()
