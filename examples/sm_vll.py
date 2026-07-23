"""SM + vector-like lepton doublet (VLL) with feynlag, end to end.

One SM lepton generation plus a vector-like doublet Ψ = (N, E) whose two
chiralities BOTH transform as (SU(2)_L doublet, Y = −1/2) — the same reps as
the SM lepton doublet.  That makes the bare Dirac mass −M Ψ̄_L Ψ_R gauge
invariant (dimension 3; M carries unit_dim=1 for the power-counting check),
and the mixing Yukawa −λ_E Ψ̄_L H e_R ties the heavy doublet to the SM
singlet e_R.  After EWSB the charged-lepton mass matrix

    rows (ē_L, Ē_L) × cols (e_R, E_R):  [[y_e v/√2, 0], [λ_E v/√2, M]]

is diagonalized biunitarily (θ_L from M·Mᵀ, θ_R from Mᵀ·M — the analytic
diagonalize_svd_2x2 route; the generic symbolic SVD returns unusable nested
radicals).  Headline physics, all extracted below and pinned in
tests/test_vll.py:

- θ_R is the LARGE angle (∝ λ_E v/M) — the doublet-VLL signature; θ_L is
  v²/M²-suppressed.
- The Z FCNC lives ONLY in the right-handed current: e_L and E_L are both
  T³ = −1/2 doublet members, so the LH Z coupling commutes with the θ_L
  rotation (exact zero FCNC), while e_R (T³=0) and E_R (T³=−1/2) differ,
  giving Z ē₁_R e₂_R = −(g_Z/2) sinθ_R cosθ_R and a −sin²θ_R/2 shift of the
  light lepton's RH Z coupling.
- h ē e deviates from −m_e/v by sinθ_L sinθ_R M/v (h-coupling sum rule).
- A right-handed W current appears: W⁺ N̄_R e_R ∝ sinθ_R (absent in the SM).

Conventions: no bare L̄_L Ψ_R mixing mass — it can always be removed by a
field redefinition of the two identically-charged left doublets.  The VLL is
written as TWO WeylFermions with identical reps rather than one DiracFermion
(the chirality-None gauge-current path is untested — see fields.py).

Run:  python examples/sm_vll.py
"""

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, ExternalParameter, Lagrangian, Model, Rotation, SU2,
    U1, WeylFermion, charged_current_rotation, diagonalize_svd_2x2, diracPL,
    diracPR, electroweak_scaffold, extract_fermion_vertices,
    fermion_gauge_current, fermion_mass_matrix, rotation_2x2,
    weinberg_rotation,
)


def main():
    # --- symmetries and parameters -------------------------------------
    # the SM electroweak scaffold (gauge groups, Higgs, potential); this
    # example's subject is the vector-like lepton added below.
    ew = electroweak_scaffold(lam=0.129)
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    gw, g1, v, lam, mu2 = ew.gw, ew.g1, ew.v, ew.lam, ew.mu2
    ye = ExternalParameter("ye", 0.01, positive=True)
    lamE = ExternalParameter("lamE", 0.4, positive=True)
    MPsi = ExternalParameter("MPsi", 1000.0, positive=True, unit_dim=1)

    # --- fields -----------------------------------------------------------
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eRf = WeylFermion("eRf", reps={U1Y: -1}, chirality="R", nflavors=1,
                      component_names=["eR"])
    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                       chirality="L", nflavors=1, component_names=["NL", "EL"])
    PsiR = WeylFermion("PsiR", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                       chirality="R", nflavors=1, component_names=["NR", "ER"])
    W, B = ew.W, ew.B

    Gp, H0 = H.components
    nuL, eL = Ll.components
    nuLbar, eLbar = Ll.bar_components
    eR, eRbar = eRf.components[0], eRf.bar_components[0]
    NL, EL = PsiL.components
    NLbar, ELbar = PsiL.bar_components
    NR, ER = PsiR.components
    NRbar, ERbar = PsiR.bar_components

    i = sp.Symbol("fl_i", integer=True)

    # --- Lagrangian ---------------------------------------------------------
    # SM Yukawa: -ye Lbar_L H e_R + h.c.
    LYuk = -(ye.s * Gp * Bilinear(nuLbar[i], diracPR, eR[i])
             + ye.s * H0 * Bilinear(eLbar[i], diracPR, eR[i]))
    LYuk += -(ye.s * sp.conjugate(Gp) * Bilinear(eRbar[i], diracPL, nuL[i])
              + ye.s * sp.conjugate(H0) * Bilinear(eRbar[i], diracPL, eL[i]))

    # mixing Yukawa: -lamE Psibar_L H e_R + h.c.  (Psi_L carries L_L's reps,
    # so the component pattern is identical to the SM Yukawa's)
    LYuk += -(lamE.s * Gp * Bilinear(NLbar[i], diracPR, eR[i])
              + lamE.s * H0 * Bilinear(ELbar[i], diracPR, eR[i]))
    LYuk += -(lamE.s * sp.conjugate(Gp) * Bilinear(eRbar[i], diracPL, NL[i])
              + lamE.s * sp.conjugate(H0) * Bilinear(eRbar[i], diracPL, EL[i]))

    # bare vector-like mass: -MPsi Psibar_L Psi_R + h.c. (gauge invariant —
    # both chiralities share the same reps; dim 3 with unit_dim(MPsi)=1)
    LYuk += -MPsi.s * (Bilinear(NLbar[i], diracPR, NR[i])
                       + Bilinear(ELbar[i], diracPR, ER[i]))
    LYuk += -MPsi.s * (Bilinear(NRbar[i], diracPL, NL[i])
                       + Bilinear(ERbar[i], diracPL, EL[i]))

    current = (fermion_gauge_current(Ll, i) + fermion_gauge_current(eRf, i)
               + fermion_gauge_current(PsiL, i)
               + fermion_gauge_current(PsiR, i))

    L = Lagrangian()
    ew.add_higgs(L)                                    # kinetic + potential
    L.add(LYuk, sector="yukawa")
    L.add(current, sector="yukawa")

    model = Model("SM-VLL", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Ll, eRf, PsiL, PsiR],
                  parameters=ew.parameters + [ye, lamE, MPsi],
                  lagrangian=L)

    # --- pipeline -----------------------------------------------------------
    report = model.check_invariance()
    print("invariance:", report)
    report.raise_on_failure()
    print("tadpole:   ", model.solve_tadpoles([mu2]))

    # gauge rotations (Weinberg + W±) from feynlag.models
    Z, A = weinberg_rotation(model, SU2L, U1Y)
    Wp, Wm = charged_current_rotation(model, SU2L)

    # --- charged-lepton mass matrix ------------------------------------------
    j = sp.Symbol("fl_j", integer=True)
    bars, rights = (eLbar, ELbar), (eR, ER)
    M2 = sp.Matrix(2, 2, lambda a, b: fermion_mass_matrix(
        LYuk, bars[a], rights[b], model.vacuum, 1, (i, j),
        gamma=diracPR)[0, 0])
    print("\ncharged-lepton mass matrix (rows eLbar, ELbar; cols eR, ER):")
    sp.pprint(M2)

    m_N = fermion_mass_matrix(LYuk, NLbar, NR, model.vacuum, 1, (i, j),
                              gamma=diracPR)[0, 0]
    print("m_N =", m_N, " (exactly MPsi; the neutrino stays massless —")
    print("        no bilinear pairs nuL with any right-handed state)")

    # --- biunitary diagonalization -------------------------------------------
    thL, thR = sp.symbols("thL thR", real=True)
    e1L, e2L = sp.IndexedBase("e1L"), sp.IndexedBase("e2L")
    e1R, e2R = sp.IndexedBase("e1R"), sp.IndexedBase("e2R")
    e1Lbar, e2Lbar = sp.IndexedBase("e1Lbar"), sp.IndexedBase("e2Lbar")
    e1Rbar, e2Rbar = sp.IndexedBase("e1Rbar"), sp.IndexedBase("e2Rbar")

    rotL, rotR = diagonalize_svd_2x2(M2, [eL[i], EL[i]], [eR[i], ER[i]],
                                     [e1L[i], e2L[i]], [e1R[i], e2R[i]],
                                     angle_left=thL, angle_right=thR)
    print("\nθ_L (v²/M²-suppressed):", rotL.angle_relation)
    print("θ_R (the large doublet-mixing angle):", rotR.angle_relation)

    # dual verification of the diagonalization (CONVENTIONS.md): numeric at
    # the benchmark; the exact-zero off-diagonals are pinned in test_vll.py
    bench = {ye.s: 0.01, lamE.s: 0.4, v.s: 246.0, MPsi.s: 1000.0}
    thL_num = float(rotL.angle_solution.subs(bench))
    thR_num = float(rotR.angle_solution.subs(bench))
    D = rotation_2x2(thL_num) * M2.subs(bench) * rotation_2x2(thR_num).T
    assert abs(D[0, 1]) < 1e-9 and abs(D[1, 0]) < 1e-9, D
    assert abs(D[0, 0]) < abs(D[1, 1])   # new field 1 = light state
    print(f"benchmark: θ_L={thL_num:.3e}, θ_R={thR_num:.3e}, "
          f"m_e={abs(D[0, 0]):.4f}, m_E={abs(D[1, 1]):.2f} GeV")

    # register the fermion rotations: per chirality, field-side AND bar-side
    # with the same real matrix (Rotation carries Indexed old fields — the
    # substitution reaches every Bilinear slot because all terms share the
    # same flavor index i; expand_bilinear then distributes the sums)
    model.rotate(rotL)
    model.rotate(Rotation([eLbar[i], ELbar[i]], [e1Lbar[i], e2Lbar[i]],
                          rotation_2x2(thL)))
    model.rotate(rotR)
    model.rotate(Rotation([eRbar[i], ERbar[i]], [e1Rbar[i], e2Rbar[i]],
                          rotation_2x2(thR)))

    # --- physical couplings ---------------------------------------------------
    h = sp.Symbol("H0_r", real=True)
    LYuk_phys = model.physical_lagrangian(sector="yukawa")
    table = extract_fermion_vertices(LYuk_phys, [Z, A, Wp, Wm, h])

    mu = sp.Symbol("mu", integer=True)
    gammaL = DiracGamma(mu) * diracPL
    gammaR = DiracGamma(mu) * diracPR

    def coeff(bar, gamma, field, boson):
        entry = table.get((bar, gamma, field))
        if entry is None:
            return sp.S.Zero
        return entry.get(1, {}).get((boson,), sp.S.Zero)

    def show(name, expr):
        print(f"  {name:22s} ->", sp.simplify(expr))

    print("\nZ couplings (LH protected, FCNC only in the RH current):")
    show("Z e1L e1L (unchanged)", coeff(e1Lbar[i], gammaL, e1L[i], Z))
    show("Z e1L e2L (LH FCNC)", coeff(e1Lbar[i], gammaL, e2L[i], Z))
    show("Z e1R e1R (shifted)", coeff(e1Rbar[i], gammaR, e1R[i], Z))
    show("Z e1R e2R (RH FCNC)", coeff(e1Rbar[i], gammaR, e2R[i], Z))

    print("\nphoton couplings (U(1)_EM unbroken — no FCNC, charge -e):")
    show("A e1L e1L", coeff(e1Lbar[i], gammaL, e1L[i], A))
    show("A e1L e2L", coeff(e1Lbar[i], gammaL, e2L[i], A))

    print("\nW couplings (RH charged current = new doublet-VLL effect):")
    show("W+ nu e1", coeff(nuLbar[i], gammaL, e1L[i], Wp))
    show("W+ nu e2", coeff(nuLbar[i], gammaL, e2L[i], Wp))
    show("W+ NR e1R (RH curr.)", coeff(NRbar[i], gammaR, e1R[i], Wp))
    show("W+ NR e2R", coeff(NRbar[i], gammaR, e2R[i], Wp))

    print("\nh couplings (deviation from -m_e/v = sinθ_L sinθ_R M/v):")
    show("h e1 e1 (PR)", coeff(e1Lbar[i], diracPR, e1R[i], h))
    show("h e1 e2 (PR, FCNC)", coeff(e1Lbar[i], diracPR, e2R[i], h))
    show("h e2 e1 (PR, FCNC)", coeff(e2Lbar[i], diracPR, e1R[i], h))

    # h-coupling sum rule: coeff(h e1 e1) = -(m_e - sinθ_L sinθ_R M)/v with
    # m_e = (cosθ_L ye + sinθ_L lamE)(v/√2)cosθ_R + sinθ_L sinθ_R M
    cL, sL = sp.cos(thL), sp.sin(thL)
    cR, sR = sp.cos(thR), sp.sin(thR)
    m_e_expr = (cL * ye.s + sL * lamE.s) * v.s / sp.sqrt(2) * cR \
        + sL * sR * MPsi.s
    residue = sp.simplify(coeff(e1Lbar[i], diracPR, e1R[i], h)
                          + (m_e_expr - sL * sR * MPsi.s) / v.s)
    print("h-coupling sum rule residue (must be 0):", residue)


if __name__ == "__main__":
    main()
