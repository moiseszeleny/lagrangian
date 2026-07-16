"""SM + vector-like lepton doublet: pinned physics validation.

The model (see examples/sm_vll.py): one SM lepton generation plus a
vector-like doublet Ψ=(N,E) with BOTH chiralities in the SM lepton doublet's
reps, a gauge-invariant bare mass −M Ψ̄_L Ψ_R, and a mixing Yukawa
−λ_E Ψ̄_L H e_R.  Pinned here:

- mass matrix [[y_e v/√2, 0], [λ_E v/√2, M]]; ν massless; m_N = M exactly
- analytic biunitary angles (diagonalize_svd_2x2 / solve_mixing_angle_2x2)
- THE headline: the Z FCNC lives only in the right-handed current
  (LH exactly zero — e_L and E_L are both T³=−1/2 doublet members), with
  RH FCNC = −(g_Z/2) sinθ_R cosθ_R and RH diagonal shift −sin²θ_R/2
- photon couplings untouched (U(1)_EM unbroken)
- RH charged current W⁺ N̄_R e_R ∝ sinθ_R (absent in the SM)
- h-coupling sum rule: coeff(h ē₁e₁) = −(m_e − sinθ_L sinθ_R M)/v
- SM decoupling at λ_E → 0

Everything is asserted symbolically AND (where a formula is pinned) checked
at random numeric points via numeric_equal, per CONVENTIONS.md.
"""

import sympy as sp
import pytest

from feynlag import (
    Bilinear, DiracGamma, ExternalParameter, InternalParameter, Lagrangian,
    Model, Rotation, SU2, Scalar, U1, WeylFermion, dag, Dmu,
    diagonalize_svd_2x2, diracPL, diracPR, extract_fermion_vertices,
    fermion_gauge_current, fermion_mass_matrix, numeric_equal, rotation_2x2,
    solve_mixing_angle_2x2,
)

i, j = sp.symbols("fl_i fl_j", integer=True)


@pytest.fixture(scope="module")
def vll():
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)
    ye = ExternalParameter("ye", 0.01, positive=True)
    lamE = ExternalParameter("lamE", 0.4, positive=True)
    MPsi = ExternalParameter("MPsi", 1000.0, positive=True, unit_dim=1)

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eRf = WeylFermion("eRf", reps={U1Y: -1}, chirality="R", nflavors=1,
                      component_names=["eR"])
    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                       chirality="L", nflavors=1, component_names=["NL", "EL"])
    PsiR = WeylFermion("PsiR", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                       chirality="R", nflavors=1, component_names=["NR", "ER"])
    W, B = SU2L.bosons("W"), U1Y.bosons("B")

    Gp, H0 = H.components
    nuL, eL = Ll.components
    nuLbar, eLbar = Ll.bar_components
    eR, eRbar = eRf.components[0], eRf.bar_components[0]
    NL, EL = PsiL.components
    NLbar, ELbar = PsiL.bar_components
    NR, ER = PsiR.components
    NRbar, ERbar = PsiR.bar_components

    LYuk = -(ye.s * Gp * Bilinear(nuLbar[i], diracPR, eR[i])
             + ye.s * H0 * Bilinear(eLbar[i], diracPR, eR[i]))
    LYuk += -(ye.s * sp.conjugate(Gp) * Bilinear(eRbar[i], diracPL, nuL[i])
              + ye.s * sp.conjugate(H0) * Bilinear(eRbar[i], diracPL, eL[i]))
    LYuk += -(lamE.s * Gp * Bilinear(NLbar[i], diracPR, eR[i])
              + lamE.s * H0 * Bilinear(ELbar[i], diracPR, eR[i]))
    LYuk += -(lamE.s * sp.conjugate(Gp) * Bilinear(eRbar[i], diracPL, NL[i])
              + lamE.s * sp.conjugate(H0)
              * Bilinear(eRbar[i], diracPL, EL[i]))
    LYuk += -MPsi.s * (Bilinear(NLbar[i], diracPR, NR[i])
                       + Bilinear(ELbar[i], diracPR, ER[i]))
    LYuk += -MPsi.s * (Bilinear(NRbar[i], diracPL, NL[i])
                       + Bilinear(ERbar[i], diracPL, EL[i]))

    current = (fermion_gauge_current(Ll, i) + fermion_gauge_current(eRf, i)
               + fermion_gauge_current(PsiL, i)
               + fermion_gauge_current(PsiR, i))

    HdH = (dag(H) * H.mat)[0]
    V = -mu2.s * HdH + lam.s * HdH**2
    DH = Dmu(H)

    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-V, sector="potential")
    L.add(LYuk, sector="yukawa")
    L.add(current, sector="yukawa")

    model = Model("SM-VLL", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eRf, PsiL, PsiR, W, B],
                  parameters=[gw, g1, v, lam, mu2, ye, lamE, MPsi],
                  lagrangian=L)
    model.solve_tadpoles([mu2])

    W1, W2, W3 = W.components
    B0 = B.components[0]
    Z, A = sp.symbols("Z A", real=True)
    thetaW = sp.atan(g1.s / gw.s)
    model.rotate(Rotation([W3, B0], [Z, A], rotation_2x2(-thetaW)))
    Wp, Wm = sp.symbols("Wp Wm")
    Umix = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], Umix, kind="unitary"))

    bars, rights = (eLbar, ELbar), (eR, ER)
    M2 = sp.Matrix(2, 2, lambda a, b: fermion_mass_matrix(
        LYuk, bars[a], rights[b], model.vacuum, 1, (i, j),
        gamma=diracPR)[0, 0])

    thL, thR = sp.symbols("thL thR", real=True)
    e1L, e2L = sp.IndexedBase("e1L"), sp.IndexedBase("e2L")
    e1R, e2R = sp.IndexedBase("e1R"), sp.IndexedBase("e2R")
    e1Lbar, e2Lbar = sp.IndexedBase("e1Lbar"), sp.IndexedBase("e2Lbar")
    e1Rbar, e2Rbar = sp.IndexedBase("e1Rbar"), sp.IndexedBase("e2Rbar")

    rotL, rotR = diagonalize_svd_2x2(M2, [eL[i], EL[i]], [eR[i], ER[i]],
                                     [e1L[i], e2L[i]], [e1R[i], e2R[i]],
                                     angle_left=thL, angle_right=thR)
    model.rotate(rotL)
    model.rotate(Rotation([eLbar[i], ELbar[i]], [e1Lbar[i], e2Lbar[i]],
                          rotation_2x2(thL)))
    model.rotate(rotR)
    model.rotate(Rotation([eRbar[i], ERbar[i]], [e1Rbar[i], e2Rbar[i]],
                          rotation_2x2(thR)))

    h = sp.Symbol("H0_r", real=True)
    table = extract_fermion_vertices(
        model.physical_lagrangian(sector="yukawa"), [Z, A, Wp, Wm, h])

    s = dict(model=model, LYuk=LYuk, M2=M2, table=table,
             rotL=rotL, rotR=rotR, thL=thL, thR=thR,
             g=gw.s, gp=g1.s, v=v.s, ye=ye.s, lamE=lamE.s, M=MPsi.s,
             Z=Z, A=A, Wp=Wp, h=h,
             nuLbar=nuLbar, NLbar=NLbar, NRbar=NRbar, NR=NR,
             e1L=e1L, e2L=e2L, e1R=e1R, e2R=e2R,
             e1Lbar=e1Lbar, e2Lbar=e2Lbar, e1Rbar=e1Rbar, e2Rbar=e2Rbar)
    return s


def _coeff(s, bar, gamma, field, boson):
    entry = s["table"].get((bar, gamma, field))
    if entry is None:
        return sp.S.Zero
    return entry.get(1, {}).get((boson,), sp.S.Zero)


_mu = sp.Symbol("mu", integer=True)
gammaL = DiracGamma(_mu) * diracPL
gammaR = DiracGamma(_mu) * diracPR


def test_invariance(vll):
    """Gauge invariance + hermiticity + mass dimension of the bare
    vector-like mass and the cross-Fermion mixing Yukawa — the first model
    exercising Bilinear hermiticity across different Fermion objects."""
    report = vll["model"].check_invariance()
    assert report.ok, report.failures


def test_mass_matrix(vll):
    ye, lamE, v, M = vll["ye"], vll["lamE"], vll["v"], vll["M"]
    expected = sp.Matrix([[ye * v / sp.sqrt(2), 0],
                          [lamE * v / sp.sqrt(2), M]])
    assert sp.simplify(vll["M2"] - expected) == sp.zeros(2, 2)

    # N is a pure Dirac state of mass M; the neutrino has no mass partner
    model, LYuk = vll["model"], vll["LYuk"]
    m_N = fermion_mass_matrix(LYuk, vll["NLbar"], vll["NR"], model.vacuum,
                              1, (i, j), gamma=diracPR)[0, 0]
    assert sp.simplify(m_N - M) == 0
    m_nu = fermion_mass_matrix(LYuk, vll["nuLbar"], vll["NR"], model.vacuum,
                               1, (i, j), gamma=diracPR)[0, 0]
    assert m_nu == 0


def test_mixing_angles(vll):
    """tan 2θ from solve_mixing_angle_2x2 on M·Mᵀ (θ_L) and Mᵀ·M (θ_R):
    θ_L is v²/M²-suppressed, θ_R ∝ λ_E v/M is the doublet signature."""
    ye, lamE, v, M = vll["ye"], vll["lamE"], vll["v"], vll["M"]
    M2 = vll["M2"]

    _, tan2L = solve_mixing_angle_2x2(M2 * M2.T)
    expected_L = -2 * lamE * ye * v**2 / (2 * M**2 + lamE**2 * v**2
                                          - ye**2 * v**2)
    assert sp.simplify(tan2L - expected_L) == 0
    assert numeric_equal(tan2L, expected_L, [ye, lamE, v, M])

    _, tan2R = solve_mixing_angle_2x2(M2.T * M2)
    expected_R = 2 * sp.sqrt(2) * M * lamE * v / (-2 * M**2
                                                  + lamE**2 * v**2
                                                  + ye**2 * v**2)
    assert sp.simplify(tan2R - expected_R) == 0
    assert numeric_equal(tan2R, expected_R, [ye, lamE, v, M])


def test_svd_diagonalizes(vll):
    """Dual verification of the biunitary rotation + basis-independent
    invariants: Σm² = tr(M·Mᵀ) and m_e·m_E = |det M| = y_e v M/√2."""
    M2, rotL, rotR = vll["M2"], vll["rotL"], vll["rotR"]
    thL, thR = vll["thL"], vll["thR"]
    ye, lamE, v, M = vll["ye"], vll["lamE"], vll["v"], vll["M"]

    sub = {thL: rotL.angle_solution, thR: rotR.angle_solution}
    D = (rotL.matrix * M2 * rotR.matrix.T).subs(sub)
    for off in (D[0, 1], D[1, 0]):
        assert numeric_equal(off, sp.S.Zero, [ye, lamE, v, M])

    # invariants: symbolic simplify can't crack the nested atan forms —
    # numeric random-point equality is the designed fallback (checks.py)
    assert numeric_equal(D[0, 0]**2 + D[1, 1]**2,
                         sp.trace(M2 * M2.T), [ye, lamE, v, M])
    assert numeric_equal(sp.Abs(D[0, 0] * D[1, 1]),
                         ye * v * M / sp.sqrt(2), [ye, lamE, v, M])

    # light-first ordering at the benchmark (M >> v regime)
    bench = {ye: 0.01, lamE: 0.4, v: 246.0, M: 1000.0}
    Dn = D.subs(bench)
    assert abs(Dn[0, 0]) < abs(Dn[1, 1])


def test_z_couplings(vll):
    """THE pinned physics: Z FCNC only in the right-handed current."""
    g, gp, thR = vll["g"], vll["gp"], vll["thR"]
    Z = vll["Z"]
    gZ = sp.sqrt(g**2 + gp**2)
    sw2 = gp**2 / (g**2 + gp**2)

    # LH light diagonal: the SM value, unchanged by the rotation
    zL11 = _coeff(vll, vll["e1Lbar"][i], gammaL, vll["e1L"][i], Z)
    assert sp.simplify(zL11 - gZ * (-sp.Rational(1, 2) + sw2)) == 0

    # LH FCNC: EXACT zero (e_L and E_L share T³=-1/2 — protection)
    zL12 = _coeff(vll, vll["e1Lbar"][i], gammaL, vll["e2L"][i], Z)
    assert sp.simplify(zL12) == 0

    # RH light diagonal: shifted by the doublet admixture -sin²θ_R/2
    zR11 = _coeff(vll, vll["e1Rbar"][i], gammaR, vll["e1R"][i], Z)
    expected = gZ * (sw2 - sp.sin(thR)**2 / 2)
    assert sp.simplify(zR11 - expected) == 0
    assert numeric_equal(zR11, expected, [g, gp, thR])

    # RH FCNC: -(g_Z/2) sinθ_R cosθ_R
    zR12 = _coeff(vll, vll["e1Rbar"][i], gammaR, vll["e2R"][i], Z)
    expected = -gZ / 2 * sp.sin(thR) * sp.cos(thR)
    assert sp.simplify(zR12 - expected) == 0
    assert numeric_equal(zR12, expected, [g, gp, thR])

    # RH heavy diagonal
    zR22 = _coeff(vll, vll["e2Rbar"][i], gammaR, vll["e2R"][i], Z)
    expected = gZ * (-sp.Rational(1, 2) + sw2 + sp.sin(thR)**2 / 2)
    assert sp.simplify(zR22 - expected) == 0

    # photon: diagonal -e in both eigenstates, FCNC exactly zero
    A = vll["A"]
    e_em = g * gp / sp.sqrt(g**2 + gp**2)
    for bar, field in (((vll["e1Lbar"]), vll["e1L"]),
                       ((vll["e2Lbar"]), vll["e2L"])):
        aQ = _coeff(vll, bar[i], gammaL, field[i], A)
        assert sp.simplify(aQ + e_em) == 0
    aF = _coeff(vll, vll["e1Lbar"][i], gammaL, vll["e2L"][i], A)
    assert sp.simplify(aF) == 0
    aFR = _coeff(vll, vll["e1Rbar"][i], gammaR, vll["e2R"][i], A)
    assert sp.simplify(aFR) == 0


def test_w_couplings(vll):
    """LH current split by cosθ_L/sinθ_L; the RH charged current
    W⁺ N̄_R e_R ∝ sinθ_R is the genuinely new doublet-VLL effect."""
    g, thL, thR, Wp = vll["g"], vll["thL"], vll["thR"], vll["Wp"]

    w1 = _coeff(vll, vll["nuLbar"][i], gammaL, vll["e1L"][i], Wp)
    assert sp.simplify(w1 - g / sp.sqrt(2) * sp.cos(thL)) == 0
    w2 = _coeff(vll, vll["nuLbar"][i], gammaL, vll["e2L"][i], Wp)
    assert sp.simplify(w2 + g / sp.sqrt(2) * sp.sin(thL)) == 0

    wR1 = _coeff(vll, vll["NRbar"][i], gammaR, vll["e1R"][i], Wp)
    assert sp.simplify(wR1 - g / sp.sqrt(2) * sp.sin(thR)) == 0
    wR2 = _coeff(vll, vll["NRbar"][i], gammaR, vll["e2R"][i], Wp)
    assert sp.simplify(wR2 - g / sp.sqrt(2) * sp.cos(thR)) == 0

    wL1 = _coeff(vll, vll["NLbar"][i], gammaL, vll["e1L"][i], Wp)
    assert sp.simplify(wL1 - g / sp.sqrt(2) * sp.sin(thL)) == 0


def test_higgs_couplings(vll):
    """h-coupling sum rule: coeff(h ē₁ e₁, P_R) = −(m_e − s_L s_R M)/v with
    m_e = (c_L y_e + s_L λ_E)(v/√2)c_R + s_L s_R M — the classic VLF
    deviation of the light lepton's Higgs coupling from −m_e/v."""
    thL, thR, h = vll["thL"], vll["thR"], vll["h"]
    ye, lamE, v, M = vll["ye"], vll["lamE"], vll["v"], vll["M"]
    cL, sL = sp.cos(thL), sp.sin(thL)
    cR, sR = sp.cos(thR), sp.sin(thR)

    h11 = _coeff(vll, vll["e1Lbar"][i], diracPR, vll["e1R"][i], h)
    m_e = (cL * ye + sL * lamE) * v / sp.sqrt(2) * cR + sL * sR * M
    expected = -(m_e - sL * sR * M) / v
    assert sp.simplify(h11 - expected) == 0
    assert numeric_equal(h11, expected, [ye, lamE, v, M, thL, thR])

    # FCNC h couplings, both P_R "directions", pinned exactly
    h12 = _coeff(vll, vll["e1Lbar"][i], diracPR, vll["e2R"][i], h)
    assert sp.simplify(h12 - (cL * ye + sL * lamE) * sR / sp.sqrt(2)) == 0
    h21 = _coeff(vll, vll["e2Lbar"][i], diracPR, vll["e1R"][i], h)
    assert sp.simplify(h21 - (sL * ye - cL * lamE) * cR / sp.sqrt(2)) == 0


def test_sm_decoupling(vll):
    """λ_E → 0: angles → 0, all FCNC → 0, h coupling → −m_e/v = −y_e/√2·…,
    masses → (y_e v/√2, M)."""
    thL, thR = vll["thL"], vll["thR"]
    lamE, ye, v, M = vll["lamE"], vll["ye"], vll["v"], vll["M"]
    rotL, rotR = vll["rotL"], vll["rotR"]

    assert rotL.angle_solution.subs(lamE, 0) == 0
    assert rotR.angle_solution.subs(lamE, 0) == 0

    decouple = {sp.sin(thL): 0, sp.cos(thL): 1,
                sp.sin(thR): 0, sp.cos(thR): 1, lamE: 0}

    zR12 = _coeff(vll, vll["e1Rbar"][i], gammaR, vll["e2R"][i], vll["Z"])
    assert sp.simplify(zR12.subs(decouple)) == 0
    h12 = _coeff(vll, vll["e1Lbar"][i], diracPR, vll["e2R"][i], vll["h"])
    assert sp.simplify(h12.subs(decouple)) == 0

    h11 = _coeff(vll, vll["e1Lbar"][i], diracPR, vll["e1R"][i], vll["h"])
    assert sp.simplify(h11.subs(decouple) + ye / sp.sqrt(2)) == 0

    M2_dec = vll["M2"].subs(lamE, 0)
    assert M2_dec == sp.Matrix([[ye * v / sp.sqrt(2), 0], [0, M]])
