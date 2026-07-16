"""SM × U(1)_X (Z′) model: pinned physics validation.

The model (see examples/sm_u1x.py): the full SM plus a gauged
X = a·Y + b·(B−L) — the general family-universal anomaly-free abelian
extension — with an X-charged Higgs (q_H = a/2, so Z–Z′ mix at tree level),
an SM-singlet complex scalar S (charge q_S, VEV v_X) breaking U(1)_X, a
Higgs-portal coupling λ_HS|H|²|S|², and three right-handed neutrinos
(q_νR = −b) with a Dirac Yukawa.  Pinned here:

- tadpoles: μ² = λv² + λ_HS v_X²/2, μ_S² = λ_S v_X² + λ_HS v²/2
- the 3×3 neutral gauge mass matrix over (W3, B, X); det ≡ 0
- Weinberg rotation decouples the photon EXACTLY (row/col ≡ 0), leaving the
  (Z0, X) block with tan 2θ′ = 2a g_Z g_X v²/(a²g_X²v² + 4g_X²q_S²v_X² − g_Z²v²)
- m_Z²·m_Z′² = g_Z²g_X²q_S²v²v_X²/4;  a → 0 is the unmixed B−L limit
- W± untouched: m_W² = g²v²/4, zero charged–neutral cross terms
- scalar portal: tan 2θ_s = λ_HS v v_X/(λv² − λ_S v_X²); BOTH pseudoscalars
  exactly massless (Goldstones of Z and Z′); charged Goldstone massless
- Z′ff couplings = −sinθ′·g_Z(T³ − Q s_w²) + cosθ′·g_X·q_f for all species;
  shifted Zff (SM at θ′ → 0); photon couplings exactly SM
- M_ν = Y_ν v/√2 (Dirac, 3×3 flavor-generic)
- h1/h2 fermion couplings split by cosθ_s/sinθ_s (SM at θ_s → 0)

Everything is asserted symbolically AND (where a closed formula is pinned)
checked at random numeric points via numeric_equal, per CONVENTIONS.md.
"""

import sympy as sp
import pytest

from feynlag import (
    Bilinear, DiracGamma, Dmu, ExternalParameter, InternalParameter,
    Lagrangian, Model, Rotation, SU2, SU3, Scalar, U1, WeylFermion, dag,
    diagonalize_orthogonal_2x2, diracPL, diracPR, extract_fermion_vertices,
    fermion_gauge_current, fermion_mass_matrix, numeric_equal, rotation_2x2,
    solve_mixing_angle_2x2,
)

i, j = sp.symbols("i j", integer=True)


@pytest.fixture(scope="module")
def u1x():
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    gX = ExternalParameter("gX", 0.5, positive=True)
    gs = ExternalParameter("gs", 1.22, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    U1X = U1("U1X", coupling=gX)
    SU3c = SU3("SU3c", coupling=gs)

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
        return aX.s * Y + bX.s * BmL

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2),
                          U1X: qcharge(sp.Rational(1, 2), 0)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    S = Scalar("S", reps={U1X: qS.s}, component_names=["S0"])
    S.expand_vev({S.components[0]: vX})

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2),
                                 U1X: qcharge(-sp.Rational(1, 2), -1)},
                     chirality="L", nflavors=3, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1, U1X: qcharge(-1, -1)},
                     chirality="R", nflavors=3, component_names=["eR"])
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

    HdH = (dag(H) * H.mat)[0]
    SdS = (dag(S) * S.mat)[0]
    V = (-mu2.s * HdH + lam.s * HdH**2 - muS2.s * SdS + lamS.s * SdS**2
         + lamHS.s * HdH * SdS)
    DH, DS = Dmu(H), Dmu(S)

    Ye, Yv = sp.IndexedBase("Ye"), sp.IndexedBase("Yv")
    Gp, H0 = H.components
    nuL, eL = Ll.components
    nuLbar, eLbar = Ll.bar_components
    eRc, eRbar = eR.components[0], eR.bar_components[0]
    nuRc, nuRbar = nuR.components[0], nuR.bar_components[0]

    LYuk = -(Ye[i, j] * Gp * Bilinear(nuLbar[i], diracPR, eRc[j])
             + Ye[i, j] * H0 * Bilinear(eLbar[i], diracPR, eRc[j]))
    LYuk += -(sp.conjugate(Ye[i, j]) * sp.conjugate(Gp)
              * Bilinear(eRbar[j], diracPL, nuL[i])
              + sp.conjugate(Ye[i, j]) * sp.conjugate(H0)
              * Bilinear(eRbar[j], diracPL, eL[i]))

    LYukN = -(Yv[i, j] * sp.conjugate(H0)
              * Bilinear(nuLbar[i], diracPR, nuRc[j])
              + Yv[i, j] * (-sp.conjugate(Gp))
              * Bilinear(eLbar[i], diracPR, nuRc[j]))
    LYukN += -(sp.conjugate(Yv[i, j]) * H0
               * Bilinear(nuRbar[j], diracPL, nuL[i])
               + sp.conjugate(Yv[i, j]) * (-Gp)
               * Bilinear(nuRbar[j], diracPL, eL[i]))

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
    LYuk_u = -(Yu[i, j] * sp.conjugate(H0)
               * qcolor(qLbar_u, diracPR, uRc, i, j)
               + Yu[i, j] * (-sp.conjugate(Gp))
               * qcolor(qLbar_d, diracPR, uRc, i, j))
    LYuk_u += -(sp.conjugate(Yu[i, j]) * H0
                * qcolor(uRbar, diracPL, qL_u, j, i)
                + sp.conjugate(Yu[i, j]) * (-Gp)
                * qcolor(uRbar, diracPL, qL_d, j, i))

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
    tad = model.solve_tadpoles([mu2, muS2])

    # scalar sector
    h0, G0 = sp.Symbol("H0_r", real=True), sp.Symbol("H0_i", real=True)
    s0, aX0 = sp.Symbol("S0_r", real=True), sp.Symbol("S0_i", real=True)
    Meven = model.mass_matrix([h0, s0])
    Modd = model.mass_matrix([G0, aX0])
    h1, h2 = sp.symbols("h1 h2", real=True)
    thetas = sp.Symbol("thetas", real=True)
    rot_s = diagonalize_orthogonal_2x2(Meven, [h0, s0], [h1, h2],
                                       angle=thetas)
    model.rotate(rot_s)

    # gauge sector: Weinberg first, then Z–Z′ on the surviving block
    W1, W2, W3 = W.components
    B0 = B.components[0]
    X0 = X.components[0]
    M3 = model.gauge_mass_matrix([W3, B0, X0])

    Z0, A, Z, Zp = sp.symbols("Z0 A Z Zp", real=True)
    thetaW = sp.atan(g1.s / gw.s)
    model.rotate(Rotation([W3, B0], [Z0, A], rotation_2x2(-thetaW)))

    cw, sw = sp.cos(thetaW), sp.sin(thetaW)
    R1e = sp.Matrix([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    M3r = sp.simplify(R1e * M3 * R1e.T)
    blockZX = M3r.extract([0, 2], [0, 2])
    thetap = sp.Symbol("thetap", real=True)
    rot_zzp = diagonalize_orthogonal_2x2(blockZX, [Z0, X0], [Z, Zp],
                                         angle=thetap)
    model.rotate(rot_zzp)

    Wp, Wm = sp.symbols("Wp Wm")
    Umix = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], Umix, kind="unitary"))

    table = extract_fermion_vertices(
        model.physical_lagrangian(sector="yukawa"),
        [Z, A, Zp, Wp, Wm, h1, h2])

    return dict(
        model=model, tad=tad, LYukN=LYukN,
        Meven=Meven, Modd=Modd, M3=M3, M3r=M3r, blockZX=blockZX,
        rot_s=rot_s, rot_zzp=rot_zzp, R1e=R1e,
        thetas=thetas, thetap=thetap, thetaW=thetaW,
        table=table, Yv=Yv, Ye=Ye,
        g=gw.s, gp=g1.s, gx=gX.s, a=aX.s, b=bX.s, qs=qS.s,
        v=v.s, vX=vX.s, lam=lam.s, lamS=lamS.s, lamHS=lamHS.s,
        mu2=mu2, muS2=muS2, Gp=Gp,
        W1=W1, W2=W2, W3=W3, B0=B0, X0=X0,
        Z=Z, A=A, Zp=Zp, Wp=Wp, h1=h1, h2=h2,
        nuL=nuL, eL=eL, nuLbar=nuLbar, eLbar=eLbar,
        eRc=eRc, eRbar=eRbar, nuRc=nuRc, nuRbar=nuRbar,
        qL_u=qL_u, qL_d=qL_d, qLbar_u=qLbar_u, qLbar_d=qLbar_d,
        uRc=uRc, uRbar=uRbar, dRc=dRc, dRbar=dRbar)


def _coeff(s, bar, gamma, field, boson):
    entry = s["table"].get((bar, gamma, field))
    if entry is None:
        return sp.S.Zero
    return entry.get(1, {}).get((boson,), sp.S.Zero)


def _gZ_sw2(s):
    gZ = sp.sqrt(s["g"]**2 + s["gp"]**2)
    sw2 = s["gp"]**2 / (s["g"]**2 + s["gp"]**2)
    return gZ, sw2


_mu = sp.Symbol("mu", integer=True)
gammaL = DiracGamma(_mu) * diracPL
gammaR = DiracGamma(_mu) * diracPR


def test_invariance(u1x):
    """Gauge invariance with fully SYMBOLIC U(1)_X charges (a, b, q_S) —
    the first model exercising symbol-valued reps — plus hermiticity of the
    ν-Yukawa written via H̃, and mass-dimension counting."""
    report = u1x["model"].check_invariance()
    assert report.ok, report.failures


def test_tadpoles(u1x):
    lam, lamS, lamHS = u1x["lam"], u1x["lamS"], u1x["lamHS"]
    v, vX = u1x["v"], u1x["vX"]
    tad = u1x["tad"]
    assert sp.simplify(tad[u1x["mu2"].s]
                       - (lam * v**2 + lamHS * vX**2 / 2)) == 0
    assert sp.simplify(tad[u1x["muS2"].s]
                       - (lamS * vX**2 + lamHS * v**2 / 2)) == 0


def test_neutral_gauge_mass_matrix(u1x):
    """3×3 (W3, B, X) block: the a-proportional Higgs entries mix X with
    the electroweak pair; exactly one massless eigenstate (det ≡ 0)."""
    g, gp, gx = u1x["g"], u1x["gp"], u1x["gx"]
    a, qs, v, vX = u1x["a"], u1x["qs"], u1x["v"], u1x["vX"]
    M3 = u1x["M3"]

    expected = sp.Matrix([
        [g**2 * v**2 / 4, -g * gp * v**2 / 4, -a * g * gx * v**2 / 4],
        [-g * gp * v**2 / 4, gp**2 * v**2 / 4, a * gp * gx * v**2 / 4],
        [-a * g * gx * v**2 / 4, a * gp * gx * v**2 / 4,
         gx**2 * (a**2 * v**2 + 4 * qs**2 * vX**2) / 4]])
    assert sp.simplify(M3 - expected) == sp.zeros(3, 3)
    assert sp.simplify(M3.det()) == 0


def test_zx_block_and_mixing_angle(u1x):
    """After the Weinberg rotation the photon decouples EXACTLY; the
    surviving (Z0, X) block pins tan 2θ′."""
    g, gp, gx = u1x["g"], u1x["gp"], u1x["gx"]
    a, qs, v, vX = u1x["a"], u1x["qs"], u1x["v"], u1x["vX"]
    gZ, _ = _gZ_sw2(u1x)
    M3r, blockZX = u1x["M3r"], u1x["blockZX"]

    # photon row and column identically zero
    assert sp.simplify(M3r[1, :]) == sp.zeros(1, 3)
    assert sp.simplify(M3r[:, 1]) == sp.zeros(3, 1)

    expected = sp.Matrix(
        [[gZ**2 * v**2 / 4, -a * gZ * gx * v**2 / 4],
         [-a * gZ * gx * v**2 / 4,
          gx**2 * (qs**2 * vX**2 + a**2 * v**2 / 4)]])
    assert sp.simplify(blockZX - expected) == sp.zeros(2, 2)

    _, tan2 = solve_mixing_angle_2x2(blockZX)
    expected_tan2 = 2 * a * gZ * gx * v**2 / (
        a**2 * gx**2 * v**2 + 4 * gx**2 * qs**2 * vX**2 - gZ**2 * v**2)
    assert sp.simplify(tan2 - expected_tan2) == 0
    ok, diff = numeric_equal(tan2, expected_tan2, [g, gp, gx, a, qs, v, vX])
    assert ok, diff

    # the explicit-angle rotation really diagonalizes the block
    Z0, X0 = sp.Symbol("Z0", real=True), u1x["X0"]
    rot = Rotation([Z0, X0], [u1x["Z"], u1x["Zp"]],
                   rotation_2x2(u1x["rot_zzp"].angle_solution))
    ok, residuals = rot.check(blockZX)
    assert ok, residuals


def test_masses_photon_and_invariants(u1x):
    """The chained Weinberg × Z–Z′ rotation as a single 3×3: photon mass
    exactly 0; Σm² and m_Z²·m_Z′² pinned via the trace/det invariants."""
    g, gp, gx = u1x["g"], u1x["gp"], u1x["gx"]
    a, qs, v, vX = u1x["a"], u1x["qs"], u1x["v"], u1x["vX"]
    gZ, _ = _gZ_sw2(u1x)
    thp = u1x["rot_zzp"].angle_solution
    c, s = sp.cos(thp), sp.sin(thp)

    R2e = sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R3 = R2e * u1x["R1e"]
    rot3 = Rotation([u1x["W3"], u1x["B0"], u1x["X0"]],
                    [u1x["Z"], u1x["A"], u1x["Zp"]], R3)
    ok, residuals = rot3.check(u1x["M3"])
    assert ok, residuals

    mZ2, mA2, mZp2 = rot3.masses_squared(u1x["M3"])
    assert sp.simplify(mA2) == 0

    syms = [g, gp, gx, a, qs, v, vX]
    ok, diff = numeric_equal(mZ2 + mZp2, sp.trace(u1x["blockZX"]), syms)
    assert ok, diff
    ok, diff = numeric_equal(mZ2 * mZp2,
                             gZ**2 * gx**2 * qs**2 * v**2 * vX**2 / 4, syms)
    assert ok, diff


def test_bminusl_limit(u1x):
    """a → 0 is the pure gauged B−L limit: no Z–Z′ mixing, SM m_Z,
    m_Z′ = g_X q_S v_X."""
    gx, a, qs, vX, v = (u1x["gx"], u1x["a"], u1x["qs"], u1x["vX"], u1x["v"])
    gZ, _ = _gZ_sw2(u1x)

    assert u1x["rot_zzp"].angle_solution.subs(a, 0) == 0
    block0 = u1x["blockZX"].subs(a, 0)
    assert sp.simplify(block0[0, 1]) == 0
    assert sp.simplify(block0[0, 0] - gZ**2 * v**2 / 4) == 0
    assert sp.simplify(block0[1, 1] - gx**2 * qs**2 * vX**2) == 0


def test_w_sector_unchanged(u1x):
    """W1/W2 rows of the full 5×5: diagonal m_W² = g²v²/4, no cross terms
    (the Higgs X-charge is ∝ 1₂ in SU(2) space)."""
    g, v = u1x["g"], u1x["v"]
    M5 = u1x["model"].gauge_mass_matrix(
        [u1x["W1"], u1x["W2"], u1x["W3"], u1x["B0"], u1x["X0"]])
    for r in (0, 1):
        assert sp.simplify(M5[r, r] - g**2 * v**2 / 4) == 0
        for c in range(5):
            if c != r:
                assert M5[r, c] == 0


def test_scalar_portal(u1x):
    """CP-even (h, s) block and the portal mixing angle; λ_HS → 0
    decouples the two scalars."""
    lam, lamS, lamHS = u1x["lam"], u1x["lamS"], u1x["lamHS"]
    v, vX = u1x["v"], u1x["vX"]
    Meven = u1x["Meven"]

    expected = sp.Matrix([[2 * lam * v**2, lamHS * v * vX],
                          [lamHS * v * vX, 2 * lamS * vX**2]])
    assert sp.simplify(Meven - expected) == sp.zeros(2, 2)

    _, tan2 = solve_mixing_angle_2x2(Meven)
    expected_tan2 = lamHS * v * vX / (lam * v**2 - lamS * vX**2)
    assert sp.simplify(tan2 - expected_tan2) == 0
    ok, diff = numeric_equal(tan2, expected_tan2, [lam, lamS, lamHS, v, vX])
    assert ok, diff

    assert u1x["rot_s"].angle_solution.subs(lamHS, 0) == 0
    M0 = Meven.subs(lamHS, 0)
    assert M0 == sp.diag(2 * lam * v**2, 2 * lamS * vX**2)


def test_pseudoscalars_and_goldstones(u1x):
    """Two massive neutral bosons need two Goldstones: the whole CP-odd
    (G0, Im S) block vanishes identically; the charged Goldstone too."""
    assert u1x["Modd"] == sp.zeros(2, 2)
    Mch = u1x["model"].mass_matrix([u1x["Gp"]], charged=True)
    assert sp.simplify(Mch[0, 0]) == 0


def test_zprime_fermion_couplings(u1x):
    """THE headline: Z′ff = −sinθ′·g_Z(T³ − Q s_w²) + cosθ′·g_X·q_f for
    every species, with q_f = a·Y_f + b·(B−L)_f."""
    gx, a, b = u1x["gx"], u1x["a"], u1x["b"]
    thp, Zp = u1x["thetap"], u1x["Zp"]
    gZ, sw2 = _gZ_sw2(u1x)
    cp, sn = sp.cos(thp), sp.sin(thp)
    syms = [u1x["g"], u1x["gp"], gx, a, b, thp]

    def zp_expected(T3, Q, q):
        return -sn * gZ * (T3 - Q * sw2) + cp * gx * q

    cases = [
        ("eL", u1x["eLbar"][i], gammaL, u1x["eL"][i],
         zp_expected(-sp.Rational(1, 2), -1, -a / 2 - b)),
        ("eR", u1x["eRbar"][i], gammaR, u1x["eRc"][i],
         zp_expected(0, -1, -a - b)),
        ("nuL", u1x["nuLbar"][i], gammaL, u1x["nuL"][i],
         zp_expected(sp.Rational(1, 2), 0, -a / 2 - b)),
        ("nuR", u1x["nuRbar"][i], gammaR, u1x["nuRc"][i],
         zp_expected(0, 0, -b)),
        ("uL", u1x["qLbar_u"][0][i], gammaL, u1x["qL_u"][0][i],
         zp_expected(sp.Rational(1, 2), sp.Rational(2, 3), a / 6 + b / 3)),
        ("uR", u1x["uRbar"][0][i], gammaR, u1x["uRc"][0][i],
         zp_expected(0, sp.Rational(2, 3), 2 * a / 3 + b / 3)),
        ("dL", u1x["qLbar_d"][0][i], gammaL, u1x["qL_d"][0][i],
         zp_expected(-sp.Rational(1, 2), -sp.Rational(1, 3), a / 6 + b / 3)),
    ]
    for name, bar, gamma, field, expected in cases:
        coeff = _coeff(u1x, bar, gamma, field, Zp)
        assert sp.simplify(coeff - expected) == 0, name
        ok, diff = numeric_equal(coeff, expected, syms)
        assert ok, (name, diff)

    # νR couples to the Z′ ONLY through its X charge — pure cosθ′·g_X·(−b)
    nuR_coeff = _coeff(u1x, u1x["nuRbar"][i], gammaR, u1x["nuRc"][i], Zp)
    assert sp.simplify(nuR_coeff + b * gx * cp) == 0


def test_z_couplings_modified(u1x):
    """Zff picks up a sinθ′·g_X·q_f admixture; θ′ → 0 recovers the SM."""
    gx, a, b = u1x["gx"], u1x["a"], u1x["b"]
    thp, Z = u1x["thetap"], u1x["Z"]
    gZ, sw2 = _gZ_sw2(u1x)
    cp, sn = sp.cos(thp), sp.sin(thp)

    zeR = _coeff(u1x, u1x["eRbar"][i], gammaR, u1x["eRc"][i], Z)
    expected = cp * gZ * sw2 + sn * gx * (-a - b)
    assert sp.simplify(zeR - expected) == 0
    ok, diff = numeric_equal(zeR, expected,
                             [u1x["g"], u1x["gp"], gx, a, b, thp])
    assert ok, diff
    assert sp.simplify(zeR.subs(thp, 0) - gZ * sw2) == 0

    zeL = _coeff(u1x, u1x["eLbar"][i], gammaL, u1x["eL"][i], Z)
    expected = cp * gZ * (-sp.Rational(1, 2) + sw2) + sn * gx * (-a / 2 - b)
    assert sp.simplify(zeL - expected) == 0

    # νR now talks to the Z, but only through the mixing
    znuR = _coeff(u1x, u1x["nuRbar"][i], gammaR, u1x["nuRc"][i], Z)
    assert sp.simplify(znuR + b * gx * sn) == 0
    assert znuR.subs(thp, 0) == 0


def test_photon_unchanged(u1x):
    """The Z–Z′ rotation never touches A: photon couplings are exactly the
    SM electric charges; νL and νR stay dark."""
    A = u1x["A"]
    g, gp = u1x["g"], u1x["gp"]
    e_em = g * gp / sp.sqrt(g**2 + gp**2)

    aeL = _coeff(u1x, u1x["eLbar"][i], gammaL, u1x["eL"][i], A)
    assert sp.simplify(aeL + e_em) == 0
    auL = _coeff(u1x, u1x["qLbar_u"][0][i], gammaL, u1x["qL_u"][0][i], A)
    assert sp.simplify(auL - 2 * e_em / 3) == 0
    assert _coeff(u1x, u1x["nuLbar"][i], gammaL, u1x["nuL"][i], A) == 0
    assert _coeff(u1x, u1x["nuRbar"][i], gammaR, u1x["nuRc"][i], A) == 0


def test_neutrino_dirac_mass(u1x):
    """M_ν[i,j] = Y_ν[i,j]·v/√2 — flavor-generic Dirac mass via H̃."""
    M_nu = fermion_mass_matrix(u1x["LYukN"], u1x["nuLbar"], u1x["nuRc"],
                               u1x["model"].vacuum, 3, (i, j), gamma=diracPR)
    Yv, v = u1x["Yv"], u1x["v"]
    expected = sp.Matrix(3, 3, lambda r, c: Yv[r, c] * v / sp.sqrt(2))
    assert sp.simplify(M_nu - expected) == sp.zeros(3, 3)


def test_higgs_portal_hff(u1x):
    """Portal mixing splits the SM hff strength by cosθ_s (h1) and
    −sinθ_s (h2, with the relative sign from the orthogonal rotation);
    θ_s → 0 recovers the SM."""
    ths, h1, h2 = u1x["thetas"], u1x["h1"], u1x["h2"]
    Ye, Yv = u1x["Ye"], u1x["Yv"]

    h1e = _coeff(u1x, u1x["eLbar"][i], diracPR, u1x["eRc"][j], h1)
    assert sp.simplify(h1e + sp.cos(ths) * Ye[i, j] / sp.sqrt(2)) == 0
    h2e = _coeff(u1x, u1x["eLbar"][i], diracPR, u1x["eRc"][j], h2)
    assert sp.simplify(h2e - sp.sin(ths) * Ye[i, j] / sp.sqrt(2)) == 0
    assert sp.simplify(h1e.subs(ths, 0) + Ye[i, j] / sp.sqrt(2)) == 0

    h1n = _coeff(u1x, u1x["nuLbar"][i], diracPR, u1x["nuRc"][j], h1)
    assert sp.simplify(h1n + sp.cos(ths) * Yv[i, j] / sp.sqrt(2)) == 0
