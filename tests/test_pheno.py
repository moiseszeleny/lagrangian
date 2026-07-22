"""Tests for the pheno (decay calculator) submodule.

House style: pin physics, not code paths.  Every symbolic width is checked
against an **independent explicit-4×4-matrix oracle** (``_oracle_*`` below) as
well as against the textbook closed form, so no result rests on the covariant
trace engine alone.  The oracle shares no code with that engine: it builds
``p̸`` and ``Γ`` as literal Dirac-basis matrices and takes ``Matrix.trace()`` in
the parent rest frame.  In particular it represents γ₅ as an actual matrix,
which is what independently validates the covariant engine's γ₅ drop.
"""

import pytest
import sympy as sp

from feynlag import (
    Bilinear, Dmu, ExternalParameter, InternalParameter, Lagrangian, Model,
    Rotation, SU2, Scalar, U1, WeylFermion, conjugate_pair, dag, diracPL,
    diracPR, fermion_gauge_current, rotation_2x2,
)
from feynlag.dirac import DiracGamma, PL, _dirac_rep, minkowski_metric
from feynlag.pheno import (
    DecayCalculator, TwoBodyKinematics, classify_gamma, ffs_squared,
    ffv_squared, kallen, reduce_projectors, vvs_squared,
)
from feynlag.pheno.trace import dirac_trace as direct_trace
from feynlag.verify import numeric_equal

_MET = sp.diag(1, -1, -1, -1)


# --------------------------------------------------------------------------
# explicit 4×4 oracle — shares no code with the covariant engine
# --------------------------------------------------------------------------

def _rest_frame(M, m1, m2):
    """Contravariant components of the daughters in the parent rest frame."""
    E1 = (M**2 + m1**2 - m2**2) / (2 * M)
    E2 = (M**2 + m2**2 - m1**2) / (2 * M)
    k = sp.sqrt(kallen(M**2, m1**2, m2**2)) / (2 * M)
    return [E1, 0, 0, k], [E2, 0, 0, -k]


def _slash(p):
    """``p̸ = p^μ γ_μ`` as an explicit 4×4 matrix."""
    rep = _dirac_rep()
    out = sp.zeros(4, 4)
    for mu in range(4):
        out += _MET[mu, mu] * p[mu] * rep[("g", mu)]
    return out


def _oracle_ffs(gL, gR, M, m1, m2):
    """``Σ|M|²`` for ``S → f f̄``, explicit matrices."""
    rep = _dirac_rep()
    P1, P2 = _rest_frame(M, m1, m2)
    A = _slash(P1) + m1 * sp.eye(4)
    B = _slash(P2) - m2 * sp.eye(4)
    G = gL * rep["PL"] + gR * rep["PR"]
    Gbar = sp.conjugate(gL) * rep["PR"] + sp.conjugate(gR) * rep["PL"]
    return sp.simplify((A * G * B * Gbar).trace())


def _oracle_ffv(gL, gR, M, m1, m2):
    """``⟨|M|²⟩`` for ``V → f f̄``, explicit matrices, polarization-averaged."""
    rep = _dirac_rep()
    P1, P2 = _rest_frame(M, m1, m2)
    Pt = [P1[i] + P2[i] for i in range(4)]
    A = _slash(P1) + m1 * sp.eye(4)
    B = _slash(P2) - m2 * sp.eye(4)
    total = 0
    for a in range(4):
        for b in range(4):
            # polarization sum with LOWER indices, contracted against T^{ab}
            pol = (-_MET[a, b]
                   + (_MET[a, a] * Pt[a]) * (_MET[b, b] * Pt[b]) / M**2)
            if pol == 0:
                continue
            Ga = gL * rep[("g", a)] * rep["PL"] + gR * rep[("g", a)] * rep["PR"]
            Gb = (sp.conjugate(gL) * rep[("g", b)] * rep["PL"]
                  + sp.conjugate(gR) * rep[("g", b)] * rep["PR"])
            total += pol * (A * Ga * B * Gb).trace()
    return sp.simplify(total / 3)


# --------------------------------------------------------------------------
# kinematics + trace identities
# --------------------------------------------------------------------------

def test_kallen():
    # kallen(100, 4, 9) = 10000 + 16 + 81 - 800 - 72 - 1800 = 7425
    assert kallen(100, 4, 9) == 7425


def test_trace_two_gammas():
    mu, nu = sp.symbols('mu nu')
    assert direct_trace(DiracGamma(mu) * DiracGamma(nu)) == \
        4 * minkowski_metric(mu, nu)


def test_trace_two_gammas_PL():
    mu, nu = sp.symbols('mu nu')
    assert direct_trace(DiracGamma(mu) * DiracGamma(nu) * PL()) == \
        2 * minkowski_metric(mu, nu)


def test_trace_four_gammas():
    mu, nu, alpha, beta = sp.symbols('mu nu alpha beta')
    res = direct_trace(DiracGamma(mu) * DiracGamma(nu)
                       * DiracGamma(alpha) * DiracGamma(beta))
    g = minkowski_metric
    assert res == 4 * (g(mu, nu) * g(alpha, beta)
                       - g(mu, alpha) * g(nu, beta)
                       + g(mu, beta) * g(nu, alpha))


def test_direct_trace_matches_explicit_rep():
    """``trace.py`` and the explicit Dirac rep agree at every integer index.

    Both are independent of the covariant engine, so this pins the metric and
    projector conventions the whole module shares.
    """
    rep = _dirac_rep()
    for a in range(4):
        for b in range(4):
            expected = (rep[("g", a)] * rep[("g", b)]).trace()
            assert sp.simplify(
                direct_trace(DiracGamma(a) * DiracGamma(b)) - expected) == 0
            expected_pl = (rep[("g", a)] * rep[("g", b)] * rep["PL"]).trace()
            assert sp.simplify(
                direct_trace(DiracGamma(a) * DiracGamma(b) * PL())
                - expected_pl) == 0


def test_classify_gamma():
    mu, nu = sp.symbols('mu nu')
    assert classify_gamma(diracPL) == ("S", "L")
    assert classify_gamma(diracPR) == ("S", "R")
    assert classify_gamma(DiracGamma(mu) * diracPL) == ("V", "L")
    with pytest.raises(NotImplementedError):
        classify_gamma(DiracGamma(mu) * DiracGamma(nu))


# --------------------------------------------------------------------------
# covariant engine vs. the explicit-matrix oracle
# --------------------------------------------------------------------------

def test_ffs_matches_explicit_oracle():
    """Scalar decay with both chiralities and both masses on — general case."""
    M, m1, m2, gL, gR = sp.symbols('M m1 m2 g_L g_R', positive=True)
    kin = TwoBodyKinematics(M, m1, m2)
    covariant = ffs_squared(gL, gR, kin)
    oracle = _oracle_ffs(gL, gR, M, m1, m2)
    assert sp.simplify(sp.expand(covariant - oracle)) == 0


def test_ffv_matches_explicit_oracle():
    """Vector decay with independent ``g_L``, ``g_R``.

    This independently validates dropping the γ₅ (ε-tensor) trace: the oracle
    carries γ₅ as a literal matrix and never makes that assumption, yet the two
    agree exactly.
    """
    M, m, gL, gR = sp.symbols('M m g_L g_R', positive=True)
    kin = TwoBodyKinematics(M, m, m)
    covariant = ffv_squared(gL, gR, kin)
    oracle = _oracle_ffv(gL, gR, M, m, m)
    assert sp.simplify(sp.expand(covariant - oracle)) == 0


def test_ffv_chiral_is_half_the_vector_result():
    """A pure ``P_L`` current gives exactly half a pure vector current — the
    closed-form counterpart of the γ₅ check above."""
    M, g = sp.symbols('M g', positive=True)
    kin = TwoBodyKinematics(M, 0, 0)
    vector = ffv_squared(g, g, kin) * kin.phase_space()
    chiral = ffv_squared(g, 0, kin) * kin.phase_space()
    assert sp.simplify(vector - g**2 * M / (12 * sp.pi)) == 0
    assert sp.simplify(chiral - g**2 * M / (24 * sp.pi)) == 0
    assert sp.simplify(vector - 2 * chiral) == 0


def test_gamma5_guard_raises_outside_two_body():
    """The ε-vanishing argument holds only for a two-body final state, so a
    later 1→3 extension must fail loudly rather than drop a non-zero term."""
    chain = sp.S.One
    with pytest.raises(NotImplementedError, match="independent momenta"):
        reduce_projectors(chain, "L", n_free_indices=0, n_momenta=4)
    with pytest.raises(NotImplementedError, match="free Lorentz indices"):
        reduce_projectors(chain, "L", n_free_indices=3, n_momenta=2)
    # the supported 1→2 configurations do not raise
    assert reduce_projectors(chain, "L", 0, 2)[1] == sp.Rational(1, 2)
    assert reduce_projectors(chain, None, 0, 2)[1] == 1


# --------------------------------------------------------------------------
# closed-form widths
# --------------------------------------------------------------------------

def test_higgs_to_fermions_p_wave():
    """``Γ(h→f f̄) = m_h m_f² β³/(8π v²)`` — the β³ is the CP-even scalar
    signature (a CP-odd scalar would give β¹)."""
    mh, mf, v = sp.symbols('m_h m_f v', positive=True)
    kin = TwoBodyKinematics(mh, mf, mf)
    width = ffs_squared(mf / v, mf / v, kin) * kin.phase_space()
    beta = sp.sqrt(1 - 4 * mf**2 / mh**2)
    expected = mh * mf**2 * beta**3 / (8 * sp.pi * v**2)
    assert sp.simplify(sp.expand(width - expected)) == 0


def test_higgs_to_vectors():
    """``Γ(h→VV) = g²m_h³/(64πm_V²)·√(1−4x)(1−4x+12x²)``, ``x = m_V²/m_h²``."""
    g, mh, mV = sp.symbols('g m_h m_V', positive=True)
    kin = TwoBodyKinematics(mh, mV, mV)
    width = vvs_squared(g * mV, kin) * kin.phase_space()
    x = mV**2 / mh**2
    expected = (g**2 * mh**3 / (64 * sp.pi * mV**2)
                * sp.sqrt(1 - 4 * x) * (1 - 4 * x + 12 * x**2))
    assert sp.simplify(sp.expand(width - expected)) == 0


def test_massive_vector_to_fermions_closed_form():
    """``Γ(V→f f̄) = (Mβ/24π)[(g_L²+g_R²)(1−m²/M²) + 6g_Lg_R m²/M²]``."""
    M, m, gL, gR = sp.symbols('M m g_L g_R', positive=True)
    kin = TwoBodyKinematics(M, m, m)
    width = ffv_squared(gL, gR, kin) * kin.phase_space()
    beta = sp.sqrt(1 - 4 * m**2 / M**2)
    expected = (M * beta / (24 * sp.pi)
                * ((gL**2 + gR**2) * (1 - m**2 / M**2)
                   + 6 * gL * gR * m**2 / M**2))
    assert sp.simplify(sp.expand(width - expected)) == 0


def test_amplitude_numeric_agrees_with_oracle():
    """Dual verification: random-point numeric equality, not only ``simplify``."""
    M, m, gL, gR = sp.symbols('M m g_L g_R', positive=True)
    kin = TwoBodyKinematics(M, 2 * m, m)
    covariant = ffv_squared(gL, gR, kin)
    oracle = _oracle_ffv(gL, gR, M, 2 * m, m)
    ok, diff = numeric_equal(covariant, oracle, [M, m, gL, gR],
                             sample_range=(5.0, 10.0), seed=11)
    assert ok, f"max relative difference {diff}"


# --------------------------------------------------------------------------
# end to end through a Standard Model
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sm():
    """SM electroweak + one lepton generation, in the physical basis."""
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)
    ye = ExternalParameter("ye", 0.01, positive=True)

    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])

    i = sp.Symbol("i", integer=True)
    HdH = (dag(H) * H.mat)[0]
    DH = Dmu(H)
    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-(-mu2.s * HdH + lam.s * HdH**2), sector="potential")
    L.add(fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i),
          sector="gauge")

    nuLb, eLb = Ll.bar_components
    nuL, eL = Ll.components
    eRb, eRc = eR.bar_components[0], eR.components[0]
    yuk = -ye.s * (Bilinear(nuLb[i], diracPR, eRc[i]) * H.components[0]
                   + Bilinear(eLb[i], diracPR, eRc[i]) * H.components[1])
    L.add(yuk + sp.conjugate(yuk), sector="yukawa")

    model = Model("SM_lep", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, lam, mu2, ye], lagrangian=L)
    model.solve_tadpoles([mu2])

    W1, W2, W3 = SU2L.bosons().components
    B = U1Y.bosons().components[0]
    Z, A = sp.symbols("Z A", real=True)
    model.rotate(Rotation([W3, B], [Z, A],
                          rotation_2x2(-sp.atan(g1.s / gw.s))))
    Wp, Wm = sp.symbols("Wp Wm")
    model.rotate(Rotation([W1, W2], [Wp, Wm],
                          sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2),
                          kind="unitary"))

    h = sp.Symbol("H0_r", real=True)
    G0 = sp.Symbol("H0_i", real=True)
    Gp = H.components[0]
    Gm, cmap = conjugate_pair(Gp, "Gm")
    e, ebar, nu, nubar = sp.symbols("e ebar nu nubar")
    particle_map = {eL[i]: e, eRc[i]: e, eLb[i]: ebar, eRb[i]: ebar,
                    nuL[i]: nu, nuLb[i]: nubar}
    return dict(model=model, cmap=cmap, particle_map=particle_map,
                bosons=[h, G0, Gp, Gm, Z, A, Wp, Wm],
                h=h, Z=Z, A=A, Wp=Wp, Wm=Wm, e=e, ebar=ebar, nu=nu,
                nubar=nubar, gw=gw, g1=g1, v=v, ye=ye)


@pytest.fixture(scope="module")
def calc(sm):
    mZ, mW, mh, me = sp.symbols("m_Z m_W m_h m_e", positive=True)
    masses = {sm["Z"]: mZ, sm["Wp"]: mW, sm["Wm"]: mW, sm["A"]: 0,
              sm["h"]: mh, sm["e"]: me, sm["ebar"]: me,
              sm["nu"]: 0, sm["nubar"]: 0}
    calculator = DecayCalculator(
        sm["model"], masses, boson_fields=sm["bosons"],
        fermion_sectors=("gauge", "yukawa"), conjugate_map=sm["cmap"],
        particle_map=sm["particle_map"])
    return calculator, dict(mZ=mZ, mW=mW, mh=mh, me=me)


def test_z_to_neutrinos(calc, sm):
    """``Γ(Z→νν̄) = m_Z(g²+g'²)/(96π) = m_Z³/(24πv²)``."""
    calculator, m = calc
    width = calculator.partial_widths(sm["Z"])[(sm["nubar"], sm["nu"])]
    g, gp, mZ, v = sm["gw"].s, sm["g1"].s, m["mZ"], sm["v"].s
    assert sp.simplify(width - mZ * (g**2 + gp**2) / (96 * sp.pi)) == 0
    # substituting the tree-level m_Z = v√(g²+g'²)/2 gives the familiar
    # G_F form Γ = m_Z³/(24πv²) (= G_F m_Z³/(12√2 π))
    mZ_tree = v * sp.sqrt(g**2 + gp**2) / 2
    assert sp.simplify(width.subs(mZ, mZ_tree)
                       - mZ_tree**3 / (24 * sp.pi * v**2)) == 0


def test_z_to_charged_leptons(calc, sm):
    """``Γ(Z→ℓ⁺ℓ⁻) = m_Z(5g'⁴ − 2g²g'² + g⁴)/(96π(g²+g'²))`` at ``m_ℓ→0``.

    Equivalently ``(m_Z/24π)(g_L²+g_R²)`` with the SM chiral couplings, which
    makes this a genuine test of the L/R merge: without the ``particle_map``
    the two currents would land in separate, wrong channels.
    """
    calculator, m = calc
    width = calculator.partial_widths(sm["Z"])[(sm["ebar"], sm["e"])]
    g, gp = sm["gw"].s, sm["g1"].s
    expected = (m["mZ"] * (5 * gp**4 - 2 * g**2 * gp**2 + g**4)
                / (96 * sp.pi * (g**2 + gp**2)))
    assert sp.simplify(sp.simplify(width.subs(m["me"], 0)) - expected) == 0


def test_w_to_lepton_neutrino(calc, sm):
    """``Γ(W→ℓν) = g²m_W/(48π)``."""
    calculator, m = calc
    width = calculator.partial_widths(sm["Wp"])[(sm["nubar"], sm["e"])]
    assert sp.simplify(sp.simplify(width.subs(m["me"], 0))
                       - sm["gw"].s**2 * m["mW"] / (48 * sp.pi)) == 0


def test_higgs_to_fermions_from_model(calc, sm):
    """``Γ(h→ℓ⁺ℓ⁻) = y²m_h β³/(16π)`` — the same β³ as the standalone test,
    but routed through extraction from the Yukawa sector."""
    calculator, m = calc
    width = calculator.partial_widths(sm["h"])[(sm["ebar"], sm["e"])]
    beta = sp.sqrt(1 - 4 * m["me"]**2 / m["mh"]**2)
    expected = sm["ye"].s**2 * m["mh"] * beta**3 / (16 * sp.pi)
    assert sp.simplify(sp.expand(width - expected)) == 0


def test_higgs_to_zz_carries_identical_particle_factor(calc, sm):
    """``h→ZZ`` gets ``1/2!``; ``h→W⁺W⁻`` does not (distinct daughters)."""
    calculator, m = calc
    widths = calculator.partial_widths(sm["h"])
    zz = widths[(sm["Z"], sm["Z"])]
    ww = widths[(sm["Wm"], sm["Wp"])]
    g, gp = sm["gw"].s, sm["g1"].s
    # the Z case is the W case with g → g_Z, m_W → m_Z, times ½
    ww_as_zz = ww.subs(m["mW"], m["mZ"]).subs(g, sp.sqrt(g**2 + gp**2)) / 2
    assert sp.simplify(sp.expand(zz - ww_as_zz)) == 0


def test_closed_channel_is_dropped(sm):
    """A kinematically forbidden channel produces no width at all."""
    masses = {sm["Z"]: sp.Integer(91), sm["Wp"]: sp.Integer(80),
              sm["Wm"]: sp.Integer(80), sm["A"]: 0, sm["h"]: sp.Integer(125),
              sm["e"]: sp.Integer(200), sm["ebar"]: sp.Integer(200),
              sm["nu"]: 0, sm["nubar"]: 0}
    calculator = DecayCalculator(
        sm["model"], masses, boson_fields=sm["bosons"],
        fermion_sectors=("gauge", "yukawa"), conjugate_map=sm["cmap"],
        particle_map=sm["particle_map"])
    widths = calculator.partial_widths(sm["Z"])
    assert (sm["ebar"], sm["e"]) not in widths      # 2×200 > 91: closed
    assert (sm["nubar"], sm["nu"]) in widths        # massless: always open


def test_branching_ratios_sum_to_one(calc, sm):
    calculator, _ = calc
    ratios = calculator.branching_ratios(sm["Z"])
    assert ratios
    assert sp.simplify(sp.Add(*ratios.values()) - 1) == 0


def test_numeric_widths_close_sub_threshold_channels(calc, sm):
    """A channel that is open symbolically but shut at the numeric mass point
    must contribute exactly zero — not an imaginary ``√λ``.

    ``h → W⁺W⁻`` and ``h → ZZ`` at ``m_h = 125 GeV`` are the real case: with
    symbolic masses the threshold is undecidable so the channels survive, and
    naively substituting numbers gives an imaginary width that would poison
    the total and every branching ratio.
    """
    calculator, m = calc
    calculator.parameters = sm["model"].parameters
    numbers = {m["mZ"]: 91.1876, m["mW"]: 80.377, m["mh"]: 125.25,
               m["me"]: 1.77686}
    widths = calculator.numeric_partial_widths(sm["h"], extra=numbers)

    assert widths[(sm["Wm"], sm["Wp"])] == 0.0      # 2m_W = 160.8 > 125.25
    assert widths[(sm["Z"], sm["Z"])] == 0.0        # 2m_Z = 182.4 > 125.25
    assert widths[(sm["ebar"], sm["e"])] > 0        # open

    # the naive route is exactly what must NOT be used: it is complex
    naive = calculator.numeric(
        calculator.partial_widths(sm["h"])[(sm["Wm"], sm["Wp"])],
        extra=numbers)
    assert abs(naive.imag) > 0

    ratios = calculator.numeric_branching_ratios(sm["h"], extra=numbers)
    assert abs(sum(ratios.values()) - 1.0) < 1e-12
    assert abs(ratios[(sm["ebar"], sm["e"])] - 1.0) < 1e-12


def test_color_factor_multiplies_width(sm, calc):
    """A colour-triplet pair gives exactly ``N_c = 3`` times the singlet width
    at identical couplings."""
    plain, m = calc
    common = dict(boson_fields=sm["bosons"],
                  fermion_sectors=("gauge", "yukawa"),
                  conjugate_map=sm["cmap"], particle_map=sm["particle_map"])
    colored = DecayCalculator(sm["model"], plain.masses,
                              color_factors={sm["e"]: 3}, **common)
    key = (sm["ebar"], sm["e"])
    assert sp.simplify(colored.partial_widths(sm["Z"])[key]
                       - 3 * plain.partial_widths(sm["Z"])[key]) == 0


def test_numeric_path_matches_direct_substitution(calc, sm):
    """``DecayCalculator.numeric`` (through ``ParameterSet.numeric``) agrees
    with substituting by hand, and lands on the PDG value."""
    calculator, m = calc
    calculator.parameters = sm["model"].parameters
    width = calculator.partial_widths(sm["Z"])[(sm["nubar"], sm["nu"])]
    subs = {m["mZ"]: 91.1876}
    value = calculator.numeric(width, extra=subs)
    direct = complex(width.subs({sm["gw"].s: 0.6535, sm["g1"].s: 0.3580,
                                 **subs}).evalf())
    assert abs(value - direct) < 1e-9 * max(abs(direct), 1.0)
    # Γ(Z→νν̄) ≈ 0.166 GeV per generation
    assert abs(value.real - 0.166) < 0.005
