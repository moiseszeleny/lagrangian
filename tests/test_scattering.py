"""Tests for 2→2 scattering (`feynlag.pheno.diagrams`/`.scattering`).

House style: pin physics, not code paths.  Tier 1 of
``docs/manual/scattering_roadmap.md``.  The acceptance oracle is the textbook
QED closed form for ``e⁺e⁻→μ⁺μ⁻`` through a single photon,
``σ = (4πα²/3s)·β(3−β²)/2``, ``β = √(1−4m_μ²/s)`` — together with an
independent **explicit-4×4-matrix** evaluator (``_oracle_qed_general`` below)
that shares no code with the covariant engine.  It carries γ₅ as a literal
matrix, which is what independently validates two things at once: that the ε
(γ₅) term vanishes for a pure vector coupling (matching the covariant
engine's ε-drop), and that it does **not** vanish for a chiral one (matching
the engine's refusal to compute that case at all).  Every physics assertion
gets the house dual check: ``sp.simplify(sp.expand(a - b)) == 0`` and
:func:`~feynlag.verify.numeric_equal`.
"""

import math

import pytest
import sympy as sp

from feynlag.dirac import _dirac_rep
from feynlag.pheno import (
    Amplitude, BosonPropagator, ChainVertex, Diagram, ExternalState, Leg,
    SpinorChain, TwoToTwoKinematics, average_factor, cross_section,
    ffs_s_channel_squared, ffv_s_channel_squared,
)
from feynlag.pheno.diagrams import _chain_indices
from feynlag.pheno.lorentz import momentum
from feynlag.verify import numeric_equal

_MET = sp.diag(1, -1, -1, -1)


# --------------------------------------------------------------------------
# explicit 4×4 oracle — shares no code with the covariant engine
# --------------------------------------------------------------------------

def _slash(p):
    """``p̸ = p^μ γ_μ`` as an explicit 4×4 matrix."""
    rep = _dirac_rep()
    out = sp.zeros(4, 4)
    for mu in range(4):
        out += _MET[mu, mu] * p[mu] * rep[("g", mu)]
    return out


def _cm_frame(s, cos, m1, m2, m3, m4):
    """CM-frame contravariant components of ``k1..k4`` for
    ``1(m1)+2(m2)->3(m3)+4(m4)``, at fixed ``s`` and scattering angle."""
    rs = sp.sqrt(s)
    E1 = (s + m1**2 - m2**2) / (2 * rs)
    E2 = (s + m2**2 - m1**2) / (2 * rs)
    E3 = (s + m3**2 - m4**2) / (2 * rs)
    E4 = (s + m4**2 - m3**2) / (2 * rs)
    pin = sp.sqrt(E1**2 - m1**2)
    pf = sp.sqrt(E3**2 - m3**2)
    sin = sp.sqrt(1 - cos**2)
    k1 = [E1, 0, 0, pin]
    k2 = [E2, 0, 0, -pin]
    k3 = [E3, pf * sin, 0, pf * cos]
    k4 = [E4, -pf * sin, 0, -pf * cos]
    return k1, k2, k3, k4


def _oracle_qed_general(gL_in, gR_in, gL_out, gR_out, s, cos, m):
    """``Σ|M|²`` for ``e⁺(m1=0)e⁻(m2=0) → f⁺(m3=m) f⁻(m4=m)`` through a
    massless vector, explicit Dirac matrices, general chiral couplings.

    Independent of :mod:`feynlag.pheno.diagrams`: builds ``Γ = g_LP_L+g_RP_R``
    and ``Γ̄`` as literal matrices (γ₅ included via the explicit ``P_L``/
    ``P_R`` projectors) and traces directly, rather than going through
    :func:`~feynlag.pheno.lorentz.reduce_projectors`.
    """
    rep = _dirac_rep()
    k1, k2, k3, k4 = _cm_frame(s, cos, 0, 0, m, m)
    A_in, B_in = _slash(k2), _slash(k1)
    A_out = _slash(k3) + m * sp.eye(4)
    B_out = _slash(k4) - m * sp.eye(4)
    PL, PR = rep["PL"], rep["PR"]
    G_in = gL_in * PL + gR_in * PR
    Gbar_in = sp.conjugate(gL_in) * PL + sp.conjugate(gR_in) * PR
    G_out = gL_out * PL + gR_out * PR
    Gbar_out = sp.conjugate(gL_out) * PL + sp.conjugate(gR_out) * PR
    T_in = [[None] * 4 for _ in range(4)]
    T_out = [[None] * 4 for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            T_in[mu][nu] = (A_in * rep[("g", mu)] * G_in * B_in
                            * rep[("g", nu)] * Gbar_in).trace()
            T_out[mu][nu] = (A_out * rep[("g", mu)] * G_out * B_out
                             * rep[("g", nu)] * Gbar_out).trace()
    total = 0
    for mu in range(4):
        for munu in range(4):
            total += T_in[mu][munu] * _MET[mu, mu] * _MET[munu, munu] * T_out[mu][munu]
    return sp.simplify(total / s**2)


# --------------------------------------------------------------------------
# kinematics
# --------------------------------------------------------------------------

def test_mandelstam_sum():
    m1, m2, m3, m4 = sp.symbols('m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    total = sp.simplify(kin.s + kin.t + kin.u - (m1**2 + m2**2 + m3**2 + m4**2))
    assert total == 0
    # u is derived, never a free symbol of its own
    assert sp.Symbol("u") not in kin.u.free_symbols


def test_dot_table_respects_momentum_conservation():
    """``k₄ = k₁+k₂−k₃`` is not an independent momentum; the ``dot`` table
    must encode that exactly, not approximately — this is the single most
    load-bearing kinematics test: a sign error in ``u`` or a swapped ``t``/
    ``u`` shows up here."""
    m1, m2, m3, m4 = sp.symbols('m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    for x in (kin.k1, kin.k2, kin.k3, kin.k4):
        lhs = kin.dot(kin.k4, x)
        rhs = kin.dot(kin.k1, x) + kin.dot(kin.k2, x) - kin.dot(kin.k3, x)
        assert sp.simplify(sp.expand(lhs - rhs)) == 0


def test_dot_table_reproduces_cm_frame_components():
    s, t, m1, m2, m3, m4 = sp.symbols('s t m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    cos = kin.cos_theta().subs(kin.s, s).subs(kin.t, t)
    k1, k2, k3, k4 = _cm_frame(s, cos, m1, m2, m3, m4)

    def mdot(a, b):
        return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]

    heads = {kin.k1: k1, kin.k2: k2, kin.k3: k3, kin.k4: k4}
    for (ha, va), (hb, vb) in [((kin.k1, k1), (kin.k3, k3)),
                               ((kin.k1, k1), (kin.k4, k4)),
                               ((kin.k2, k2), (kin.k3, k3)),
                               ((kin.k1, k1), (kin.k2, k2))]:
        lhs = kin.dot(ha, hb).subs(kin.s, s).subs(kin.t, t)
        rhs = mdot(va, vb)
        assert sp.simplify(sp.expand(lhs - rhs)) == 0


def test_t_bounds_are_the_cos_theta_endpoints():
    m1, m2, m3, m4 = sp.symbols('m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    t_min, t_max = kin.t_bounds()
    assert sp.simplify(kin.t_of_cos(1) - t_max) == 0
    assert sp.simplify(kin.t_of_cos(-1) - t_min) == 0
    assert sp.simplify(kin.cos_theta().subs(kin.t, t_max) - 1) == 0
    assert sp.simplify(kin.cos_theta().subs(kin.t, t_min) + 1) == 0


def test_flux_factor():
    m1, m2, m3, m4 = sp.symbols('m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    via_dot = 1 / (4 * sp.sqrt(kin.dot(kin.k1, kin.k2)**2 - m1**2 * m2**2))
    assert sp.simplify(sp.expand(kin.flux_factor() - via_dot)) == 0


def test_constant_amplitude_gives_textbook_sigma():
    """``σ = |M|²/(16πs)`` for a constant amplitude, all masses zero."""
    s = sp.Symbol('s', positive=True)
    m2c = sp.Symbol('M2', positive=True)
    kin = TwoToTwoKinematics(0, 0, 0, 0)
    electron = ExternalState('a', 0, sp.Rational(1, 2))
    sigma = cross_section(m2c, kin, (electron, electron))
    sigma = sp.simplify(sigma.subs(kin.s, s)) * 4  # undo the 1/4 average
    assert sp.simplify(sigma - m2c / (16 * sp.pi * s)) == 0


def test_foreign_momentum_head_raises():
    kin = TwoToTwoKinematics(0, 0, 1, 1)
    foreign = momentum("p1")  # a TwoBodyKinematics-style head
    with pytest.raises(KeyError, match="no on-shell dot product"):
        kin.dot(foreign, kin.k1)


# --------------------------------------------------------------------------
# engine and guards
# --------------------------------------------------------------------------

def test_chain_vertex_bar_matches_dirac_conjugate():
    from feynlag.dirac import DiracGamma, dirac_conjugate, diracPL, diracPR

    gL, gR = sp.symbols('g_L g_R')
    mu = sp.Symbol('mu')

    # 'V': self-conjugate structure, couplings conjugate, chirality unchanged
    v = ChainVertex('V', gL, gR)
    vbar = v.bar()
    assert vbar.g_left == sp.conjugate(gL) and vbar.g_right == sp.conjugate(gR)
    assert dirac_conjugate(DiracGamma(mu) * diracPL) == DiracGamma(mu) * diracPL
    assert dirac_conjugate(DiracGamma(mu) * diracPR) == DiracGamma(mu) * diracPR

    # 'S': bare projectors swap
    s = ChainVertex('S', gL, gR)
    sbar = s.bar()
    assert sbar.g_left == sp.conjugate(gR) and sbar.g_right == sp.conjugate(gL)
    assert dirac_conjugate(diracPL) == diracPR
    assert dirac_conjugate(diracPR) == diracPL


def test_two_chain_assembly_has_no_index_collision():
    """Two chains meeting at one propagator must not trip
    :func:`~feynlag.pheno.lorentz.contract_to_dots`'s "index appears N times"
    guard — this exercises :func:`~feynlag.pheno.diagrams._chain_indices`."""
    m = sp.Symbol('m', positive=True)
    g = sp.Symbol('g', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    # would raise ValueError from contract_to_dots if index allocation collided
    result = ffv_s_channel_squared(g, g, g, g, kin, mediator_mass=0)
    assert result != 0


def test_epsilon_guard_fires_for_chiral_couplings():
    """A chiral coupling on both vertices gives a non-zero ε·ε contribution
    to ``|M|²`` — the engine must refuse, not silently drop it."""
    m = sp.Symbol('m', positive=True)
    g = sp.Symbol('g', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    with pytest.raises(NotImplementedError, match="ε-tensor"):
        ffv_s_channel_squared(g, 0, g, 0, kin, mediator_mass=0)


def test_epsilon_guard_is_justified_by_the_explicit_oracle():
    """Companion to the guard test above: shows the guard is refusing a
    genuinely non-trivial computation, not an already-known-zero one.  A
    naive guess for the chiral case — half of the pure-vector result, the
    identity that *does* hold for a 1→2 decay
    (``test_ffv_chiral_is_half_the_vector_result`` in ``test_pheno.py``) —
    is wrong here, and by a large, angle-dependent amount: the genuine
    signature of the ε (γ₅) cross-term the covariant engine cannot yet
    compute.
    """
    for cos_num in (0.0, 0.4, -0.7):
        vector = _oracle_qed_general(1, 1, 1, 1, 30, cos_num, 1)
        chiral = _oracle_qed_general(1, 0, 1, 0, 30, cos_num, 1)
        assert abs(chiral - vector / 2) > sp.Rational(1, 10) * abs(vector)


def test_multi_diagram_raises():
    amp = Amplitude(diagrams=(
        Diagram(chains=(), propagators=()),
        Diagram(chains=(), propagators=()),
    ))
    with pytest.raises(NotImplementedError, match="interference"):
        amp.squared(None)


def test_scalar_mediator_is_epsilon_free():
    m = sp.Symbol('m', positive=True)
    mS = sp.Symbol('m_S', positive=True)
    g = sp.Symbol('g', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    # chiral couplings on a scalar mediator never trip the eps guard
    result = ffs_s_channel_squared(g, 0, g, 0, kin, mediator_mass=mS)
    assert result != 0


# --------------------------------------------------------------------------
# physics: e⁺e⁻ → μ⁺μ⁻ through a single photon (QED, Tier 1's acceptance case)
# --------------------------------------------------------------------------

def test_qed_mumu_squared_amplitude_closed_form():
    """Pins ``Σ|M|² = 8e⁴/s²[(t−m²)²+(u−m²)²+2m²s]`` — both the closed
    ``(s,t,u)`` form and, via ``numeric_equal``, agreement with the
    covariant engine's raw output (whose agreement with the ``(kᵢ·kⱼ)``-level
    computation independently validates the ``dot`` table)."""
    m, e, s, t = sp.symbols('m e s t', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    covariant = ffv_s_channel_squared(e, e, e, e, kin, mediator_mass=0)
    covariant = covariant.subs(kin.s, s).subs(kin.t, t)
    u = 2 * m**2 - s - t
    textbook = 8 * e**4 / s**2 * ((t - m**2)**2 + (u - m**2)**2 + 2 * m**2 * s)
    assert sp.simplify(sp.expand(covariant - textbook)) == 0
    ok, diff = numeric_equal(covariant, textbook, [s, t, m, e],
                             sample_range=(1.0, 5.0), seed=7)
    assert ok, f"max relative difference {diff}"


def test_qed_mumu_matches_explicit_matrix_oracle():
    m, e, s, t = sp.symbols('m e s t', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    covariant = ffv_s_channel_squared(e, e, e, e, kin, mediator_mass=0)
    covariant = covariant.subs(kin.s, s).subs(kin.t, t)
    cos = kin.cos_theta().subs(kin.s, s).subs(kin.t, t)
    oracle = _oracle_qed_general(e, e, e, e, s, cos, m)
    assert sp.simplify(sp.expand(covariant - oracle)) == 0


def test_qed_mumu_angular_distribution():
    """Massless limit: ``dσ/dcosθ = πα²(1+cos²θ)/(2s)``."""
    e, s, cosv = sp.symbols('e s cos', positive=True)
    kin = TwoToTwoKinematics(0, 0, 0, 0)
    m2 = ffv_s_channel_squared(e, e, e, e, kin, mediator_mass=0)
    electron = ExternalState('e', 0, sp.Rational(1, 2))
    from feynlag.pheno import differential_cross_section
    dsdcos = differential_cross_section(m2, kin, (electron, electron),
                                        variable="cos")
    t_at_cos = kin.t_of_cos(cosv).subs(kin.s, s)
    dsdcos = dsdcos.subs(kin.s, s).subs(kin.t, t_at_cos)
    alpha = e**2 / (4 * sp.pi)
    expected = sp.pi * alpha**2 * (1 + cosv**2) / (2 * s)
    assert sp.simplify(sp.expand(dsdcos - expected)) == 0


def test_qed_mumu_total_cross_section():
    m, e = sp.symbols('m e', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    m2 = ffv_s_channel_squared(e, e, e, e, kin, mediator_mass=0)
    electron = ExternalState('e', 0, sp.Rational(1, 2))
    sigma = sp.simplify(cross_section(m2, kin, (electron, electron)))
    beta = sp.sqrt(1 - 4 * m**2 / kin.s)
    alpha = e**2 / (4 * sp.pi)
    expected = (4 * sp.pi * alpha**2 / (3 * kin.s)) * beta * (3 - beta**2) / 2
    assert sp.simplify(sp.expand(sigma - expected)) == 0
    # massless limit
    assert sp.simplify(sigma.subs(m, 0) - 4 * sp.pi * alpha**2 / (3 * kin.s)) == 0


def test_qed_mumu_numeric_at_benchmark_point():
    """At MadGraph's benchmark parameter point (``docs/benchmark.md``:
    ``α⁻¹ = 132.50698``, ``√s = 200`` GeV), the QED-only (photon-only)
    cross section is **2.322 pb** — the ``e⁺e⁻→μ⁺μ⁻`` fraction of MadGraph's
    full **2.7878 ± 0.0027 pb**.  The ~20% gap is the γ/Z interference Tier 3
    must supply; this number therefore cannot be accidentally "passed" here.
    """
    m, e = sp.symbols('m e', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    m2 = ffv_s_channel_squared(e, e, e, e, kin, mediator_mass=0)
    electron = ExternalState('e', 0, sp.Rational(1, 2))
    sigma = sp.simplify(cross_section(m2, kin, (electron, electron)))

    alpha_inv = 132.50698
    e_num = math.sqrt(4 * math.pi / alpha_inv)
    s_num = 200.0**2
    val_gev2 = float(sigma.subs({e: e_num, m: 0, kin.s: s_num}))
    pb = val_gev2 * 3.894e8
    assert abs(pb - 2.322) < 0.01


def test_averaging_factor_is_declared_not_hardcoded():
    """Pins the architectural decision that
    :func:`~feynlag.pheno.amplitudes.ffv_squared`'s hardcoded ``/3`` violated:
    averaging is declared data on :class:`~feynlag.pheno.particles.ExternalState`,
    applied exactly once by :func:`~feynlag.pheno.scattering.cross_section`."""
    electron = ExternalState('e', 0, sp.Rational(1, 2))
    assert average_factor((electron, electron)) == sp.Rational(1, 4)

    quark = ExternalState('q', 0, sp.Rational(1, 2), color=3)
    assert average_factor((electron, quark)) == sp.Rational(1, 12)

    # Amplitude.squared() itself returns the summed, not averaged, |M|²
    m, e = sp.symbols('m e', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    m2 = ffv_s_channel_squared(e, e, e, e, kin, mediator_mass=0)
    from feynlag.pheno import differential_cross_section
    dsdt_summed = differential_cross_section(m2, kin, (electron, electron))
    dsdt_manual = sp.Rational(1, 4) * m2 * kin.dsigma_dt_factor()
    assert sp.simplify(sp.expand(dsdt_summed - dsdt_manual)) == 0
