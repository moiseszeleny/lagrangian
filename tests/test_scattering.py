"""Tests for 2→2 scattering (`feynlag.pheno.diagrams`/`.scattering`).

House style: pin physics, not code paths.  Tiers 1–2 of
``docs/manual/scattering_roadmap.md``.  The acceptance oracle is the textbook
QED closed form for ``e⁺e⁻→μ⁺μ⁻`` through a single photon,
``σ = (4πα²/3s)·β(3−β²)/2``, ``β = √(1−4m_μ²/s)`` — together with an
independent **explicit-4×4-matrix** evaluator (``_oracle_qed_general`` below)
that shares no code with the covariant engine.  It carries γ₅ as a literal
matrix, which is what independently validates the ε (γ₅) term for a chiral
coupling directly (Tier 2), not just that it vanishes for a pure vector one
(Tier 1). Every physics assertion gets the house dual check:
``sp.simplify(sp.expand(a - b)) == 0`` and :func:`~feynlag.verify.numeric_equal`.
"""

import math

import pytest
import sympy as sp
from sympy.physics.hep.gamma_matrices import LorentzIndex

from feynlag.dirac import _dirac_rep
from feynlag.pheno import (
    Amplitude, BosonPropagator, ChainVertex, Diagram, ExternalState, Leg,
    SpinorChain, TwoToTwoKinematics, average_factor, cross_section,
    ffs_s_channel_squared, ffv_s_channel_squared, forward_backward_asymmetry,
)
from feynlag.pheno.diagrams import _chain_indices
from feynlag.pheno.epsilon import (
    assert_epsilon_single_vanishes, epsilon_pair_tensor, epsilon_product_sign,
    gamma5_trace_coefficient, gram_determinant, levi_civita_array,
)
from feynlag.pheno.lorentz import contract_to_dots, momentum
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


def _oracle_qed_general(gL_in, gR_in, gL_out, gR_out, s, cos, m, m_med=0, w_med=0):
    """``Σ|M|²`` for ``e⁺(m1=0)e⁻(m2=0) → f⁺(m3=m) f⁻(m4=m)`` through one
    s-channel vector, explicit Dirac matrices, general chiral couplings.

    Independent of :mod:`feynlag.pheno.diagrams`: builds ``Γ = g_LP_L+g_RP_R``
    and ``Γ̄`` as literal matrices (γ₅ included via the explicit ``P_L``/
    ``P_R`` projectors) and traces directly, rather than going through
    :func:`~feynlag.pheno.lorentz.reduce_projectors`.

    ``m_med=0`` (the default, massless mediator) uses the bare-metric
    contraction ``T_in^{μν}g_{μμ'}g_{νν'}T_out^{μ'ν'}`` — a simple
    element-wise sum since a diagonal metric makes it so.  A massive
    mediator (``m_med≠0``) instead needs the true rank-2 contraction against
    the non-diagonal numerator ``N_{μμ'} = g_{μμ'} − q_μq_{μ'}/m_med²``
    (``q = k1+k2``, lowered via the ``(+,−,−,−)`` metric), summed over all
    four indices — the minimal extension needed to exercise
    :func:`~feynlag.pheno.propagator.vector_propagator_numerator`'s sign
    convention against a genuinely chiral coupling; only one test
    (``test_z_massive_mediator_matches_explicit_matrix_oracle``) uses it, the
    massless oracle above stays the fast workhorse for everything else.
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

    if m_med == 0:
        total = 0
        for mu in range(4):
            for nu in range(4):
                total += T_in[mu][nu] * _MET[mu, mu] * _MET[nu, nu] * T_out[mu][nu]
        return sp.simplify(total / s**2)

    q_lower = [_MET[i, i] * (k1[i] + k2[i]) for i in range(4)]
    N = sp.zeros(4, 4)
    for a in range(4):
        for b in range(4):
            N[a, b] = (_MET[a, a] if a == b else 0) - q_lower[a] * q_lower[b] / m_med**2
    total = 0
    for mu in range(4):
        for mup in range(4):
            for nu in range(4):
                for nup in range(4):
                    total += T_in[mu][nu] * N[mu, mup] * T_out[mup][nup] * N[nu, nup]
    denom = (s - m_med**2)**2 + m_med**2 * w_med**2
    return sp.simplify(total / denom)


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
# ε (γ₅) algebra primitives (Tier 2, `feynlag.pheno.epsilon`)
# --------------------------------------------------------------------------

def test_gamma5_trace_coefficient_is_derived_from_the_explicit_rep():
    """``κ = -4i`` — and it is not an assumed constant: ``Tr[γ^aγ^bγ^cγ^dγ₅]
    == κ·ε^{abcd}`` holds for every one of the 256 index tuples, checked
    against the literal 4×4 matrices in ``_dirac_rep()``, independently of
    ``gamma5_trace_coefficient``'s own derivation."""
    kappa = gamma5_trace_coefficient()
    assert kappa == -4 * sp.I

    rep = _dirac_rep()
    g = [rep[("g", m)] for m in range(4)]
    g5 = rep["g5"]
    LC = levi_civita_array()
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    trace = (g[a] * g[b] * g[c] * g[d] * g5).trace()
                    assert sp.simplify(trace - kappa * LC[a, b, c, d]) == 0


def test_epsilon_product_sign_from_explicit_arrays():
    """``s_det = -1`` — cross-checked by ``ε^{abcd}ε_{abcd} == -24`` computed
    directly (lowering all four indices of the totally antisymmetric array
    with the ``(+,-,-,-)`` metric multiplies every non-zero component by
    ``det(metric) = -1``, so the self-contraction is ``-1 · 4! = -24``)."""
    assert epsilon_product_sign() == -1

    LC = levi_civita_array()
    total = sp.S.Zero
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    coeff = LC[a, b, c, d]
                    if coeff == 0:
                        continue
                    lower = _MET[a, a] * _MET[b, b] * _MET[c, c] * _MET[d, d] * coeff
                    total += coeff * lower
    assert total == -24


def test_epsilon_pair_tensor_reproduces_the_gram_determinant():
    """All-momentum slots reduce ``ε^{a…}ε^{b…}`` to ``-det[p_i·p'_j]`` — the
    roadmap's stated Gram-determinant identity, verified two ways: against
    the symbolic Gram determinant built from ``kin.dot``, and against an
    explicit numeric Levi-Civita contraction of CM-frame four-vectors
    (``eps(p1,p2,p3,p4)^2 == -det[Gram]`` via ``det(Gram) = det(metric)
    ·det(components)^2``)."""
    m1, m2, m3, m4 = sp.symbols('m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    heads = [kin.k1, kin.k2, kin.k3, kin.k4]
    slots = tuple(('m', h) for h in heads)
    ee = epsilon_pair_tensor(slots, slots).contract_metric(LorentzIndex.metric)
    val = contract_to_dots(ee, kin.dot)
    gram = gram_determinant(heads, kin.dot)
    assert sp.simplify(sp.expand(val - (-gram))) == 0

    s_num, t_num = sp.Rational(50), sp.Rational(-7)
    m1n, m2n, m3n, m4n = sp.Integer(0), sp.Integer(0), sp.Integer(2), sp.Integer(2)
    subs = {kin.s: s_num, kin.t: t_num, m1: m1n, m2: m2n, m3: m3n, m4: m4n}
    cos = kin.cos_theta().subs(subs)
    k1n, k2n, k3n, k4n = _cm_frame(s_num, cos, m1n, m2n, m3n, m4n)
    LC = levi_civita_array()
    eps_val = sp.S.Zero
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    coeff = LC[a, b, c, d]
                    if coeff == 0:
                        continue
                    eps_val += coeff * k1n[a] * k2n[b] * k3n[c] * k4n[d]
    gram_num = gram.subs(subs)
    assert sp.simplify(eps_val**2 - (-gram_num)) == 0


def test_single_epsilon_gram_determinant_vanishes_for_two_to_two():
    """A 2→2 diagram's four external momenta satisfy ``k4=k1+k2-k3``, so
    their Gram determinant vanishes identically for symbolic masses — the
    computed fact :func:`assert_epsilon_single_vanishes` relies on."""
    m1, m2, m3, m4 = sp.symbols('m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    heads = [kin.k1, kin.k2, kin.k3, kin.k4]
    det = sp.simplify(gram_determinant(heads, kin.dot))
    assert det == 0


def test_single_epsilon_guard_raises_for_four_independent_momenta():
    """The guard isn't vacuous: four genuinely independent momenta (a
    synthetic orthonormal ``dot`` table) raise, while a real 2→2 diagram's
    four momenta (rank 3) do not."""
    class _Head:
        def __init__(self, name):
            self.name = name

    heads = [_Head(f"q{i}") for i in range(4)]
    table = {(heads[i].name, heads[j].name): sp.Integer(1 if i == j else 0)
             for i in range(4) for j in range(4)}

    def dot(a, b):
        return table[(a.name, b.name)]

    with pytest.raises(NotImplementedError, match="Gram determinant"):
        assert_epsilon_single_vanishes(heads, dot)

    m1, m2, m3, m4 = sp.symbols('m1 m2 m3 m4', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    assert_epsilon_single_vanishes([kin.k1, kin.k2, kin.k3, kin.k4], kin.dot)


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


def test_chiral_squared_amplitude_matches_explicit_matrix_oracle():
    """Tier 2's headline acceptance test.  With FULLY SYMBOLIC, independent
    chiral couplings on both vertices (not the pure-vector ``g_L=g_R`` case,
    where the ε term is trivially zero) the covariant engine's ``Σ|M|²``
    matches the independent explicit-4×4-matrix oracle exactly.  This single
    identity fixes ``κ`` (:func:`~feynlag.pheno.epsilon.gamma5_trace_coefficient`),
    the ``∓`` sign in the ``P_{L,R}`` split, and the ``ε_{0123}`` convention
    simultaneously, and — because the couplings are independent, not
    related by any special relation — its generality is what shows the
    single-ε cross terms genuinely cancel, not just for one lucky point.
    """
    m, s, t = sp.symbols('m s t', positive=True)
    gL, gR, hL, hR = sp.symbols('g_L g_R h_L h_R', real=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    covariant = ffv_s_channel_squared(gL, gR, hL, hR, kin, mediator_mass=0)
    covariant = covariant.subs(kin.s, s).subs(kin.t, t)
    cos = kin.cos_theta().subs(kin.s, s).subs(kin.t, t)
    oracle = _oracle_qed_general(gL, gR, hL, hR, s, cos, m)
    assert sp.simplify(sp.expand(covariant - oracle)) == 0


def test_chiral_result_is_not_half_the_vector_result_and_engine_gets_it_right():
    """What Tier 1 could only refuse, Tier 2 computes: a naive guess for the
    chiral case — half of the pure-vector result, the identity that *does*
    hold for a 1→2 decay (``test_ffv_chiral_is_half_the_vector_result`` in
    ``test_pheno.py``) — is wrong here, and by a large, angle-dependent
    amount, the genuine signature of the ε (γ₅) cross-term.  The covariant
    engine matches the independent oracle at the same points instead of
    guessing or refusing.
    """
    m = sp.Symbol('m', positive=True)
    g = sp.Symbol('g', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    covariant_chiral = ffv_s_channel_squared(g, 0, g, 0, kin, mediator_mass=0)
    for cos_num in (0.0, 0.4, -0.7):
        vector = _oracle_qed_general(1, 1, 1, 1, 30, cos_num, 1)
        chiral = _oracle_qed_general(1, 0, 1, 0, 30, cos_num, 1)
        assert abs(chiral - vector / 2) > sp.Rational(1, 10) * abs(vector)

        t_num = kin.t_of_cos(cos_num).subs(kin.s, 30).subs(m, 1)
        engine = covariant_chiral.subs({kin.s: 30, kin.t: t_num, g: 1, m: 1})
        assert abs(complex(sp.simplify(engine - chiral))) < 1e-9


def test_pure_LL_squared_amplitude_closed_form():
    """``Σ|M|² = 4(u−m²)²/s²`` for a purely left-handed coupling on both
    vertices — the textbook helicity decomposition of the vector-coupling
    result ``8(t²+u²)/s²`` (at ``m=0``, ``4u²/s²``), and this engine's first
    genuinely non-zero use of the ε (γ₅) term (a pure chiral coupling has
    the largest possible ε contribution)."""
    m, s, t = sp.symbols('m s t', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    covariant = ffv_s_channel_squared(1, 0, 1, 0, kin, mediator_mass=0)
    covariant = covariant.subs(kin.s, s).subs(kin.t, t)
    u = 2 * m**2 - s - t
    expected = 4 * (u - m**2)**2 / s**2
    assert sp.simplify(sp.expand(covariant - expected)) == 0
    assert sp.simplify(covariant.subs(m, 0) - 4 * (s + t)**2 / s**2) == 0


def test_massive_mediator_epsilon_piece_drops_the_qq_term():
    """The massive propagator's ``q_μq_ν/M²`` numerator term drops out of
    the ε contribution automatically: ``q=k1+k2`` puts a repeated momentum
    into the same ε, giving two equal Gram-determinant rows ⟹ 0 — checked
    directly on :func:`~feynlag.pheno.epsilon.epsilon_pair_tensor`,
    independent of any chain/vertex machinery, with all four external masses
    symbolic."""
    from feynlag.pheno.propagator import vector_propagator_numerator

    m1, m2, m3, m4, M = sp.symbols('m1 m2 m3 m4 M', positive=True)
    kin = TwoToTwoKinematics(m1, m2, m3, m4)
    in_idx, in_bidx = _chain_indices("in")
    out_idx, out_bidx = _chain_indices("out")
    slots_a = (('m', kin.k2), ('i', in_idx), ('m', kin.k1), ('i', in_bidx))
    slots_b = (('m', kin.k3), ('i', out_idx), ('m', kin.k4), ('i', out_bidx))
    ee = epsilon_pair_tensor(slots_a, slots_b)

    q = lambda i: kin.k1(i) + kin.k2(i)
    n1_massive = vector_propagator_numerator(q, M, in_idx, out_idx)
    n2_massive = vector_propagator_numerator(q, M, in_bidx, out_bidx)
    n1_bare = LorentzIndex.metric(-in_idx, -out_idx)
    n2_bare = LorentzIndex.metric(-in_bidx, -out_bidx)

    massive = (ee * n1_massive).contract_metric(LorentzIndex.metric).expand()
    massive = (massive * n2_massive).contract_metric(LorentzIndex.metric)
    bare = (ee * n1_bare).contract_metric(LorentzIndex.metric).expand()
    bare = (bare * n2_bare).contract_metric(LorentzIndex.metric)

    val_massive = contract_to_dots(massive, kin.dot)
    val_bare = contract_to_dots(bare, kin.dot)
    assert sp.simplify(sp.expand(val_massive - val_bare)) == 0


def test_z_massive_mediator_matches_explicit_matrix_oracle():
    """The one test needing the massive-mediator oracle extension: a
    genuinely chiral coupling through a massive vector matches the explicit
    matrix computation exactly — checking that
    :func:`~feynlag.pheno.propagator.vector_propagator_numerator`'s sign
    convention composes correctly with the ε piece.  The ``qq/M²`` term's
    unique contribution is already proven to drop for equal external masses
    (``test_massive_mediator_epsilon_piece_drops_the_qq_term``); this is the
    end-to-end check that nothing about combining it with a real chiral
    coupling introduces a sign mismatch.
    """
    m, e, s, t, M = sp.symbols('m e s t M', positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    covariant = ffv_s_channel_squared(e, 0, e, 0, kin, mediator_mass=M)
    covariant = covariant.subs(kin.s, s).subs(kin.t, t)
    cos = kin.cos_theta().subs(kin.s, s).subs(kin.t, t)
    oracle = _oracle_qed_general(e, 0, e, 0, s, cos, m, m_med=M)
    assert sp.simplify(sp.expand(covariant - oracle)) == 0


def test_reduce_projectors_is_no_longer_on_the_chain_path():
    """The chain-level engine computes the ε term directly now (Tier 2)
    rather than proving it away via
    :func:`~feynlag.pheno.lorentz.reduce_projectors` — confirmed by
    ``diagrams.py`` not even importing it any more.  It stays frozen as the
    1→2-only helper ``amplitudes.ffs_squared``/``ffv_squared`` use (see
    ``test_pheno.py::test_gamma5_guard_raises_outside_two_body``)."""
    import feynlag.pheno.diagrams as diagrams_mod
    assert not hasattr(diagrams_mod, "reduce_projectors")


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


# --------------------------------------------------------------------------
# physics: e⁺e⁻ → f f̄ through the Z alone (Tier 2's chiral acceptance case)
# --------------------------------------------------------------------------

def test_z_only_angular_distribution_matches_lep_born_form():
    """Massless final state through a purely chiral (Z-like) mediator:
    ``Σ|M|² = (g_L²+g_R²)(h_L²+h_R²)(1+cos²θ) + 2(g_L²−g_R²)(h_L²−h_R²)cosθ``
    — the LEP Born-level angular distribution, Eq. (1.55) of [LEPEWWG06]
    (``dσ/dcosθ ∝ (1+cos²θ)+2A_eA_f cosθ`` at zero beam polarization, with
    ``A_f=(g_L²−g_R²)/(g_L²+g_R²)`` its Eq. (1.56) — the two forms are
    algebraically identical once the ``(g_L²+g_R²)(h_L²+h_R²)`` denominators
    in ``A_eA_f`` are cleared)."""
    gL, gR, hL, hR = sp.symbols('g_L g_R h_L h_R', real=True)
    s, cosv = sp.symbols('s cos', real=True)
    kin = TwoToTwoKinematics(0, 0, 0, 0)
    m2 = ffv_s_channel_squared(gL, gR, hL, hR, kin, mediator_mass=0)
    t_at_cos = kin.t_of_cos(cosv).subs(kin.s, s)
    m2 = m2.subs(kin.s, s).subs(kin.t, t_at_cos)
    expected = ((gL**2 + gR**2) * (hL**2 + hR**2) * (1 + cosv**2)
               + 2 * (gL**2 - gR**2) * (hL**2 - hR**2) * cosv)
    assert sp.simplify(sp.expand(m2 - expected)) == 0


def test_z_only_forward_backward_asymmetry_is_three_quarters_Ae_Af():
    """``A_FB = (3/4)A_eA_f`` — Eq. (1.66) of [LEPEWWG06] — checked
    symbolically, and pinned at the numeric LEP effective weak mixing angle
    ``sin²θ_eff^lept = 0.23153`` (giving the standard leptonic
    ``A_ℓ ≈ 0.147`` and ``A_FB^{0,ℓ} ≈ 0.016``)."""
    gL, gR, hL, hR = sp.symbols('g_L g_R h_L h_R', positive=True)
    kin = TwoToTwoKinematics(0, 0, 0, 0)
    m2 = ffv_s_channel_squared(gL, gR, hL, hR, kin, mediator_mass=0)
    afb = forward_backward_asymmetry(m2, kin)
    Ae = (gL**2 - gR**2) / (gL**2 + gR**2)
    Af = (hL**2 - hR**2) / (hL**2 + hR**2)
    expected = sp.Rational(3, 4) * Ae * Af
    assert sp.simplify(afb - expected) == 0

    sin2w = 0.23153
    T3, Q = -0.5, -1.0
    gL_num = T3 - Q * sin2w
    gR_num = -Q * sin2w
    Al_num = (gL_num**2 - gR_num**2) / (gL_num**2 + gR_num**2)
    assert abs(Al_num - 0.147) < 0.001
    afb_num = sp.Rational(3, 4) * Al_num * Al_num
    assert abs(afb_num - 0.0162) < 0.001


def test_z_only_total_cross_section_is_epsilon_independent():
    """The ε (γ₅) term is odd in ``cosθ`` and integrates to zero over the
    full range, so a chiral mediator's total cross section equals that of a
    pure-vector coupling with the same ``g_L²+g_R²`` — Tier 1's 2.322 pb
    QED benchmark cannot shift from Tier 2 landing; ``A_FB`` is an angular
    observable only."""
    gL, gR, hL, hR = sp.symbols('g_L g_R h_L h_R', positive=True)
    kin = TwoToTwoKinematics(0, 0, 0, 0)
    electron = ExternalState('e', 0, sp.Rational(1, 2))
    chiral_m2 = ffv_s_channel_squared(gL, gR, hL, hR, kin, mediator_mass=0)
    sigma_chiral = sp.simplify(cross_section(chiral_m2, kin, (electron, electron)))

    ge = sp.sqrt((gL**2 + gR**2) / 2)
    hf = sp.sqrt((hL**2 + hR**2) / 2)
    vector_m2 = ffv_s_channel_squared(ge, ge, hf, hf, kin, mediator_mass=0)
    sigma_vector = sp.simplify(cross_section(vector_m2, kin, (electron, electron)))
    assert sp.simplify(sp.expand(sigma_chiral - sigma_vector)) == 0
