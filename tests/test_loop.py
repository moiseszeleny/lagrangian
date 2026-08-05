r"""Loop-induced Higgs decays — Tier 3 of the decays roadmap.

House style: pin physics against the PDG.  The widths are effective-vertex
computations (the roadmap's documented ethos exception), so the acceptance
oracle is the measured SM value, and every closed form was verified numerically
before being written down (see the module docstring's citations).
"""

import math

import pytest

from feynlag.pheno import (
    A_half, A_one, higgs_gammagamma_width, higgs_gg_width, higgs_zgamma_width,
)
from feynlag.pheno.loop import A_half_zgamma, A_one_zgamma

# SM parameter point
MH, MT, MW, MZ, V = 125.25, 172.76, 80.377, 91.1876, 246.0
ALPHA, ALPHA_S = 1 / 137.036, 0.1126
SW2 = 1 - MW**2 / MZ**2


# --------------------------------------------------------------------------
# form-factor limits — independent of the SM point
# --------------------------------------------------------------------------

def test_form_factor_decoupling_limits():
    """As τ→0 (a very heavy loop particle) the form factors approach their
    textbook constants: A_{1/2}→4/3, A_1→−7."""
    assert abs(A_half(1e-8).real - 4 / 3) < 1e-4
    assert abs(A_one(1e-8).real - (-7)) < 1e-4


def test_form_factors_real_above_threshold():
    """For the physical top and W (τ<1) the loop functions are real — no
    absorptive part, since the Higgs cannot produce the loop particle on
    shell."""
    tau_t = MH**2 / (4 * MT**2)
    tau_W = MH**2 / (4 * MW**2)
    assert abs(A_half(tau_t).imag) < 1e-12
    assert abs(A_one(tau_W).imag) < 1e-12


# --------------------------------------------------------------------------
# h → γγ
# --------------------------------------------------------------------------

def test_higgs_gammagamma_width():
    """Γ(h→γγ) ≈ 9.3 keV (PDG)."""
    g = higgs_gammagamma_width(MH, MT, MW, V, ALPHA)
    assert abs(g * 1e6 - 9.3) < 0.3          # keV, ~3%


def test_gammagamma_destructive_interference():
    """The W loop dominates and the top loop has the opposite sign, so they
    interfere *destructively* — the total amplitude is smaller in magnitude
    than the W piece alone.  A real physics check, not just a magnitude."""
    tau_W = MH**2 / (4 * MW**2)
    tau_t = MH**2 / (4 * MT**2)
    a_W = A_one(tau_W).real                 # negative
    a_t = 3 * (2 / 3)**2 * A_half(tau_t).real  # positive
    assert a_W < 0 < a_t
    assert abs(a_W + a_t) < abs(a_W)        # destructive


# --------------------------------------------------------------------------
# h → gg (LO and NLO)
# --------------------------------------------------------------------------

def test_higgs_gg_width_lo():
    """Γ(h→gg) at leading order ≈ 0.20 MeV (the exact one-loop form factor)."""
    g = higgs_gg_width(MH, MT, V, ALPHA_S, qcd=False)
    assert abs(g * 1e3 - 0.20) < 0.02        # MeV


def test_higgs_gg_width_nlo_kfactor():
    """The NLO-QCD K-factor (~1.64) lifts the LO width to the measured
    ~0.34 MeV (the 8.2% branching ratio)."""
    lo = higgs_gg_width(MH, MT, V, ALPHA_S, qcd=False)
    nlo = higgs_gg_width(MH, MT, V, ALPHA_S, qcd=True)
    K = nlo / lo
    assert 1.5 < K < 1.8                      # ≈ 1.64
    assert abs(nlo * 1e3 - 0.33) < 0.03       # MeV


# --------------------------------------------------------------------------
# h → Zγ
# --------------------------------------------------------------------------

def test_higgs_zgamma_width():
    """Γ(h→Zγ) ≈ 6.3 keV (PDG)."""
    g = higgs_zgamma_width(MH, MT, MW, MZ, V, ALPHA, SW2)
    assert abs(g * 1e6 - 6.3) < 0.3          # keV, ~5%


# --------------------------------------------------------------------------
# calculator wiring + the complete BR picture
# --------------------------------------------------------------------------

def test_calculator_loop_widths():
    """`DecayCalculator.loop_widths` returns the three channels at their PDG
    values."""
    from feynlag.pheno import DecayCalculator
    calc = DecayCalculator.__new__(DecayCalculator)   # no model needed
    w = calc.loop_widths(MH, MT, MW, MZ, V, ALPHA, ALPHA_S)
    assert abs(w[("gamma", "gamma")] * 1e6 - 9.3) < 0.3
    assert abs(w[("g", "g")] * 1e3 - 0.33) < 0.03      # qcd=True default
    assert abs(w[("Z", "gamma")] * 1e6 - 6.3) < 0.3


def test_complete_higgs_br_shape():
    """Tree + off-shell + loop reproduces the canonical Higgs BR shape: b̄b
    dominant, then WW*, gg ~8%, γγ ~0.2%."""
    # loop widths
    Ggg = higgs_gg_width(MH, MT, V, ALPHA_S, qcd=True)
    Ggaga = higgs_gammagamma_width(MH, MT, MW, V, ALPHA)
    GZga = higgs_zgamma_width(MH, MT, MW, MZ, V, ALPHA, SW2)
    # representative tree + off-shell widths (in GeV), from the sibling tiers
    widths = {
        "bb": 2.35e-3, "WW*": 0.80e-3, "tautau": 0.26e-3,
        "cc": 0.12e-3, "ZZ*": 0.089e-3,
        "gg": Ggg, "gammagamma": Ggaga, "Zgamma": GZga,
    }
    total = sum(widths.values())
    br = {k: v / total for k, v in widths.items()}
    assert br["bb"] == max(br.values())            # b̄b dominant
    assert 0.05 < br["gg"] < 0.11                  # gg ~8%
    assert 0.001 < br["gammagamma"] < 0.004        # γγ ~0.2%
    assert abs(sum(br.values()) - 1) < 1e-9


# ---------------------------------------------------------------------------
# The spin-0 (charged-scalar) form factor and the BSM-ready amplitude.
# ---------------------------------------------------------------------------

def test_scalar_form_factor_decoupling_limit():
    """A_0 -> -1/3, the sign forced by consistency with A_1 and A_{1/2}.

    [ChakrabartyEtAl21] Eq. (28) gives F_W -> +7, F_t -> -4/3, F+ -> +1/3 in
    the reciprocal variable.  feynlag has A_1 = -F_W and A_{1/2} = -F_t, so
    A_0 = -F+ is the only consistent choice -- an overall amplitude sign is
    convention, but the RELATIVE signs are physics.
    """
    from feynlag.pheno.loop import A_zero
    assert abs(A_zero(1e-8).real - (-1 / 3)) < 1e-6
    # relative to the other two, at the same (tiny) tau
    assert abs(A_half(1e-8).real - 4 / 3) < 1e-6
    assert abs(A_one(1e-8).real - (-7)) < 1e-4


def test_diphoton_amplitude_reproduces_the_sm_content():
    """The general assembler with SM content == the SM-only width."""
    from feynlag.pheno.loop import higgs_diphoton_amplitude
    m_h, m_t, m_W, v, alpha = 125.25, 172.69, 80.377, 246.0, 1 / 137.036
    amp = higgs_diphoton_amplitude(m_h, [(1, 1, m_W, 1, 1),
                                         (1, 0.5, m_t, 2 / 3, 3)])
    direct = A_one(m_h**2 / (4 * m_W**2)) + 3 * (2 / 3)**2 * A_half(
        m_h**2 / (4 * m_t**2))
    assert abs(amp - direct) < 1e-12


def test_charged_scalar_loop_suppresses_and_decouples():
    """A charged Higgs suppresses h->gamma gamma and decouples as m_H+ -> inf.

    With g_{hH+H-} = -m_h^2/v ([ChakrabartyEtAl21] Eq. 26) the coefficient
    kappa = g v / (2 m_H+^2) is negative, and A_0 < 0, so kappa*A_0 > 0 adds
    against the dominant A_1 = -7: the rate goes DOWN.  That is why light
    charged scalars are constrained by the diphoton measurement.
    """
    from feynlag.pheno.loop import higgs_gammagamma_width
    m_h, m_t, m_W, v, alpha = 125.25, 172.69, 80.377, 246.0, 1 / 137.036
    sm = higgs_gammagamma_width(m_h, m_t, m_W, v, alpha)

    def with_charged(m_hp):
        kappa = (-m_h**2 / v) * v / (2 * m_hp**2)
        return higgs_gammagamma_width(m_h, m_t, m_W, v, alpha,
                                      extra_loops=[(kappa, 0, m_hp, 1, 1)])

    assert with_charged(200.0) < with_charged(400.0) < sm      # suppression
    assert abs(with_charged(5000.0) / sm - 1) < 1e-3           # decoupling


def test_extra_loops_defaults_leave_the_sm_untouched():
    """The generalization must not move the PDG-pinned SM numbers."""
    from feynlag.pheno.loop import higgs_gammagamma_width
    m_h, m_t, m_W, v, alpha = 125.25, 172.69, 80.377, 246.0, 1 / 137.036
    got = higgs_gammagamma_width(m_h, m_t, m_W, v, alpha) * 1e6   # keV
    assert 8.5 < got < 10.0                                       # PDG ~9.3 keV


def test_diphoton_amplitude_rejects_unknown_spin():
    """A spin outside {0, 1/2, 1} raises rather than being silently dropped."""
    from feynlag.pheno.loop import higgs_diphoton_amplitude
    with pytest.raises(ValueError, match="form factor for spin"):
        higgs_diphoton_amplitude(125.0, [(1, 2, 100.0, 1, 1)])
