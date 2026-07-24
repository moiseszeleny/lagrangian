"""Off-shell (1→3) VV* decays — Tier 2 of the decays roadmap.

House style: pin physics.  The acceptance oracle is the Keung–Marciano /
Djouadi closed form for $h\\to VV^*$,
$\\Gamma=\\frac{3 g_V^4 m_h}{512\\pi^3}\\delta_V R(x)$, $x=m_V^2/m_h^2$, which the
covariant $|M|^2$ + Dalitz integral must reproduce.
"""

import numpy as np
import pytest
import sympy as sp

from feynlag.pheno import (
    ThreeBodyKinematics, breit_wigner, offshell_scalar_vv_width,
    scalar_offshell_vv_width, scalar_vv_squared,
)

# parameter point (matches examples/sm_decays.py)
MH, MW, MZ, GW, GZ, G, GP, VEV = (125.25, 80.377, 91.1876, 2.085, 2.4952,
                                  0.6535, 0.3580, 246.0)
GZc = np.sqrt(G**2 + GP**2)
SW2 = GP**2 / (G**2 + GP**2)


def keung_marciano_R(x):
    """The Djouadi $R(x)$ phase-space function, $x=m_V^2/m_h^2$."""
    return (3 * (1 - 8*x + 20*x**2) / np.sqrt(4*x - 1)
            * np.arccos((3*x - 1) / (2*x**1.5))
            - (1 - x) / (2*x) * (2 - 13*x + 47*x**2)
            - 1.5 * (1 - 6*x + 4*x**2) * np.log(x))


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def test_breit_wigner_peaks_on_shell():
    """|BW|² is maximal at q² = m² and falls as (q²−m²)⁻² far off-shell."""
    m, w = 80.0, 2.0
    on = float(breit_wigner(m**2, m, w))
    off = float(breit_wigner((m + 40)**2, m, w))
    assert on > off
    assert abs(on - 1 / (m**2 * w**2)) < 1e-9 * on          # peak height


def test_scalar_vv_squared_closed_form():
    """The covariant assembly reduces to the expected polynomial in the Dalitz
    invariants (a closed form, no free tensor indices)."""
    mS, mV = sp.symbols("m_S m_V", positive=True)
    kin = ThreeBodyKinematics(mS, mV, sp.S.Zero, sp.S.Zero)
    m2 = scalar_vv_squared(mV, kin)
    s12, s23 = kin.s12, kin.s23
    expected = (mS**2*s12 + mV**2*(-mS**2 + s12 + 2*s23) - s12**2 - s12*s23) / mV**2
    assert sp.simplify(m2 - expected) == 0


def test_three_body_dalitz_bounds():
    """s₁₂ bounds are real and ordered inside the s₂₃ range."""
    kin = ThreeBodyKinematics(sp.Float(MH), sp.Float(MW), sp.S.Zero, sp.S.Zero)
    lo23, hi23 = (float(b) for b in kin.s23_range())
    for s23 in np.linspace(lo23 + 1, hi23 - 1, 5):
        lo, hi = (float(b) for b in kin.s12_bounds(s23))
        assert lo < hi


# --------------------------------------------------------------------------
# the physics: h → WW* and h → ZZ* against Keung–Marciano
# --------------------------------------------------------------------------

def test_higgs_ww_star_width():
    """Γ(h→WW*) reproduces the Keung–Marciano value ≈ 0.80 MeV.

    9 fermion channels (3 lepton + 2 quark generations × 3 colour), each with
    the V−A coupling g/√2, and a factor 2 for which W is on-shell.
    """
    channels = [(G / np.sqrt(2), 0.0, 9)]
    width = scalar_offshell_vv_width(MH, MW, GW, G * MW, channels,
                                     identical=False, backend="gauss")
    km = 3 * G**4 * MH / (512 * np.pi**3) * 1.0 * keung_marciano_R(MW**2 / MH**2)
    assert abs(width - km) / km < 0.02
    assert abs(width * 1e3 - 0.80) < 0.03          # ~0.80 MeV


def test_higgs_zz_star_width():
    """Γ(h→ZZ*) reproduces the Keung–Marciano value ≈ 0.089 MeV.

    Real chiral Z→ff couplings summed over all fermions; identical Z's, so no
    factor 2.
    """
    channels = []
    for T3, Q, Nc, cnt in [(0.5, 0, 1, 3), (-0.5, -1, 1, 3),
                           (0.5, 2/3, 3, 2), (-0.5, -1/3, 3, 3)]:
        channels.append((GZc * (T3 - Q * SW2), GZc * (-Q * SW2), Nc * cnt))
    width = scalar_offshell_vv_width(MH, MZ, GZ, GZc * MZ, channels,
                                     identical=True, backend="gauss")
    delta_Z = 7/12 - 10/9 * SW2 + 40/27 * SW2**2
    km = (3 * GZc**4 * MH / (512 * np.pi**3) * delta_Z
          * keung_marciano_R(MZ**2 / MH**2))
    assert abs(width - km) / km < 0.03
    assert abs(width * 1e3 - 0.089) < 0.005         # ~0.089 MeV


def test_width_negligibly_dependent_on_gamma_v_below_threshold():
    """Below threshold q² never reaches m_V², so the resonance width drops out
    of the propagator — Γ_V = 0 and the physical Γ_W agree."""
    c = float((G * MW)**2 * (G / np.sqrt(2))**2)
    g0 = offshell_scalar_vv_width(MH, MW, 0.0, c, backend="gauss")
    gw = offshell_scalar_vv_width(MH, MW, GW, c, backend="gauss")
    assert abs(g0 - gw) / gw < 1e-3


def test_narrow_width_factorisation():
    """For a heavy scalar (m_S > 2m_V, both V on-shell) the 1→3 width →
    Γ(S→VV)·BR(V→ff) — the assembly reduces to the Tier-1 on-shell result."""
    mS = 400.0                       # well above 2 m_W
    # single channel via the off-shell integral
    c = float((G * MW)**2 * (G / np.sqrt(2))**2)
    g3 = offshell_scalar_vv_width(mS, MW, GW, c, backend="gauss")
    # narrow-width expectation: Γ(S→WW on-shell) × BR(W→ this channel)
    # Γ(S→WW) with hWW=g m_W coupling (VVS), and BR = Γ(W→ffbar')/Γ_W
    from feynlag.pheno import TwoBodyKinematics, vvs_squared
    kin = TwoBodyKinematics(sp.Float(mS), sp.Float(MW), sp.Float(MW))
    gWW = float(sp.simplify(vvs_squared(G * MW, kin) * kin.phase_space()))
    gW_chan = G**2 * MW / (48 * np.pi)               # Γ(W→one massless channel)
    expected = gWW * gW_chan / GW
    assert abs(g3 - expected) / expected < 0.05


@pytest.mark.skipif(not __import__("feynlag").pheno.have_scipy(),
                    reason="SciPy not installed")
def test_scipy_matches_gauss():
    """The SciPy and numpy Gauss–Legendre backends agree."""
    c = float((G * MW)**2 * (G / np.sqrt(2))**2)
    gg = offshell_scalar_vv_width(MH, MW, GW, c, backend="gauss")
    gs = offshell_scalar_vv_width(MH, MW, GW, c, backend="scipy")
    assert abs(gg - gs) / gs < 1e-3


def test_calculator_offshell_wiring():
    """`DecayCalculator.offshell_vv_width` extracts the hWW coupling from the
    model and reproduces Γ(h→WW*), proving the end-to-end wiring."""
    import importlib.util
    import pathlib
    from feynlag.pheno import DecayCalculator
    path = pathlib.Path(__file__).parent.parent / "examples" / "sm_decays.py"
    spec = importlib.util.spec_from_file_location("sm_decays_off", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.build_model()

    mZ, mW, mh, mt = sp.symbols("m_Z m_W m_h m_t", positive=True)
    masses = {s["Z"]: mZ, s["Wp"]: mW, s["Wm"]: mW, s["A"]: 0, s["h"]: mh,
              s["tau"]: mt, s["taubar"]: mt, s["nu"]: 0, s["nubar"]: 0}
    calc = DecayCalculator(s["model"], masses, boson_fields=s["bosons"],
                           fermion_sectors=("gauge", "yukawa"),
                           conjugate_map=s["conjugate_map"],
                           particle_map=s["particle_map"],
                           parameters=s["model"].parameters)
    # the extracted hWW coupling is g²v/2 = g·m_W(tree) ≈ g·m_W
    assert abs(calc._svv_coupling(s["h"], s["Wp"]).real - G**2 * VEV / 2) < 1e-6
    width = calc.offshell_vv_width(s["h"], s["Wp"], MH, MW, GW,
                                   [(G / np.sqrt(2), 0.0, 9)],
                                   identical=False, backend="gauss")
    assert abs(width * 1e3 - 0.80) < 0.03
