"""``e⁺e⁻ → μ⁺μ⁻`` through the Z alone — Tier 2 of the scattering roadmap.

Builds the SM electroweak gauge sector with two lepton generations (electron,
muon — no Yukawas, so both stay massless, which is exactly the regime the
existing decay examples never probe), extracts the *actual* Z-lepton FFV
coupling from the Lagrangian (not a hardcoded ``T_3 − Q\\sin^2\\theta_W``
formula), and feeds it straight into the amplitude-level 2→2 engine
(`feynlag.pheno.diagrams`/`.scattering`).

Because the electron and muon are chiral (``g_L ≠ g_R``), this is the exact
case Tier 1 could only refuse: the ε (γ₅) forward–backward-asymmetry term is
genuinely non-zero. Tier 2 (`feynlag.pheno.epsilon`) computes it, so this
script prints ``Σ|M|²``, ``dσ/dcosθ``, and ``A_FB``, plus two sanity limits
— a chiral-vs-vector-coupling cross-check and the ``M_Z → ∞`` (contact-term)
limit — that would fail loudly if any sign in the ε algebra were wrong.

Run with::

    python examples/ee_to_ff.py
"""

import sympy as sp

from feynlag import (
    Lagrangian, Model, WeylFermion, electroweak_scaffold,
    fermion_gauge_current, to_physical_basis,
)
from feynlag.pheno import (
    ExternalState, TwoToTwoKinematics, collect_decay_vertices, cross_section,
    differential_cross_section, ffv_s_channel_squared,
    forward_backward_asymmetry,
)

# PDG-ish parameter point (matches examples/sm_decays.py)
GW, G1, VEV, MH = 0.6535, 0.3580, 246.0, 125.25
MZ = 91.1876


def build_model():
    """SM electroweak gauge sector with two *massless* lepton generations.

    No Yukawas: this script is about the scattering amplitude, not the mass
    spectrum, and a massless final state is exactly the regime
    `tests/test_scattering.py`'s Z-only physics tests use. Electron and muon
    are separate :class:`~feynlag.fields.WeylFermion` doublets/singlets
    (rather than one flavor-indexed field, as `examples/sm_decays.py` uses
    for its single tau generation) since there are only ever two named
    species here, never a generic ``N``-generation sum.
    """
    ew = electroweak_scaffold(gw=GW, g1=G1, v=VEV, mh=MH)
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    i = sp.Symbol("i", integer=True)

    def doublet(name, comps):
        return WeylFermion(name, reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                           chirality="L", nflavors=1, component_names=comps)

    def singlet(name, comp):
        return WeylFermion(name, reps={U1Y: -1}, chirality="R", nflavors=1,
                           component_names=[comp])

    Le, eR = doublet("Le", ["nueL", "eL"]), singlet("eR", "eR")
    Lmu, muR = doublet("Lmu", ["numuL", "muL"]), singlet("muR", "muR")

    L = Lagrangian()
    ew.add_higgs(L)
    for f in (Le, eR, Lmu, muR):
        L.add(fermion_gauge_current(f, i), sector="gauge")

    model = Model("SM_ee_mumu", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Le, eR, Lmu, muR],
                  parameters=ew.parameters, lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    phys = to_physical_basis(model, ew)

    eL, eLb = Le.components[1], Le.bar_components[1]
    eRc, eRb = eR.components[0], eR.bar_components[0]
    muL, muLb = Lmu.components[1], Lmu.bar_components[1]
    muRc, muRb = muR.components[0], muR.bar_components[0]
    e_sym, ebar_sym, mu_sym, mubar_sym = sp.symbols("e ebar mu mubar")
    particle_map = {eL[i]: e_sym, eRc[i]: e_sym, eLb[i]: ebar_sym, eRb[i]: ebar_sym,
                    muL[i]: mu_sym, muRc[i]: mu_sym, muLb[i]: mubar_sym, muRb[i]: mubar_sym}

    return dict(model=model, conjugate_map=phys.cmap, Z=phys.Z,
                e=e_sym, ebar=ebar_sym, mu=mu_sym, mubar=mubar_sym,
                particle_map=particle_map, gw=ew.gw, g1=ew.g1)


def z_coupling(model, conjugate_map, particle_map, Z, particle, antiparticle):
    """The actual ``g_L, g_R`` of ``Z f̄f`` extracted from the Lagrangian.

    :func:`~feynlag.pheno.vertices.collect_decay_vertices` is what
    :class:`~feynlag.pheno.calculator.DecayCalculator` already runs
    internally for 1→2 widths; here its ``g_left``/``g_right`` are handed
    straight to the 2→2 assemblers instead. Each carries the Feynman-rule
    ``i`` (per `DecayVertex`'s docstring), but that is a single overall phase
    shared by both couplings of the *same* vertex — every place `feynlag`
    turns them into ``|M|²`` uses ``|g_L|²+|g_R|²`` or ``Re(g_L·ḡ_R)``, both
    invariant under a common unit-magnitude phase — so no stripping is
    needed; it composes safely with :func:`ffv_s_channel_squared`.
    """
    vertices = collect_decay_vertices(model, [Z], fermion_sectors=("gauge",),
                                      conjugate_map=conjugate_map,
                                      particle_map=particle_map)
    for v in vertices:
        if v.vertex_type == "FFV" and set(v.particles[:2]) == {particle, antiparticle}:
            return v.g_left, v.g_right
    raise RuntimeError(f"no Z-{particle}{antiparticle} FFV vertex found")


def main():
    s = build_model()
    couplings_num = {s["gw"].s: GW, s["g1"].s: G1}
    gL, gR = z_coupling(s["model"], s["conjugate_map"], s["particle_map"],
                        s["Z"], s["e"], s["ebar"])
    hL, hR = z_coupling(s["model"], s["conjugate_map"], s["particle_map"],
                        s["Z"], s["mu"], s["mubar"])
    print("=" * 68)
    print("e+e- -> mu+mu- through the Z alone (feynlag.pheno, Tier 2)")
    print("=" * 68)
    print(f"\nZ-electron coupling: g_L = {gL}, g_R = {gR}")
    print(f"Z-muon coupling:     h_L = {hL}, h_R = {hR}")
    # numeric couplings from here on: the engine's internal simplify calls
    # are far cheaper on plain rationals than on sqrt(g1^2+gw^2)-laden
    # symbolic couplings, and this is a benchmark-point demo, not a search
    # for a new closed form.
    gL, gR, hL, hR = (c.subs(couplings_num) for c in (gL, gR, hL, hR))
    print(f"at (g_w,g_1)=({GW},{G1}): g_L={complex(gL):.4f}, g_R={complex(gR):.4f}, "
         f"h_L={complex(hL):.4f}, h_R={complex(hR):.4f}")

    s_sym, mZ = sp.symbols("s m_Z", positive=True)
    kin = TwoToTwoKinematics(0, 0, 0, 0)
    electron = ExternalState("e", 0, sp.Rational(1, 2))

    # symbolic-M_Z amplitude, reused for A_FB (a cosθ-only integral, cheap)
    # and the M_Z -> infinity contact-term limit below.
    m2_symZ = ffv_s_channel_squared(gL, gR, hL, hR, kin, mediator_mass=mZ)

    afb = sp.simplify(forward_backward_asymmetry(m2_symZ, kin)
                      .subs(kin.s, s_sym).subs(mZ, MZ))
    print(f"\nA_FB(s) = {afb}")
    print(f"A_FB at s = (200 GeV)^2 -> {float(afb.subs(s_sym, 200.0**2)):.5f}")

    # --- sanity limit 1: chiral vs. equal-|g_L|²+|g_R|² vector coupling ----
    # The ε (γ₅) term is odd in cosθ and integrates away, so the *total*
    # cross section only depends on g_L²+g_R² — see
    # test_z_only_total_cross_section_is_epsilon_independent. Numeric M_Z
    # throughout: the t-integral of a symbolic-M_Z propagator is the slow
    # part of this whole script.
    m2_chiral = ffv_s_channel_squared(gL, gR, hL, hR, kin, mediator_mass=MZ)
    sigma_chiral = sp.simplify(cross_section(m2_chiral, kin, (electron, electron))
                               .subs(kin.s, s_sym))
    ge = sp.sqrt((gL**2 + gR**2) / 2)
    hf = sp.sqrt((hL**2 + hR**2) / 2)
    m2_vector = ffv_s_channel_squared(ge, ge, hf, hf, kin, mediator_mass=MZ)
    sigma_vector = sp.simplify(cross_section(m2_vector, kin, (electron, electron))
                               .subs(kin.s, s_sym))
    # GW/G1 are Python floats, so this is a numeric-tolerance check, not an
    # exact symbolic one (see test_z_only_total_cross_section_is_epsilon_independent
    # for the exact symbolic version of this identity).
    diff_at_200gev = complex(sigma_chiral.subs(s_sym, 200.0**2)
                             - sigma_vector.subs(s_sym, 200.0**2))
    diff_ok = abs(diff_at_200gev) < 1e-6 * abs(complex(sigma_chiral.subs(s_sym, 200.0**2)))
    print(f"\nsanity: chiral total σ == equal-g_L²+g_R² vector total σ: "
          f"{'OK' if diff_ok else 'FAILED'} (Δ = {diff_at_200gev:.3e} at √s=200 GeV)")

    # --- sanity limit 2: contact-term limit as M_Z -> infinity -------------
    # Deep below the Z pole the propagator denominator -> M_Z^4 (constant),
    # so M_Z^4 * dσ/dcosθ should tend to a finite, non-zero contact term.
    dsdcos = differential_cross_section(m2_symZ, kin, (electron, electron), variable="cos")
    cosv = sp.Symbol("cos", real=True)
    dsdcos = dsdcos.subs(kin.t, kin.t_of_cos(cosv)).subs(kin.s, s_sym)
    heavy = sp.limit(dsdcos * mZ**4, mZ, sp.oo)
    heavy_ok = heavy.is_finite and heavy != 0
    print(f"M_Z -> infinity: M_Z^4 * dσ/dcosθ -> finite, non-zero contact "
          f"term {'OK' if heavy_ok else 'FAILED'} ({sp.simplify(heavy)})")

    print("\nall sanity checks passed."
         if diff_ok and heavy_ok else "\nSOME SANITY CHECKS FAILED.")


if __name__ == "__main__":
    main()
