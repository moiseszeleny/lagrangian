"""SM two-body decay widths from extracted Feynman rules (`feynlag.pheno`).

Builds the electroweak SM with one lepton generation, then runs the decay
calculator over it to get symbolic partial widths, branching ratios and
numbers at the PDG parameter point:

    Γ(Z→νν̄)   = m_Z(g²+g'²)/(96π)  = m_Z³/(24πv²)   ≈ 0.166 GeV
    Γ(Z→ℓ⁺ℓ⁻) = m_Z(5g'⁴−2g²g'²+g⁴)/(96π(g²+g'²))   ≈ 0.084 GeV
    Γ(W→ℓν)   = g²m_W/(48π)                          ≈ 0.227 GeV
    Γ(h→ℓ⁺ℓ⁻) = y²m_h β³/(16π)

Run with::

    python examples/sm_decays.py
"""

import sympy as sp

from feynlag import (
    Bilinear, ExternalParameter, Lagrangian, Model, SU2, U1, WeylFermion,
    diracPR, electroweak_scaffold, fermion_gauge_current, to_physical_basis,
)
from feynlag.pheno import DecayCalculator

# PDG-ish parameter point
GW, G1, VEV, MH = 0.6535, 0.3580, 246.0, 125.25
MZ, MW, MTAU = 91.1876, 80.377, 1.77686


def build_model():
    """SM electroweak + one lepton generation, rotated to the physical basis.

    The electroweak scaffold (gauge groups, Higgs, physical-basis rotations)
    comes from :mod:`feynlag.models`; only the τ lepton + its Yukawa are
    written out here — the actual subject of this example.
    """
    ew = electroweak_scaffold(gw=GW, g1=G1, v=VEV, mh=MH)
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    # y = √2 m_τ / v  reproduces the τ mass after EWSB
    ytau = ExternalParameter("ytau", sp.sqrt(2) * MTAU / VEV, positive=True)

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "tauL"])
    tauR = WeylFermion("tauR", reps={U1Y: -1}, chirality="R", nflavors=1,
                       component_names=["tauR"])

    i = sp.Symbol("i", integer=True)
    L = Lagrangian()
    ew.add_higgs(L)                                    # kinetic + potential
    L.add(fermion_gauge_current(Ll, i) + fermion_gauge_current(tauR, i),
          sector="gauge")

    nuLb, tauLb = Ll.bar_components
    nuL, tauL = Ll.components
    tauRb, tauRc = tauR.bar_components[0], tauR.components[0]
    yuk = -ytau.s * (Bilinear(nuLb[i], diracPR, tauRc[i]) * H.components[0]
                     + Bilinear(tauLb[i], diracPR, tauRc[i]) * H.components[1])
    L.add(yuk + sp.conjugate(yuk), sector="yukawa")

    model = Model("SM_lepton", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Ll, tauR],
                  parameters=ew.parameters + [ytau], lagrangian=L)
    model.solve_tadpoles([ew.mu2])

    # physical basis: the standard Weinberg + W± rotations
    phys = to_physical_basis(model, ew)

    # A Dirac fermion is two WeylFermions in feynlag, so state which legs are
    # the same physical particle — otherwise the τ's L and R currents would be
    # counted as two separate decay channels.
    tau, taubar, nu, nubar = sp.symbols("tau taubar nu nubar")
    particle_map = {tauL[i]: tau, tauRc[i]: tau,
                    tauLb[i]: taubar, tauRb[i]: taubar,
                    nuL[i]: nu, nuLb[i]: nubar}

    return dict(model=model, conjugate_map=phys.cmap, particle_map=particle_map,
                bosons=phys.bosons,
                h=phys.h, Z=phys.Z, A=phys.A, Wp=phys.Wp, Wm=phys.Wm,
                tau=tau, taubar=taubar, nu=nu, nubar=nubar,
                gw=ew.gw, g1=ew.g1, v=ew.v, ytau=ytau)


def main():
    s = build_model()
    mZ, mW, mh, mtau = sp.symbols("m_Z m_W m_h m_tau", positive=True)
    masses = {s["Z"]: mZ, s["Wp"]: mW, s["Wm"]: mW, s["A"]: 0, s["h"]: mh,
              s["tau"]: mtau, s["taubar"]: mtau, s["nu"]: 0, s["nubar"]: 0}

    calc = DecayCalculator(
        s["model"], masses,
        boson_fields=s["bosons"],
        fermion_sectors=("gauge", "yukawa"),
        conjugate_map=s["conjugate_map"],
        particle_map=s["particle_map"],
        parameters=s["model"].parameters,
    )

    numbers = {mZ: MZ, mW: MW, mh: MH, mtau: MTAU}

    print("=" * 68)
    print("SM two-body decay widths from feynlag.pheno")
    print("=" * 68)

    for parent, label in ((s["Z"], "Z"), (s["Wp"], "W+"), (s["h"], "h")):
        widths = calc.partial_widths(parent)
        if not widths:
            continue
        # NB: the numeric_* methods, not numeric(partial_widths(...)) — with
        # symbolic masses a channel below threshold cannot be recognised as
        # closed, and at the numeric point its √λ turns imaginary. h→WW/ZZ at
        # m_h = 125 GeV is exactly that case.
        values = calc.numeric_partial_widths(parent, extra=numbers)
        ratios = calc.numeric_branching_ratios(parent, extra=numbers)
        print(f"\n{label}:")
        for children, width in sorted(widths.items(), key=lambda kv: str(kv[0])):
            kids = " ".join(str(c) for c in children)
            value = values[children]
            flag = "" if value else "   [closed at this mass point]"
            print(f"  {label} -> {kids:<16} Γ = {value:9.5f} GeV{flag}")
            print(f"       {sp.simplify(width)}")
        print(f"  {'total (2-body, this field content)':<34} "
              f"Γ = {calc.numeric_total_width(parent, extra=numbers):9.5f} GeV")
        for children, ratio in sorted(ratios.items(), key=lambda kv: str(kv[0])):
            kids = " ".join(str(c) for c in children)
            print(f"       BR({label} -> {kids:<14}) = {ratio:7.4f}")

    # --- sanity anchors against the known SM values ----------------------
    print("\n" + "-" * 68)
    print("cross-check against the standard results")
    print("-" * 68)
    g, gp = s["gw"].s, s["g1"].s
    z_widths = calc.numeric_partial_widths(s["Z"], extra=numbers)

    z_nu = calc.partial_widths(s["Z"])[(s["nubar"], s["nu"])]
    assert sp.simplify(z_nu - mZ * (g**2 + gp**2) / (96 * sp.pi)) == 0
    print(f"  Γ(Z->nu nubar) = m_Z(g²+g'²)/96π   -> "
          f"{z_widths[(s['nubar'], s['nu'])]:.5f} GeV  (PDG 0.1663)")

    z_ll = calc.partial_widths(s["Z"])[(s["taubar"], s["tau"])]
    assert sp.simplify(sp.simplify(z_ll.subs(mtau, 0))
                       - mZ * (5 * gp**4 - 2 * g**2 * gp**2 + g**4)
                       / (96 * sp.pi * (g**2 + gp**2))) == 0
    print(f"  Γ(Z->tau tau)  = m_Z(5g'⁴−2g²g'²+g⁴)/96π(g²+g'²) -> "
          f"{z_widths[(s['taubar'], s['tau'])]:.5f} GeV  (PDG 0.0841)")

    w_lnu = calc.partial_widths(s["Wp"])[(s["nubar"], s["tau"])]
    assert sp.simplify(sp.simplify(w_lnu.subs(mtau, 0))
                       - g**2 * mW / (48 * sp.pi)) == 0
    w_num = calc.numeric_partial_widths(s["Wp"], extra=numbers)
    print(f"  Γ(W->tau nu)   = g²m_W/48π          -> "
          f"{w_num[(s['nubar'], s['tau'])]:.5f} GeV  (PDG 0.2264)")

    h_tau = calc.partial_widths(s["h"])[(s["taubar"], s["tau"])]
    beta = sp.sqrt(1 - 4 * mtau**2 / mh**2)
    assert sp.simplify(sp.expand(
        h_tau - s["ytau"].s**2 * mh * beta**3 / (16 * sp.pi))) == 0
    h_num = calc.numeric_partial_widths(s["h"], extra=numbers)
    print(f"  Γ(h->tau tau)  = y²m_h β³/16π       -> "
          f"{h_num[(s['taubar'], s['tau'])] * 1e3:.5f} MeV  (SM ~0.26 MeV)")

    # h → WW/ZZ are open symbolically but closed at m_h = 125 GeV
    assert h_num[(s["Wm"], s["Wp"])] == 0.0
    assert h_num[(s["Z"], s["Z"])] == 0.0
    print("  h->WW, h->ZZ correctly closed at m_h = 125 GeV "
          "(symbolically present, kinematically shut)")
    print("\nall symbolic cross-checks passed.")


if __name__ == "__main__":
    main()
