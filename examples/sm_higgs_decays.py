"""Higgs → all tree-level fermion pairs via the `DiracParticle` API.

Tier 1 of the decays roadmap (`docs/manual/decays_roadmap.md`): every
kinematically-open tree-level `h → f f̄` channel — ττ, μμ, b̄b, c̄c, s̄s —
declared as one :class:`~feynlag.pheno.particles.DiracParticle` each, so the
two Weyl legs, the mass and the colour N_c travel together and the
per-leg colour over-count (the "81×" trap) is unrepresentable.

Uses **running** quark masses at the Higgs scale (m̄_b(m_h) ≈ 2.79 GeV, not the
4.8 GeV pole mass) so the branching ratios land near the canonical table:
b̄b-dominated, with ττ a distant second.

Run with::

    python examples/sm_higgs_decays.py
"""

import sympy as sp

from feynlag import (
    Bilinear, ExternalParameter, Lagrangian, Model, SU2, U1, WeylFermion,
    diracPR, electroweak_scaffold,
)
from feynlag.pheno import DecayCalculator, DiracParticle

VEV, MH = 246.0, 125.25

# Fermion content: (name, colour N_c, running mass at ~m_h in GeV).  The
# up/charm/top and the light leptons are included only where the channel is
# open and matters; top is kinematically closed (2 m_t ≫ m_h) so it is omitted.
FERMIONS = [
    ("tau", 1, 1.777),
    ("mu",  1, 0.1057),
    ("b",   3, 2.79),
    ("c",   3, 0.620),
    ("s",   3, 0.055),
]


def build_model():
    """Higgs + one down-type-like Weyl pair per fermion, colour-diagonal
    single-component Yukawas.  Returns ``(model, h, particles, yukawas)``."""
    ew = electroweak_scaffold(v=VEV, mh=MH)
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    H0 = H.components[1]
    i = sp.Symbol("i", integer=True)

    L = Lagrangian()
    ew.add_higgs(L)                                    # kinetic + potential

    fields = [H]
    particles, yukawas = [], {}
    # one left doublet + right singlet per fermion, with a single colour
    # component (colour multiplicity handled by DiracParticle.color).
    for name, nc, _mass in FERMIONS:
        QL = WeylFermion(f"Q{name}L", reps={SU2L: 2, U1Y: sp.Rational(1, 6)},
                         chirality="L", nflavors=1,
                         component_names=[f"u{name}L", f"{name}L"])
        fR = WeylFermion(f"{name}R", reps={U1Y: -sp.Rational(1, 3)},
                         chirality="R", nflavors=1,
                         component_names=[f"{name}R"])
        fields += [QL, fR]
        y = ExternalParameter(f"y{name}", sp.sqrt(2) * _mass / VEV, positive=True)
        dLb = QL.bar_components[1]
        fRc = fR.components[0]
        term = -y.s * Bilinear(dLb[i], diracPR, fRc[i]) * H0
        L.add(term + sp.conjugate(term), sector="yukawa")
        particles.append((name, nc, QL.components[1], fR.components[0], y))
        yukawas[name] = y

    fields += [ew.W, ew.B]
    params = ew.parameters + list(yukawas.values())
    model = Model("SM_higgs_decays", gauge_groups=ew.gauge_groups,
                  fields=fields, parameters=params, lagrangian=L)
    model.solve_tadpoles([ew.mu2])

    h = H.vev_expansions[H0][1]                        # physical CP-even Higgs
    return model, h, particles, yukawas


def main():
    model, h, raw, yukawas = build_model()

    # masses as symbols so the widths stay symbolic; numbers substituted below
    mh = sp.Symbol("m_h", positive=True)
    running = {n: m for n, _, m in FERMIONS}
    dirac = []
    mass_syms, mass_vals = {}, {mh: MH}
    for name, nc, left, right, y in raw:
        mf = sp.Symbol(f"m_{name}", positive=True)
        mass_syms[name] = mf
        mass_vals[mf] = running[name]
        dirac.append(DiracParticle(name, left=left, right=right, mass=mf,
                                   color=nc))

    calc = DecayCalculator(model, {h: mh}, boson_fields=[h],
                           fermion_sectors=("yukawa",), particles=dirac,
                           parameters=model.parameters)

    print("=" * 60)
    print("Higgs tree-level fermionic decays (feynlag.pheno, Tier 1)")
    print("=" * 60)

    widths = calc.partial_widths(h)
    numeric = calc.numeric_partial_widths(h, extra=mass_vals)
    total = sum(numeric.values())

    # order by width, descending
    ordered = sorted(numeric, key=lambda k: -numeric[k])
    print(f"\n{'channel':<16}{'Γ [MeV]':>12}{'BR (Tier 1)':>14}")
    print("-" * 42)
    for children in ordered:
        kids = "".join(str(c) for c in children)
        w = numeric[children]
        print(f"h -> {kids:<11}{w * 1e3:>12.4f}{w / total:>14.4f}")
    print("-" * 42)
    print(f"{'total (Tier 1)':<16}{total * 1e3:>12.4f}")
    assert calc.unmatched_channels == [], calc.unmatched_channels

    # --- cross-check: the symbolic bb width is N_c y_b² m_h β³/16π ----------
    b, bbar = sp.Symbol("b"), sp.Symbol("bbar")
    bb = next(w for c, w in widths.items() if set(c) == {b, bbar})
    yb, mb = yukawas["b"].s, mass_syms["b"]
    beta3 = (mh**2 - 4 * mb**2)**sp.Rational(3, 2)
    assert sp.simplify(bb - 3 * yb**2 * beta3 / (16 * sp.pi * mh**2)) == 0
    print("\nΓ(h→b̄b) = 3 y_b² m_h β³/16π  ✓  (colour N_c = 3, applied once)")
    print(f"b̄b share of the tree-level fermionic width: "
          f"{numeric[(bbar, b)] / total:.1%}")
    print("\nMissing from the *total* Higgs width (see decays_roadmap.md):")
    print("  off-shell WW*/ZZ* (Tier 2), loop-induced gg/γγ (Tier 3).")


if __name__ == "__main__":
    main()
