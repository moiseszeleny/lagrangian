"""Fermi theory of muon decay: a four-fermion (dim-6) effective operator.

The classic example of the four-fermion (FFFF) track (Phase C2).  Below the
electroweak scale the exchanged W boson is integrated out and the charged-current
weak interaction becomes a contact operator

    L ⊃ −(4 G_F/√2) (ν̄_μ γ^μ P_L μ)(ē γ_μ P_L ν_e) + h.c.,

    with   G_F/√2 = g²/(8 m_W²)   (the matching to the full SM).

Fermi theory lives *below* electroweak symmetry breaking, where SU(2)_L×U(1)_Y
is already broken to U(1)_em — so the relevant gauge symmetry is electromagnetism,
under which the muon-decay operator is invariant (charge of ν̄_μγμ is −1, of
ēγν_e is +1, summing to zero).  We therefore declare the four leptons as U(1)_em
charge eigenstates.

This is a genuine dimension-6 operator (four spin-½ fields → mass dimension 6),
so the mass-dimension check must be told to admit it via
``check_invariance(max_dim=6)``.  Its Wilson coefficient G_F carries the
compensating mass dimension −2, so the Lagrangian *term* is dimension 4 as it
must be; the "dim-6" refers to the operator content.

The four external fermions (ν_μ, μ, e, ν_e) are all distinct particles, so there
is no Fierz/exchange ambiguity — feynlag supports exactly this "four distinct
components" case and raises on a repeated leg.

Run:  python examples/fermi_theory.py
"""

import tempfile
from pathlib import Path

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, DiracGammaLower, ExternalParameter, Lagrangian,
    Model, ParameterSet, U1, WeylFermion, diracPL,
    extract_fermion_vertices, four_fermion_feynman_rule, verify_ufo_numeric,
)
from feynlag.charges import (
    ChargeRegistry, check_charge_conservation, check_hermiticity_pairing,
)
from feynlag.export.ufo import UFOParticle, write_ufo


def main():
    # --- symmetry and fields (U(1)_em, below the EW scale) ---------------
    e_em = ExternalParameter("e_em", 0.303, positive=True)   # electric coupling
    U1em = U1("U1em", coupling=e_em)

    # G_F carries mass dimension −2 (GeV⁻²) — the EFT suppression scale
    GF = ExternalParameter("G_F", 1.16637e-5, positive=True, unit_dim=-2)

    # the four leptons as electric-charge eigenstates (all left-handed here)
    numu = WeylFermion("numu", reps={U1em: 0}, chirality="L",
                       component_names=["numuL"])
    mu = WeylFermion("mu", reps={U1em: -1}, chirality="L",
                     component_names=["muL"])
    ele = WeylFermion("ele", reps={U1em: -1}, chirality="L",
                      component_names=["eL"])
    nue = WeylFermion("nue", reps={U1em: 0}, chirality="L",
                      component_names=["nueL"])

    # --- the four-fermion operator ---------------------------------------
    i = sp.Symbol("i", integer=True)
    m = sp.Symbol("mu", integer=True)
    gL_up = DiracGamma(m) * diracPL        # γ^μ P_L
    gL_dn = DiracGammaLower(m) * diracPL    # γ_μ P_L  (shared μ ⇒ contraction)

    numuLbar = numu.bar_components[0]
    muL = mu.components[0]
    eLbar = ele.bar_components[0]
    nueL = nue.components[0]

    B1 = Bilinear(numuLbar[i], gL_up, muL[i])     # ν̄_μ γ^μ P_L μ   (charge −1)
    B2 = Bilinear(eLbar[i], gL_dn, nueL[i])       # ē γ_μ P_L ν_e   (charge +1)
    op = -(4 * GF.s / sp.sqrt(2)) * B1 * B2
    L_fermi = op + sp.conjugate(op)               # + h.c.

    L = Lagrangian()
    L.add(L_fermi, sector="other")

    model = Model("Fermi", gauge_groups=[U1em],
                  fields=[numu, mu, ele, nue],
                  parameters=[e_em, GF], lagrangian=L)

    # --- invariance ------------------------------------------------------
    # Every Lagrangian term is dimension 4; here the four-fermion OPERATOR is
    # dim-6 and G_F's dim −2 compensates, so the *term* is dim-4 and passes the
    # default renormalisable check.
    report = model.check_invariance()          # default max_dim=4
    print("check_invariance():", report, " (term is dim-4: G_F carries dim −2)")
    report.raise_on_failure()

    # The max_dim flag matters under the *operator-dimension* convention, where
    # the Wilson coefficient is kept dimensionless and the four-fermion operator
    # reads as dimension 6 — then admitting it needs an explicit opt-in.
    from feynlag import check_mass_dimension
    ok4, worst = check_mass_dimension(B1 * B2, [numu, mu, ele, nue], max_dim=4)
    ok6, _ = check_mass_dimension(B1 * B2, [numu, mu, ele, nue], max_dim=6)
    print(f"operator dimension (dimensionless coefficient) = {worst}: "
          f"max_dim=4 {'OK' if ok4 else 'REJECTED'}, "
          f"max_dim=6 {'OK' if ok6 else 'REJECTED'}")

    # --- extract the contact vertex --------------------------------------
    table = extract_fermion_vertices(L_fermi, [])
    print(f"\n{len(table)} four-fermion vertex key(s):")
    for key, by_n in table.items():
        (bar1, g1chain, f1), (bar2, g2chain, f2) = key
        coeff = by_n[0][()]
        rule = four_fermion_feynman_rule(coeff, (g1chain, g2chain))
        print(f"  ({bar1} {f1})({bar2} {f2})")
        print(f"    coefficient   = {sp.simplify(coeff)}")
        print(f"    Feynman rule  = {sp.simplify(rule)}  (× the two Dirac chains)")
        print(f"    Dirac chains  = {g1chain} ,  {g2chain}")

    # --- charge conservation + hermiticity pairing -----------------------
    registry = ChargeRegistry({numu.components[0]: 0, mu.components[0]: -1,
                               ele.components[0]: -1, nue.components[0]: 0})
    cc = check_charge_conservation(registry, fermion_table=table)
    print("\ncharge conservation:", "OK" if cc.ok else cc.failures)
    hp = check_hermiticity_pairing(fermion_table=table)
    print("hermiticity pairing:", "OK" if hp.ok else hp.failures)

    # --- the SM matching relation ----------------------------------------
    g, v = sp.Symbol("g", positive=True), sp.Symbol("v", positive=True)
    mW = g * v / 2
    print("\nSM matching:  G_F/√2 = g²/(8 m_W²) =",
          sp.simplify(g**2 / (8 * mW**2)), " (with m_W = g v/2 ⟹ G_F/√2 = 1/2v²)")

    # --- UFO export ------------------------------------------------------
    MMU = ExternalParameter("MMU", 0.10566, positive=True, unit_dim=1)
    p_numu, p_numub, p_mm, p_mp, p_em, p_ep, p_nue, p_nueb = sp.symbols(
        "p_numu p_numub p_mm p_mp p_em p_ep p_nue p_nueb")
    particles = [
        UFOParticle(p_numu, 14, "vm", antiname="vm~", antisymbol=p_numub,
                    spin=2),
        UFOParticle(p_mm, 13, "mu-", antiname="mu+", antisymbol=p_mp, spin=2,
                    mass="MMU", charge=-1),
        UFOParticle(p_em, 11, "e-", antiname="e+", antisymbol=p_ep, spin=2,
                    charge=-1),
        UFOParticle(p_nue, 12, "ve", antiname="ve~", antisymbol=p_nueb, spin=2),
    ]
    params = ParameterSet(GF, MMU)
    c = -4 * GF.s / sp.sqrt(2)
    # Export BOTH the operator vertex and its h.c. conjugate: a Hermitian
    # Lagrangian's op + h.c. is two four-fermion vertices, and MadGraph needs
    # the pair to route fermion-number flow through the contact interaction (a
    # model with only one fails diagram generation).  scripts/madgraph_fermi.py
    # uses exactly this UFO to reproduce Γ(μ→eνν) = G_F²m_μ⁵/192π³.
    ffv = [
        {"bar1": p_numub, "field1": p_mm, "bar2": p_ep, "field2": p_nue,
         "couplings": {("VL", "VL"): c}},                      # (ν̄_μγμ)(ēγν_e)
        {"bar1": p_mp, "field1": p_numu, "bar2": p_nueb, "field2": p_em,
         "couplings": {("VL", "VL"): c}},                      # + h.c.
    ]
    out = Path(tempfile.mkdtemp()) / "Fermi_UFO"
    write_ufo(out, "Fermi", params, particles, four_fermion_vertices=ffv)
    print(f"\nUFO written to {out}")
    rt = verify_ufo_numeric(out)
    print("UFO round-trip:", "OK" if rt.ok else rt.failures,
          "| coupling GC =", [complex(v) for v in rt.couplings.values()])


if __name__ == "__main__":
    main()
