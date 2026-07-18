"""Quark flavour mixing (CKM) in feynlag — the insertion route, not a
symbolic 3×3 SVD.

Works in the quark mass basis (diagonal Yukawas) and inserts the CKM matrix
directly into the charged (W) current from the exact PDG standard
parametrization, so it is unitary by construction. The neutral (Z, γ) currents
stay flavour-diagonal — the GIM mechanism — so no flavour-changing neutral
currents appear. Ends with a UFO export whose nine complex CKM elements
round-trip numerically.

Run:  python examples/sm_ckm.py
"""

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, ExternalParameter, ParameterSet, WeylFermion,
    diracPL, extract_fermion_vertices, fermion_feynman_rule, standard_ckm,
    verify_ufo_numeric,
)
from feynlag.export.ufo import UFOParticle, write_ufo


def main():
    # --- the CKM matrix, unitary by construction ---------------------------
    params, V = standard_ckm()
    sub = {p.s: p.expr for p in params if getattr(p, "expr", None) is not None}
    angles = {p.s: p.value for p in params if getattr(p, "expr", None) is None}
    print("CKM matrix (standard parametrization):")
    sp.pprint(V)
    VdV = (V.subs(sub).conjugate().T * V.subs(sub)).applyfunc(sp.simplify)
    print("\nV†V =")
    sp.pprint(VdV)
    print("\n|Vud|, |Vus|, |Vcb| =",
          [round(abs(complex(V.subs(sub)[a, b].subs(angles))), 4)
           for a, b in [(0, 0), (0, 1), (1, 2)]])

    # --- mass-basis quarks and the charged current -------------------------
    gw = ExternalParameter("gw", 0.6535, positive=True)
    Wp = sp.Symbol("Wp")
    uL = WeylFermion("uL", reps={}, chirality="L", nflavors=3,
                     component_names=["uL"])
    dL = WeylFermion("dL", reps={}, chirality="L", nflavors=3,
                     component_names=["dL"])
    ubar, dfield = uL.bar_components[0], dL.components[0]
    mu = sp.Symbol("mu", integer=True)
    gamma_L = DiracGamma(mu) * diracPL

    # W⁺ ū_i γ^μ P_L V_ij d_j  (CKM inserted, explicit 3×3 sum)
    L_W = gw.s / sp.sqrt(2) * Wp * sum(
        V[a, b] * Bilinear(ubar[a], gamma_L, dfield[b])
        for a in range(3) for b in range(3))
    table = extract_fermion_vertices(sp.expand(L_W), [Wp])

    print("\ncharged-current vertices  W⁺ ū_i d_j  ->  i g/√2 V_ij γ^μ P_L:")
    for a, b in [(0, 0), (0, 1), (0, 2)]:
        coeff = table[(ubar[a], gamma_L, dfield[b])][1][(Wp,)]
        rule = fermion_feynman_rule(coeff, gamma_L, (Wp,))
        print(f"  W u{a+1} d{b+1}  ->", sp.simplify(rule / gamma_L), "γ^μP_L")

    # neutral current is diagonal in the mass basis (GIM): ū_i d_j with i≠j
    # never appears, because we never inserted V there.
    print("\n(neutral Z/γ currents stay flavour-diagonal by construction — GIM)")

    # --- UFO export: nine complex CKM internals round-trip -----------------
    Wm, u, ubar_s = sp.symbols("Wm u ubar")
    d, dbar_s = sp.symbols("d dbar")
    Vud = next(p for p in params if p.name == "Vud")
    particles = [
        UFOParticle(Wp, 24, "W+", antiname="W-", spin=3, charge=1,
                    antisymbol=Wm),
        UFOParticle(u, 2, "u", antiname="u~", spin=2,
                    charge=sp.Rational(2, 3), antisymbol=ubar_s),
        UFOParticle(d, 1, "d", antiname="d~", spin=2,
                    charge=sp.Rational(-1, 3), antisymbol=dbar_s),
    ]
    out = "/tmp/SM_CKM_UFO"
    write_ufo(out, "SM_CKM", ParameterSet(gw, *params), particles,
              fermion_vertices=[dict(bar=ubar_s, field=d, bosons=(Wp,),
                                     left=gw.s / sp.sqrt(2) * Vud.s)])
    report = verify_ufo_numeric(out)
    print(f"\nUFO round-trip: {'PASS' if report.ok else 'FAIL'} "
          f"({len(report.parameters)} params, Vud = "
          f"{report.parameters['Vud']:.4f})")


if __name__ == "__main__":
    main()
