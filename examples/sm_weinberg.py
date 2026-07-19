"""SM + the dim-5 Weinberg operator: Majorana neutrino masses, end to end.

The Weinberg operator is the unique dimension-5 gauge invariant you can add to
the Standard Model,

    L ⊃ (c_ij / Λ) (L_iᵀ C ε L_j)(Hᵀ ε H) + h.c.,

and it is *the* effective operator behind neutrino mass: after EWSB it becomes a
**Majorana** mass ``m_ν = −c v²/Λ`` for the left-handed neutrinos (a symmetric
matrix, diagonalized by Takagi factorization), naturally tiny because it is
suppressed by the high scale Λ (the seesaw).

This exercises feynlag's Majorana machinery (Phase D.2):

- the charge-conjugation matrix ``diracC`` and the ``MajoranaBilinear`` atom
  ``ψ₁ᵀ C Γ ψ₂`` (same-chirality, unlike the Dirac ``Bilinear``);
- ``suggest_yukawa(..., max_dim=5)``, which *enumerates* the Weinberg operator
  once you allow dimension-5 terms;
- ``majorana_mass_matrix`` + ``diagonalize_takagi`` for the physical masses;
- ``extract_majorana_vertices`` for the ``ν̄νh`` / ``ν̄νhh`` couplings.

UFO export of Majorana vertices is not yet supported (MadGraph Majorana-fermion
export is a separate follow-up), so no UFO directory is written here.

Run:  python examples/sm_weinberg.py
"""

import sympy as sp

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, Scalar, SU2, U1,
    WeylFermion, Dmu, dag, diracC, diracPL, MajoranaBilinear,
    diagonalize_takagi, extract_majorana_vertices, majorana_feynman_rule,
    majorana_mass_matrix, suggest_yukawa,
)


def main():
    # --- symmetries, parameters, fields ---------------------------------
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    Lam = ExternalParameter("Lam", 1.0e14, positive=True, unit_dim=1)  # seesaw scale
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    # two lepton generations (enough to show the mixing / Takagi)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=2, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=2,
                     component_names=["eR"])

    # --- let `suggest` enumerate the dim-5 operator basis ----------------
    print("suggest_yukawa(max_dim=4):")
    for t in suggest_yukawa([Ll, eR], [H], [SU2L, U1Y], max_dim=4):
        print("   ", t)
    print("suggest_yukawa(max_dim=5) adds the Weinberg operator:")
    for t in suggest_yukawa([Ll, eR], [H], [SU2L, U1Y], max_dim=5):
        print("   ", t)

    # --- write the Weinberg operator explicitly (symmetric c_ij) --------
    i, j = sp.symbols("i j", integer=True)
    c = sp.IndexedBase("c")            # symmetric Weinberg Wilson coefficient
    Gp, H0 = H.components
    nuL, eL = Ll.components
    CPL = diracC * diracPL
    epsH = [H0, -Gp]                    # (εH)_a = ε_ab H^b
    comp = [nuL, eL]
    O = sp.S.Zero
    for a in range(2):
        for b in range(2):
            O += (c[i, j] / Lam.s) * epsH[a] * epsH[b] * \
                MajoranaBilinear(comp[a][i], CPL, comp[b][j])
    O = sp.expand(O)

    HdH = (dag(H) * H.mat)[0]
    V = -mu2.s * HdH + lam.s * HdH**2
    L = Lagrangian()
    L.add((dag(Dmu(H)) * Dmu(H))[0], sector="kinetic")
    L.add(-V, sector="potential")
    L.add(O + sp.conjugate(O), sector="other")

    model = Model("SM-Weinberg", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, Lam, lam, mu2], lagrangian=L)

    # --- validate (dim-5 needs max_dim=5) --------------------------------
    report = model.check_invariance(max_dim=5)
    print("\ninvariance:", report)
    report.raise_on_failure()
    print("tadpole:", model.solve_tadpoles([mu2]))

    # --- Majorana neutrino mass + Takagi diagonalization -----------------
    Mnu = majorana_mass_matrix(O, nuL, model.vacuum, 2, (i, j), gamma=CPL)
    print("\nMajorana mass matrix  m_ν = −c v²/Λ:")
    sp.pprint(Mnu)

    bench = {c[0, 0]: 0.8, c[0, 1]: 0.3, c[1, 0]: 0.3, c[1, 1]: 1.0,
             v.s: 246.0, Lam.s: 1.0e14}
    Mnum = sp.Matrix(2, 2, lambda a, b: sp.nsimplify(Mnu[a, b].subs(bench)))
    U, D = diagonalize_takagi(Mnum)
    masses = [complex(D[k, k]).real for k in range(2)]
    print(f"\nphysical Majorana masses at Λ=10¹⁴ GeV, v=246 GeV:")
    print(f"   m_ν1 = {masses[0]:.3e} GeV,  m_ν2 = {masses[1]:.3e} GeV "
          f"(sub-eV, the seesaw)")
    ok = sp.simplify(U * D * U.T - Mnum) == sp.zeros(2, 2)
    print("   Takagi check  M = U D Uᵀ:", ok)

    # --- ν̄νh coupling ----------------------------------------------------
    Lphys = sp.expand(model.physical_lagrangian(sector="other"))
    h = sp.Symbol("H0_r", real=True)
    G0 = sp.Symbol("H0_i", real=True)
    table = extract_majorana_vertices(Lphys, [h, G0])
    key = (nuL[i], CPL, nuL[j])
    c_h = table.get(key, {}).get(1, {}).get((h,), 0)
    rule = sp.simplify(majorana_feynman_rule(c_h, CPL, (h,)))
    print("\nν̄_i ν_j h coupling =", rule, " (∝ c v/Λ — the seesaw-suppressed "
          "Higgs–neutrino coupling)")


if __name__ == "__main__":
    main()
