"""SM + type-I seesaw with feynlag, end to end.

The Standard Model extended by right-handed neutrinos ``ν_R`` (total gauge
singlets) with a Dirac Yukawa and a large right-handed Majorana mass:

    L ⊃ −Y_ν L̄ H̃ ν_R  −  ½ M_R ν_Rᵀ C ν_R  +  h.c.

After EWSB the Dirac mass is ``m_D = Y_ν v/√2``.  In the left-handed basis
``n = (ν_L, ν_R^c)`` the neutral-fermion Majorana mass matrix is

    M_ν = [[ 0 , m_D ], [ m_Dᵀ, M_R ]]   (Takagi-diagonalised).

For ``M_R ≫ m_D`` this is the **seesaw**: a naturally tiny light neutrino
``m_ν ≈ −m_D M_R⁻¹ m_Dᵀ`` and a heavy state ``≈ M_R``, mixed by
``V ≈ m_D M_R⁻¹``.  The light–heavy mixing ``V`` controls every heavy-neutrino
coupling, extracted below through feynlag's vertex machinery via the
``MajoranaRotation`` (the physical neutrinos are Majorana, mixing ``ν_L`` with
the charge-conjugate ``ν_R^c``).

Headline physics (pinned in tests/test_seesaw.py):

- ``m_ν = −m_D²/M_R`` (1 generation), matching the exact Takagi light mass;
- the heavy-neutrino production coupling ``W ℓ̄ N = (g/√2)·V``, with
  ``V ≈ m_D/M_R`` — vanishing as ``M_R → ∞`` (decoupling);
- the neutral current ``Z ν̄ N ∝ (g_Z/2)·V``.

Run:  python examples/sm_seesaw.py
"""

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, Dmu, ExternalParameter, InternalParameter, Lagrangian,
    MajoranaBilinear, MajoranaRotation, Model, Rotation, SU2, Scalar, U1,
    WeylFermion, dag, diagonalize_takagi, diracC, diracPL, diracPR,
    extract_fermion_vertices, fermion_gauge_current, fermion_mass_matrix,
    majorana_mass_matrix, rotation_2x2, seesaw_light_mass, seesaw_mass_matrix,
)


def _num(x):
    return complex(sp.N(x))


def main():
    # --- symmetries, parameters, fields (1 generation) ------------------
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)
    yv = ExternalParameter("yv", 0.01, positive=True)          # Dirac Yukawa
    MR = ExternalParameter("MR", 1.0e3, positive=True, unit_dim=1)  # seesaw scale

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=1,
                      component_names=["nuR"])

    i, j = sp.symbols("i j", integer=True)
    Gp, H0 = H.components
    nuL, eL = Ll.components
    nuLbar, eLbar = Ll.bar_components
    nR, nRbar = nuR.components[0], nuR.bar_components[0]
    CPL = diracC * diracPL

    # Dirac Yukawa via H̃ = (H0*, −Gp*):  −Y_ν L̄ H̃ ν_R + h.c.
    LYukD = -(yv.s * sp.conjugate(H0) * Bilinear(nuLbar[i], diracPR, nR[j])
              + yv.s * (-sp.conjugate(Gp)) * Bilinear(eLbar[i], diracPR, nR[j]))
    LYukD += -(sp.conjugate(yv.s) * H0 * Bilinear(nRbar[j], diracPL, nuL[i])
               + sp.conjugate(yv.s) * (-Gp) * Bilinear(nRbar[j], diracPL, eL[i]))
    # Right-handed Majorana mass:  −½ M_R ν_Rᵀ C ν_R + h.c.
    op = -sp.Rational(1, 2) * MR.s * MajoranaBilinear(nR[i], CPL, nR[j])
    LMaj = op + sp.conjugate(op)
    current = fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i)

    L = Lagrangian()
    L.add((dag(Dmu(H)) * Dmu(H))[0], sector="kinetic")
    L.add(-(-mu2.s * (dag(H) * H.mat)[0] + lam.s * (dag(H) * H.mat)[0]**2),
          sector="potential")
    L.add(LYukD + current, sector="yukawa")
    L.add(LMaj, sector="other")     # kept separate: the Majorana mass, not a vertex

    model = Model("SM-seesaw", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, nuR, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, lam, mu2, yv, MR], lagrangian=L)

    print("invariance:", model.check_invariance())
    model.check_invariance().raise_on_failure()
    print("tadpole:   ", model.solve_tadpoles([mu2]))

    # gauge rotations (Weinberg + W±)
    W, B = SU2L.bosons(), U1Y.bosons()
    W1, W2, W3 = W.components
    B0 = B.components[0]
    Z, A = sp.symbols("Z A", real=True)
    thetaW = sp.atan(g1.s / gw.s)
    model.rotate(Rotation([W3, B0], [Z, A], rotation_2x2(-thetaW)))
    Wp, Wm = sp.symbols("Wp Wm")
    model.rotate(Rotation([W1, W2], [Wp, Wm],
                          sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2),
                          kind="unitary"))

    # --- the seesaw mass matrix -----------------------------------------
    mD = fermion_mass_matrix(LYukD, nuLbar, nR, model.vacuum, 1, (i, j),
                             gamma=diracPR)
    MRmat = majorana_mass_matrix(LMaj, nR, model.vacuum, 1, (i, j), gamma=CPL)
    Mnu = seesaw_mass_matrix(mD, MRmat)
    print("\nseesaw mass matrix M_ν (basis ν_L, ν_R^c):")
    sp.pprint(Mnu)
    print("seesaw light mass  m_ν = −m_D²/M_R =",
          sp.simplify(seesaw_light_mass(mD, MRmat)[0, 0]))

    # --- Takagi diagonalisation at a benchmark --------------------------
    bench = {yv.s: 0.01, v.s: 246.0, MR.s: 1000.0, gw.s: 0.6535, g1.s: 0.3580}
    Mn = sp.Matrix(2, 2, lambda a, b: sp.nsimplify(Mnu[a, b].subs(bench)))
    U, D = diagonalize_takagi(Mn)
    masses = [abs(_num(D[k, k])) for k in range(2)]
    light = 0 if masses[0] < masses[1] else 1
    heavy = 1 - light
    mixing = abs(_num(U.conjugate()[0, heavy]))
    print(f"\nbenchmark (y_ν=0.01, v=246, M_R=1000 GeV):")
    print(f"   m_light = {masses[light]:.4e} GeV,  m_heavy = {masses[heavy]:.1f} GeV")
    print(f"   light–heavy mixing V = |U*[0,heavy]| = {mixing:.5f}  (≈ m_D/M_R)")

    # --- rotate the weak neutrinos → physical Majorana χ, extract couplings
    chiL, chiR = sp.IndexedBase("chiL"), sp.IndexedBase("chiR")
    chiLbar, chiRbar = sp.IndexedBase("chiLbar"), sp.IndexedBase("chiRbar")
    rot = MajoranaRotation(U, nuL, nR, nuLbar, nRbar,
                           chiL, chiR, chiLbar, chiRbar, n_L=1)
    Lchi = rot.apply(model.physical_lagrangian(sector="yukawa"), (i, j), 1)
    tab = extract_fermion_vertices(Lchi, [Wp, Wm, Z, A])

    mu = sp.Symbol("mu", integer=True)
    gL = DiracGamma(mu) * diracPL
    gsq = float((gw.s / sp.sqrt(2)).subs(bench))
    gz = float(sp.sqrt(gw.s**2 + g1.s**2).subs(bench))

    def cval(c):
        return abs(_num(sp.simplify(c).subs(bench))) if c != 0 else 0.0

    print("\nheavy-neutrino couplings (feynlag-extracted, physical basis):")
    for k, tag in ((light, "light ν"), (heavy, "heavy N")):
        cW = cval(tab.get((eLbar[0], gL, chiL[k]), {}).get(1, {}).get((Wm,), 0))
        print(f"   W⁻ e {tag}:  |coupling|/(g/√2) = {cW / gsq:.5f}")
    for k, tag in ((light, "light ν"), (heavy, "heavy N")):
        cZ = cval(tab.get((chiLbar[light], gL, chiL[k]), {}).get(1, {}).get((Z,), 0))
        print(f"   Z ν {tag}:   |coupling|/g_Z    = {cZ / gz:.5f}")
    print("→ the heavy-N couplings are suppressed by the mixing V ≈ m_D/M_R;")
    print("  they vanish as M_R → ∞ (the seesaw decoupling).")


if __name__ == "__main__":
    main()
