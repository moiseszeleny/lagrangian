"""Export an electroweak + leptons Standard Model UFO from feynlag.

Builds the SM Higgs + SU(2)_L×U(1)_Y + one lepton generation, runs the pipeline
to the physical basis, extracts the FFV lepton currents, the VVV (WWγ/WWZ)
self-couplings and the hVV couplings, and writes a MadGraph-importable UFO.

Parameter values are computed with the *stock* MadGraph ``sm`` EW input scheme
(aEWM1, Gf, MZ), so the emitted couplings are numerically identical to the
standard model shipped with MadGraph — the basis for the cross-section
round-trip in ``madgraph_roundtrip.py``.

Run:  python scripts/export_sm_ufo.py [output_dir]
"""

import math
import sys

import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, ExternalParameter, InternalParameter, Lagrangian,
    Model, ParameterSet, Rotation, SU2, Scalar, U1, WeylFermion,
    ChargeRegistry, Dmu, conjugate_pair, cubic_couplings, dag, diracPL,
    diracPR, extract_fermion_vertices, fermion_gauge_current, rotation_2x2,
    verify_ufo_numeric,
)
from feynlag.export.ufo import UFOParticle, write_ufo


def _ew_values():
    """SM electroweak parameters in the stock (aEWM1, Gf, MZ) scheme."""
    aEWM1, Gf, MZ = 132.50698, 1.16639e-5, 91.1876
    aEW = 1 / aEWM1
    ee = math.sqrt(4 * math.pi * aEW)
    MW = math.sqrt(MZ**2 / 2
                   + math.sqrt(MZ**4 / 4 - aEW * math.pi * MZ**2
                               / (Gf * math.sqrt(2))))
    sw = math.sqrt(1 - MW**2 / MZ**2)
    return dict(gw=ee / sw, g1=ee / math.sqrt(1 - (1 - MW**2 / MZ**2)),
                v=2 * MW * sw / ee, MZ=MZ, MW=MW, MH=125.0,
                WZ=2.4952, WW=2.0850, WH=4.07e-3)


def build_model():
    """SM EW + one lepton generation, rotated to the physical basis."""
    val = _ew_values()
    gw = ExternalParameter("gw", val["gw"], positive=True)
    g1 = ExternalParameter("g1", val["g1"], positive=True)
    v = ExternalParameter("v", val["v"], positive=True, unit_dim=1)
    lam = ExternalParameter("lam", val["MH"]**2 / (2 * val["v"]**2))
    mu2 = InternalParameter("mu2", unit_dim=2)

    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])

    HdH = (dag(H) * H.mat)[0]
    DH = Dmu(H)
    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-(-mu2.s * HdH + lam.s * HdH**2), sector="potential")
    current = (fermion_gauge_current(Ll, sp.Symbol("i", integer=True))
               + fermion_gauge_current(eR, sp.Symbol("i", integer=True)))
    L.add(current, sector="gauge")

    model = Model("SM_EW_lep", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, lam, mu2], lagrangian=L)
    model.solve_tadpoles([mu2])

    W1, W2, W3 = SU2L.bosons().components
    B = U1Y.bosons().components[0]
    Z, A = sp.symbols("Z A", real=True)
    thetaW = sp.atan(g1.s / gw.s)
    model.rotate(Rotation([W3, B], [Z, A], rotation_2x2(-thetaW)))
    Wp, Wm = sp.symbols("Wp Wm")
    U = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], U, kind="unitary"))

    return model, dict(SU2L=SU2L, U1Y=U1Y, H=H, Ll=Ll, eR=eR, Z=Z, A=A,
                       Wp=Wp, Wm=Wm, gw=gw, g1=g1, v=v, val=val)


def export(path):
    model, s = build_model()
    gw, g1, v = s["gw"], s["g1"], s["v"]
    Z, A, Wp, Wm = s["Z"], s["A"], s["Wp"], s["Wm"]
    i = sp.Symbol("i", integer=True)
    mu = sp.Symbol("mu", integer=True)
    gL, gR = DiracGamma(mu) * diracPL, DiracGamma(mu) * diracPR

    # --- fermion currents in the physical basis --------------------------
    L_gauge = sp.expand(model.physical_lagrangian(sector="gauge"))
    ftab = extract_fermion_vertices(L_gauge, [Z, A, Wp, Wm])
    Ll, eR = s["Ll"], s["eR"]
    nuLbar, eLbar = Ll.bar_components
    nuL, eL = Ll.components
    eRbar, eRc = eR.bar_components[0], eR.components[0]

    def cc(key, gamma, boson):
        return sp.simplify(ftab.get(key, {}).get(1, {}).get((boson,), 0))

    # neutral-current couplings (flavour-universal → shared by e and μ)
    A_lL = cc((eLbar[i], gL, eL[i]), gL, A)
    A_lR = cc((eRbar[i], gR, eRc[i]), gR, A)
    Z_lL = cc((eLbar[i], gL, eL[i]), gL, Z)
    Z_lR = cc((eRbar[i], gR, eRc[i]), gR, Z)
    Z_nu = cc((nuLbar[i], gL, nuL[i]), gL, Z)
    # charged current
    W_lL = cc((nuLbar[i], gL, eL[i]), gL, Wp)

    # --- gauge self-couplings (physical basis) ---------------------------
    g, gp = gw.s, g1.s
    cw, sw = g / sp.sqrt(g**2 + gp**2), gp / sp.sqrt(g**2 + gp**2)
    Uc = sp.Matrix([[1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0],
                    [sp.I / sp.sqrt(2), -sp.I / sp.sqrt(2), 0, 0],
                    [0, 0, cw, sw]])
    cubic = cubic_couplings(s["SU2L"], physical=[Wp, Wm, Z, A], U=Uc)
    # The triple-gauge coupling built from the COMPLEX W± rotation
    # (W± = (W1∓iW2)/√2) comes out of cubic_couplings with the opposite overall
    # sign to MadGraph's VVV1 Lorentz convention — validated by the e+e-→W+W-
    # round-trip (only the ν t-channel × γ/Z s-channel interference is
    # sensitive to it: without the flip the diagrams add instead of gauge-
    # cancelling, giving ~98 pb instead of the correct ~19.5 pb). The real-basis
    # QCD gluon self-coupling (ggg = −g_s) is unaffected and already matches.
    gAWW = -sp.simplify(cubic.get((A, Wp, Wm), 0))
    gZWW = -sp.simplify(cubic.get((Z, Wp, Wm), 0))

    # --- hVV from the kinetic sector -------------------------------------
    h = sp.Symbol("H0_r", real=True)
    hWW = sp.I * g**2 * v.s / 2       # = i g m_W  (pinned in test_gauge_sector)
    hZZ = sp.I * (g**2 + gp**2) * v.s / 2

    # --- particles --------------------------------------------------------
    em, ep = sp.symbols("em ep")
    mm, mp = sp.symbols("mm mp")
    ve, veb = sp.symbols("ve veb")
    vm, vmb = sp.symbols("vm vmb")
    hs = sp.Symbol("h")
    gwv, g1v = s["val"]["gw"], s["val"]["g1"]
    particles = [
        UFOParticle(em, 11, "e-", antiname="e+", spin=2, charge=-1,
                    antisymbol=ep),
        UFOParticle(mm, 13, "mu-", antiname="mu+", spin=2, charge=-1,
                    antisymbol=mp),
        UFOParticle(ve, 12, "ve", antiname="ve~", spin=2, charge=0,
                    antisymbol=veb),
        UFOParticle(vm, 14, "vm", antiname="vm~", spin=2, charge=0,
                    antisymbol=vmb),
        UFOParticle(A, 22, "a", spin=3, charge=0),
        UFOParticle(Z, 23, "Z", spin=3, charge=0, mass="MZ", width="WZ"),
        UFOParticle(Wp, 24, "W+", antiname="W-", spin=3, charge=1, mass="MW",
                    width="WW", antisymbol=Wm),
        UFOParticle(hs, 25, "h", spin=1, charge=0, mass="MH", width="WH"),
    ]

    # --- parameters -------------------------------------------------------
    val = s["val"]
    MZ = InternalParameter("MZ", positive=True, unit_dim=1)
    MW = InternalParameter("MW", positive=True, unit_dim=1)
    MZ.define(sp.sqrt(g**2 + gp**2) * v.s / 2)
    MW.define(g * v.s / 2)
    MH = ExternalParameter("MH", val["MH"], positive=True, unit_dim=1)
    WZ = ExternalParameter("WZ", val["WZ"], positive=True, unit_dim=1)
    WW = ExternalParameter("WW", val["WW"], positive=True, unit_dim=1)
    WH = ExternalParameter("WH", val["WH"], positive=True, unit_dim=1)
    lam_ext = next(p for p in model.parameters.externals if p.name == "lam")
    params = ParameterSet(gw, g1, v, lam_ext, MH, WZ, WW, WH, MZ, MW)

    # --- vertices ---------------------------------------------------------
    fermion_vertices = []
    for lm, lp in ((em, ep), (mm, mp)):
        fermion_vertices += [
            dict(bar=lp, field=lm, bosons=(A,), left=A_lL, right=A_lR),
            dict(bar=lp, field=lm, bosons=(Z,), left=Z_lL, right=Z_lR),
        ]
    for nu, nub, lm, lp in ((ve, veb, em, ep), (vm, vmb, mm, mp)):
        fermion_vertices += [
            dict(bar=nub, field=nu, bosons=(Z,), left=Z_nu, right=0),
            # W⁺ ν̄ ℓ⁻ and W⁻ ℓ⁺ ν
            dict(bar=nub, field=lm, bosons=(Wp,), left=W_lL, right=0),
            dict(bar=lp, field=nu, bosons=(Wm,), left=W_lL, right=0),
        ]

    vvv = {(A, Wp, Wm): gAWW, (Z, Wp, Wm): gZWW}

    from feynlag.vertices.vertex import Vertex
    bosonic = [Vertex((hs, Wp, Wm), hWW, "VVS"),
               Vertex((hs, Z, Z), hZZ, "VVS")]

    write_ufo(path, "FEYNLAG_SM", params, particles,
              bosonic_vertices=bosonic, vvv=vvv,
              fermion_vertices=fermion_vertices)
    return path, model, s


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/FEYNLAG_SM_UFO"
    path, model, s = export(out)
    report = verify_ufo_numeric(path)
    status = "PASS" if report.ok else "FAIL"
    print(f"exported {path}")
    print(f"UFO round-trip: {status} "
          f"({len(report.parameters)} params, {len(report.couplings)} couplings)")
    if not report.ok:
        for name, err in report.failures.items():
            print("  FAIL", name, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
