"""CKM / quark flavour mixing.

Pins the physics of the pragmatic (insertion, not symbolic-SVD) route:

- the two-generation Cabibbo rotation, driven through the ordinary real
  mass-basis rotation machinery, puts mixing in the charged (W) current while
  the neutral (Z) current stays flavour-diagonal — the GIM mechanism, shown as
  an exact symbolic cancellation;
- the three-generation standard parametrization is exactly unitary
  (``V†V = 1``) and reproduces the PDG magnitudes and CP phase;
- an explicit CKM-inserted W current extracts to ``i g/√2 V_ij γ^μ P_L``.
"""

import cmath

import sympy as sp

from feynlag import (
    DiracGamma, ExternalParameter, ParameterSet, Rotation, WeylFermion,
    diracPL, rotation_2x2, standard_ckm, verify_ufo_numeric,
    Bilinear, extract_fermion_vertices, fermion_feynman_rule,
)
from feynlag.export.ufo import UFOParticle, write_ufo

i = sp.Symbol("i", integer=True)
mu = sp.Symbol("mu", integer=True)
gamma_L = DiracGamma(mu) * diracPL


def _weyl(name):
    return WeylFermion(name, reps={}, chirality="L", nflavors=1,
                       component_names=[name])


class TestTwoGenerationGIM:
    """Cabibbo rotation through the real mass-basis machinery."""

    th = sp.Symbol("theta_c", real=True)

    def _rotated(self, current_builder):
        # weak-basis single-flavour quarks
        u1, u2 = _weyl("u1w"), _weyl("u2w")
        d1, d2 = _weyl("d1w"), _weyl("d2w")
        # mass-basis handles
        D1, D2 = sp.IndexedBase("d1"), sp.IndexedBase("d2")
        D1b, D2b = sp.IndexedBase("d1bar"), sp.IndexedBase("d2bar")
        d1w, d2w = d1.components[0], d2.components[0]
        d1wb, d2wb = d1.bar_components[0], d2.bar_components[0]

        V = rotation_2x2(self.th)
        sub = Rotation([d1w[i], d2w[i]], [D1[i], D2[i]], V).substitution()
        sub.update(Rotation([d1wb[i], d2wb[i]], [D1b[i], D2b[i]], V)
                   .substitution())
        L = current_builder(u1, u2, d1, d2)
        handles = dict(u1=u1, u2=u2, D1=D1, D2=D2, D1b=D1b, D2b=D2b)
        return sp.expand(L.xreplace(sub)), handles

    def test_mixing_enters_the_W_current(self):
        gw, Wp = sp.Symbol("gw", positive=True), sp.Symbol("Wp")

        def W_current(u1, u2, d1, d2):
            u1b, u2b = u1.bar_components[0], u2.bar_components[0]
            d1w, d2w = d1.components[0], d2.components[0]
            return gw / sp.sqrt(2) * Wp * (
                Bilinear(u1b[i], gamma_L, d1w[i])
                + Bilinear(u2b[i], gamma_L, d2w[i]))

        L, h = self._rotated(W_current)
        table = extract_fermion_vertices(L, [Wp])
        u1b = h["u1"].bar_components[0]
        c, s = sp.cos(self.th), sp.sin(self.th)
        # ū1 couples to BOTH mass eigenstates: V_11 ∝ cos, V_12 ∝ −sin
        diag = table[(u1b[i], gamma_L, h["D1"][i])][1][(Wp,)]
        off = table[(u1b[i], gamma_L, h["D2"][i])][1][(Wp,)]
        assert sp.simplify(diag - gw / sp.sqrt(2) * c) == 0
        assert sp.simplify(off + gw / sp.sqrt(2) * s) == 0

    def test_no_neutral_current_fcnc(self):
        gz, Zb = sp.Symbol("gz", positive=True), sp.Symbol("Zb")

        def Z_current(u1, u2, d1, d2):
            d1w, d2w = d1.components[0], d2.components[0]
            d1wb, d2wb = d1.bar_components[0], d2.bar_components[0]
            return gz * Zb * (Bilinear(d1wb[i], gamma_L, d1w[i])
                              + Bilinear(d2wb[i], gamma_L, d2w[i]))

        L, h = self._rotated(Z_current)
        table = extract_fermion_vertices(L, [Zb])
        # diagonal survives, off-diagonal cancels exactly (GIM)
        assert sp.simplify(
            table[(h["D1b"][i], gamma_L, h["D1"][i])][1][(Zb,)] - gz) == 0
        off_key = (h["D1b"][i], gamma_L, h["D2"][i])
        assert off_key not in table or sp.simplify(
            table[off_key].get(1, {}).get((Zb,), 0)) == 0


class TestStandardParametrization:
    def test_exactly_unitary(self):
        params, V = standard_ckm()
        sub = {p.s: p.expr for p in params if getattr(p, "expr", None) is not None}
        Vx = V.subs(sub)
        VdV = Vx.conjugate().T * Vx
        for a in range(3):
            for b in range(3):
                expected = 1 if a == b else 0
                assert sp.simplify(VdV[a, b]) == expected, (a, b, VdV[a, b])

    def test_pdg_magnitudes_and_phase(self):
        params, V = standard_ckm()
        sub = {p.s: p.expr for p in params if getattr(p, "expr", None) is not None}
        angles = {p.s: p.value for p in params
                  if getattr(p, "expr", None) is None}
        Vx = V.subs(sub).subs(angles)
        assert abs(abs(complex(Vx[0, 0])) - 0.9744) < 1e-3   # |Vud|
        assert abs(abs(complex(Vx[0, 1])) - 0.2250) < 1e-3   # |Vus|
        assert abs(abs(complex(Vx[1, 2])) - 0.0418) < 1e-3   # |Vcb|
        # the CP phase lives in Vub (and Vtd): nonzero imaginary part
        assert abs(complex(Vx[0, 2]).imag) > 1e-4


class TestInsertedWVertex:
    def test_W_ud_vertex_is_ig_over_root2_Vud(self):
        gw, Wp = sp.Symbol("gw", positive=True), sp.Symbol("Wp")
        params, V = standard_ckm()
        uL = WeylFermion("uL", reps={}, chirality="L", nflavors=3,
                         component_names=["uL"])
        dL = WeylFermion("dL", reps={}, chirality="L", nflavors=3,
                         component_names=["dL"])
        ubar = uL.bar_components[0]
        dfield = dL.components[0]
        # explicit charged current with CKM inserted, concrete flavour indices
        L = gw / sp.sqrt(2) * Wp * sum(
            V[a, b] * Bilinear(ubar[a], gamma_L, dfield[b])
            for a in range(3) for b in range(3))
        table = extract_fermion_vertices(sp.expand(L), [Wp])
        coeff = table[(ubar[0], gamma_L, dfield[0])][1][(Wp,)]
        rule = fermion_feynman_rule(coeff, gamma_L, (Wp,))
        assert sp.simplify(rule - sp.I * gw / sp.sqrt(2) * V[0, 0] * gamma_L) == 0


class TestCKMUFOExport:
    def test_complex_ckm_internals_roundtrip(self, tmp_path):
        gw = ExternalParameter("gw", 0.6535, positive=True)
        params, V = standard_ckm()
        Vud = next(p for p in params if p.name == "Vud")

        Wp, Wm = sp.symbols("Wp Wm")
        u, ubar = sp.symbols("u ubar")
        d, dbar = sp.symbols("d dbar")
        particles = [
            UFOParticle(Wp, 24, "W+", antiname="W-", spin=3, charge=1,
                        antisymbol=Wm),
            UFOParticle(u, 2, "u", antiname="u~", spin=2,
                        charge=sp.Rational(2, 3), antisymbol=ubar),
            UFOParticle(d, 1, "d", antiname="d~", spin=2,
                        charge=sp.Rational(-1, 3), antisymbol=dbar),
        ]
        fermion_vertices = [
            dict(bar=ubar, field=d, bosons=(Wp,),
                 left=gw.s / sp.sqrt(2) * Vud.s),
        ]
        out = tmp_path / "CKM_UFO"
        write_ufo(out, "CKMtest", ParameterSet(gw, *params), particles,
                  fermion_vertices=fermion_vertices)

        report = verify_ufo_numeric(out)
        assert report.ok, report.failures
        # the 9 CKM elements are emitted as complex internals and evaluate
        assert abs(abs(report.parameters["Vud"]) - 0.9744) < 1e-3
        assert abs(report.parameters["Vub"].imag) > 1e-4
