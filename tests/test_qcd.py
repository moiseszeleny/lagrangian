"""SU(3) color / QCD vertex-dynamics validation.

Everything the vertex-extraction pipeline needs (Dmu, fermion_gauge_current,
cubic_couplings) is already generic over gauge group (see CLAUDE.md) — this
file is the first thing anywhere in the codebase to actually exercise that
generic machinery with SU(3), pinning:

- gauge invariance of a color-triplet fermion's kinetic/current term,
- the qqg coupling (T^1_{01} = 1/2 in the fundamental, from the Gell-Mann
  matrices), consistent with this project's own D_mu = d_mu - i g T^a A^a_mu
  sign convention (already used for the SU(2) W/Z lepton currents),
- the ggg coupling (f^123 = 1) and its total antisymmetry under leg
  exchange, mirroring test_gauge_sector_sm.py::test_cubic_gauge_couplings.
- the gggg (4-gluon) coupling, assembled into the 3 UFO VVVV1/2/3 Lorentz
  structures by export.ufo.vvvv.assemble_vvvv — the "Phase 5" step
  yangmills.py's quartic_couplings() docstring deferred and that was never
  built or pinned for any gauge group before now (see
  tests/test_yangmills.py for the independent first-principles derivation
  this value is cross-checked against).
"""

import sympy as sp
import pytest

from feynlag import (
    DiracGamma, ExternalParameter, Lagrangian, Model, SU3, WeylFermion,
    cubic_couplings, diracPL, extract_fermion_vertices, fermion_gauge_current,
    quartic_couplings,
)
from feynlag.export.ufo.vvvv import assemble_vvvv

i = sp.Symbol("fl_i", integer=True)


@pytest.fixture(scope="module")
def qcd():
    gs = ExternalParameter("gs", 1.22, positive=True)
    SU3c = SU3("SU3c", coupling=gs)
    q = WeylFermion("q", reps={SU3c: 3}, chirality="L", nflavors=1,
                    component_names=["q_1", "q_2", "q_3"])
    G = SU3c.bosons("G")
    return SU3c, q, G, gs


def test_quark_gluon_gauge_invariance(qcd):
    """First-ever exercise of Model.check_invariance() with a color-triplet
    field through the full Model path (SU(2)/U(1) fermion currents are
    already covered by test_invariance.py; this is the SU(3) analogue)."""
    SU3c, q, G, gs = qcd
    current = fermion_gauge_current(q, i)
    L = Lagrangian().add(current, sector="yukawa")
    model = Model("QCD-quark", gauge_groups=[SU3c], fields=[q, G],
                  parameters=[gs], lagrangian=L)
    report = model.check_invariance()
    assert report.ok, report.failures


def test_qqg_coupling_pinned(qcd):
    """T^1_{0,1} = 1/2 (lambda^1/2 in the fundamental) => the (qbar_0, q_1)
    current at gluon G_1 has coefficient +gs/2, matching this project's own
    D_mu = d_mu - i g T^a A^a_mu sign convention (already pinned for the W/Z
    lepton currents in test_fermion_sector.py) — not to be second-guessed
    against an external textbook's differently-signed D_mu convention."""
    SU3c, q, G, gs = qcd
    current = fermion_gauge_current(q, i)
    G1 = G.components[0]
    table = extract_fermion_vertices(current, [G1])
    mu = sp.Symbol("mu", integer=True)
    gamma_L = DiracGamma(mu) * diracPL
    qbar, qc = q.bar_components, q.components
    key = (qbar[0][i], gamma_L, qc[1][i])
    coeff = table[key][1][(G1,)]
    assert sp.simplify(coeff - gs.s / 2) == 0


def test_ggg_coupling_pinned(qcd):
    """f^123 = 1 (Gell-Mann structure constant) => g_1 g_2 g_3 coupling is
    exactly -gs; total antisymmetry under leg exchange; no g_1 g_1 g_2
    vertex (manifestly zero triple)."""
    SU3c, q, G, gs = qcd
    cubic = cubic_couplings(SU3c)
    G1, G2, G3 = G.components[0], G.components[1], G.components[2]

    ggg = cubic.get((G1, G2, G3), 0)
    assert sp.simplify(ggg + gs.s) == 0

    # total antisymmetry under leg exchange
    assert sp.simplify(cubic[(G2, G1, G3)] + ggg) == 0
    assert sp.simplify(cubic[(G1, G3, G2)] + ggg) == 0

    assert (G1, G1, G2) not in cubic


def test_gggg_coupling_pinned(qcd):
    """4-gluon self-coupling for the quadruple (G_1,G_2,G_4,G_5) (adjoint
    indices 0,1,3,4), assembled by assemble_vvvv — cross-checked in
    tests/test_yangmills.py against an independent direct differentiation
    of -1/4 F^a F^a, not merely re-derived from the same code path here."""
    SU3c, q, G, gs = qcd
    quartic = quartic_couplings(SU3c)
    G1, G2, G4, G5 = (G.components[0], G.components[1], G.components[3],
                      G.components[4])
    gggg = assemble_vvvv(quartic, (G1, G2, G4, G5))

    assert sp.simplify(gggg["VVVV1"] - sp.Rational(3, 2) * gs.s ** 2) == 0
    assert sp.simplify(gggg["VVVV2"] - sp.Rational(3, 4) * gs.s ** 2) == 0
    assert sp.simplify(gggg["VVVV3"] + sp.Rational(3, 4) * gs.s ** 2) == 0
