"""Tests for the reusable SM builders in ``feynlag.models``.

House style: pin physics.  The scaffold + rotations must reproduce the same
electroweak masses and invariance the hand-built examples did, and a refactored
example must reproduce its previously-pinned physics result — proving the
modularization is behaviour-preserving, not merely importable.
"""

import sympy as sp

from feynlag import (
    Lagrangian, Model, WeylFermion, fermion_gauge_current,
)
from feynlag.models import (
    ElectroweakScaffold, PhysicalBasis, charged_current_rotation,
    electroweak_scaffold, higgs_doublet, standard_model, to_physical_basis,
    weinberg_rotation,
)


def test_scaffold_gauge_masses():
    """m_W² = g²v²/4, m_Z² = (g²+g'²)v²/4, photon massless."""
    sm = standard_model(generations=1)
    g, gp, v = sm.scaffold.gw.s, sm.scaffold.g1.s, sm.scaffold.v.s
    W1, W2, W3 = sm.scaffold.SU2L.bosons().components
    B = sm.scaffold.U1Y.bosons().components[0]
    M2 = sm.model.gauge_mass_matrix([W1, W2, W3, B])
    assert sp.simplify(M2[0, 0] - g**2 * v**2 / 4) == 0
    assert sp.simplify(M2[1, 1] - g**2 * v**2 / 4) == 0
    block = M2[2:4, 2:4]
    assert sp.simplify(sp.trace(block) - (g**2 + gp**2) * v**2 / 4) == 0
    assert sp.simplify(block.det()) == 0        # massless photon


def test_scaffold_higgs_mass():
    """The scaffold's Higgs potential gives m_h² = 2λv² after tadpole solving."""
    scaffold = electroweak_scaffold()
    L = Lagrangian()
    scaffold.add_higgs(L)
    model = Model("higgs_only", gauge_groups=scaffold.gauge_groups,
                  fields=scaffold.fields, parameters=scaffold.parameters,
                  lagrangian=L)
    model.solve_tadpoles([scaffold.mu2])
    h = scaffold.H.vev_expansions[scaffold.H.components[1]][1]
    M2 = model.mass_matrix([h])
    lam, v = scaffold.lam.s, scaffold.v.s
    assert sp.simplify(M2[0, 0] - 2 * lam * v**2) == 0


def test_physical_basis_handles():
    """to_physical_basis returns the same eight physical boson handles the
    hand-built examples used, and the Goldstone conjugate map."""
    sm = standard_model(generations=1)
    pb = sm.physical
    assert isinstance(pb, PhysicalBasis)
    assert [str(b) for b in pb.bosons] == \
        ["H0_r", "H0_i", "Gp", "Gm", "Z", "A", "Wp", "Wm"]
    assert pb.cmap == {sp.conjugate(pb.Gp): pb.Gm}


def test_standard_model_is_gauge_invariant():
    sm = standard_model(generations=1, physical_basis=False)
    report = sm.model.check_invariance()
    assert report.ok, report


def test_weinberg_rotation_custom_symbol_composes():
    """The Weinberg primitive accepts an intermediate output symbol (the
    U(1)_X-style chained case) without clobbering, and a second rotation can
    consume it."""
    scaffold = electroweak_scaffold()
    L = Lagrangian()
    scaffold.add_higgs(L)
    model = Model("chain", gauge_groups=scaffold.gauge_groups,
                  fields=scaffold.fields, parameters=scaffold.parameters,
                  lagrangian=L)
    model.solve_tadpoles([scaffold.mu2])
    Z0, A = weinberg_rotation(model, scaffold.SU2L, scaffold.U1Y, z="Z0")
    assert str(Z0) == "Z0" and str(A) == "A"
    Wp, Wm = charged_current_rotation(model, scaffold.SU2L)
    assert str(Wp) == "Wp" and str(Wm) == "Wm"
    # both rotations registered
    assert len(model.rotations) == 2


def test_multi_generation():
    """Three generations build and stay gauge invariant."""
    sm = standard_model(generations=3, physical_basis=False)
    assert len(sm.leptons) == 3
    assert sm.model.check_invariance().ok


def test_refactored_sm_decays_reproduces_z_width():
    """The refactored example still yields Γ(Z→νν̄) = m_Z(g²+g'²)/96π — the
    modularization is behaviour-preserving, not just importable."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parent.parent / "examples" / "sm_decays.py"
    spec = importlib.util.spec_from_file_location("sm_decays_ex", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    s = mod.build_model()
    from feynlag.pheno import DecayCalculator
    mZ = sp.Symbol("m_Z", positive=True)
    masses = {s["Z"]: mZ, s["Wp"]: sp.Symbol("m_W", positive=True),
              s["Wm"]: sp.Symbol("m_W", positive=True), s["A"]: 0,
              s["h"]: sp.Symbol("m_h", positive=True),
              s["tau"]: sp.Symbol("m_tau", positive=True),
              s["taubar"]: sp.Symbol("m_tau", positive=True),
              s["nu"]: 0, s["nubar"]: 0}
    calc = DecayCalculator(s["model"], masses, boson_fields=s["bosons"],
                           fermion_sectors=("gauge", "yukawa"),
                           conjugate_map=s["conjugate_map"],
                           particle_map=s["particle_map"])
    width = calc.partial_widths(s["Z"])[(s["nubar"], s["nu"])]
    g, gp = s["gw"].s, s["g1"].s
    assert sp.simplify(width - mZ * (g**2 + gp**2) / (96 * sp.pi)) == 0
