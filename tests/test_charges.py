"""Electric-charge registry, vacuum-derived operator, and the vertex checks.

Pins the physics: for an SM-lite electroweak model the charge operator derived
from the vacuum is Q ∝ T3 + Y (W± come out ±1 automatically from the W1/W2
rotation); the declared charges are consistent with it; every extracted vertex
conserves charge and pairs with its hermitian conjugate; and the guards fire on
a charge-breaking VEV, an unknown leg, and an ambiguous (rank-2) charge.
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, Rotation, SU2, U1,
    Scalar, Dmu, dag, conjugate_pair, rotation_2x2,
    ChargeRegistry, check_charge_conservation, check_charge_consistency,
    check_hermiticity_pairing, derive_charge_operator, physical_charges,
    Vertex,
)


@pytest.fixture
def smlite():
    """A rotated SM-lite electroweak model plus its physical-field handles."""
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    HdH = (dag(H) * H.mat)[0]
    V = -mu2.s * HdH + lam.s * HdH**2
    DH = Dmu(H)
    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-V, sector="potential")
    W, B = SU2L.bosons("W"), U1Y.bosons("B")
    model = Model("SMlite", gauge_groups=[SU2L, U1Y], fields=[H, W, B],
                  parameters=[gw, g1, v, lam, mu2], lagrangian=L)
    model.solve_tadpoles([mu2])

    W1, W2, W3 = W.components
    Bc = B.components[0]
    Z, A = sp.symbols("Z A", real=True)
    thetaW = sp.atan(g1.s / gw.s)
    model.rotate(Rotation([W3, Bc], [Z, A], rotation_2x2(-thetaW)))
    Wp, Wm = sp.symbols("Wp Wm")
    Um = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], Um, kind="unitary"))

    Gp = H.components[0]
    Gm, cmap = conjugate_pair(Gp, "Gm")
    h = sp.Symbol("H0_r", real=True)
    G0 = sp.Symbol("H0_i", real=True)
    handles = dict(model=model, H=H, Gp=Gp, Gm=Gm, cmap=cmap, h=h, G0=G0,
                   Z=Z, A=A, Wp=Wp, Wm=Wm, SU2L=SU2L, U1Y=U1Y,
                   fields=[h, G0, Gp, Gm, Z, A, Wp, Wm])
    return handles


def _registry(s):
    return ChargeRegistry(
        {s["h"]: 0, s["G0"]: 0, s["Gp"]: 1, s["Z"]: 0, s["A"]: 0,
         s["Wp"]: 1, s["Wm"]: -1}, conjugate_map=s["cmap"])


def _conjugates(s):
    return {s["Gp"]: s["Gm"], s["Gm"]: s["Gp"], s["Wp"]: s["Wm"],
            s["Wm"]: s["Wp"]}


def test_charge_operator_is_T3_plus_Y(smlite):
    coeffs = derive_charge_operator(smlite["model"])
    # one T3 (SU2 cartan) and one Y (U1) candidate, equal coefficients
    by_group = {g.name: c for (g, idx), c in coeffs.items()}
    assert by_group["SU2L"] == by_group["U1Y"] != 0


def test_W_charges_derived_automatically(smlite):
    coeffs = derive_charge_operator(smlite["model"])
    q = physical_charges(smlite["model"], coeffs)
    # normalise so hypercharge coefficient = 1 (already is); W± come from the
    # W1/W2 rotation, never declared
    assert q[smlite["Wp"]] == 1
    assert q[smlite["Wm"]] == -1
    assert q[smlite["Gp"]] == 1
    assert q[smlite["Z"]] == 0 and q[smlite["A"]] == 0


def test_declared_charges_consistent_with_vacuum(smlite):
    report = check_charge_consistency(smlite["model"], _registry(smlite))
    assert report.ok, report.mismatches


def test_mistuned_declared_charge_fails_consistency(smlite):
    s = smlite
    bad = ChargeRegistry(
        {s["h"]: 0, s["G0"]: 0, s["Gp"]: 2, s["Z"]: 0, s["A"]: 0,
         s["Wp"]: 1, s["Wm"]: -1}, conjugate_map=s["cmap"])
    report = check_charge_consistency(s["model"], bad)
    assert not report.ok
    assert any(sym == s["Gp"] for sym, _, _ in report.mismatches)


def test_all_vertices_conserve_charge(smlite):
    s = smlite
    verts = s["model"].vertices(s["fields"], conjugate_map=s["cmap"],
                                simplifier=sp.simplify)
    report = check_charge_conservation(_registry(s), bosonic_vertices=verts)
    assert report.ok, report.failures
    assert report.n_checked > 10


def test_hand_built_charge_violating_vertex_fails(smlite):
    s = smlite
    bad = Vertex((s["h"], s["Wp"], s["Wp"]), sp.I, "VVS")
    report = check_charge_conservation(_registry(s), bosonic_vertices=[bad])
    assert not report.ok


def test_all_vertices_pair_hermitian(smlite):
    s = smlite
    verts = s["model"].vertices(s["fields"], conjugate_map=s["cmap"],
                                simplifier=sp.simplify)
    report = check_hermiticity_pairing(bosonic_vertices=verts,
                                       conjugates=_conjugates(s))
    assert report.ok, report.failures
    assert report.n_checked > 10


def test_dropped_conjugate_vertex_fails_pairing(smlite):
    s = smlite
    verts = s["model"].vertices(s["fields"], conjugate_map=s["cmap"],
                                simplifier=sp.simplify)
    # drop one charged vertex so its partner is orphaned
    dropped = None
    for i, v in enumerate(verts):
        if s["Wp"] in v.particles and s["Gp"] not in v.particles:
            dropped = i
            break
    assert dropped is not None
    report = check_hermiticity_pairing(bosonic_vertices=verts[:dropped]
                                       + verts[dropped + 1:],
                                       conjugates=_conjugates(s))
    assert not report.ok


def test_fermion_vertex_charge_conservation():
    """A W⁺ lepton current ν̄ γ e conserves charge (0 − 1 + 1 = 0); a
    wrong-charge boson breaks it."""
    from feynlag import WeylFermion
    from feynlag.dirac import DiracGamma, diracPL

    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", component_names=["nuL", "eL"])
    nuL, eL = Ll.components
    i = sp.Symbol("i", integer=True)
    mu = sp.Symbol("mu", integer=True)
    gamma = DiracGamma(mu) * diracPL
    Wp, Zp = sp.symbols("Wp Zp")

    good = {(Ll.bar(nuL)[i], gamma, eL[i]): {1: {(Wp,): sp.Symbol("c")}}}
    reg = ChargeRegistry({nuL: 0, eL: -1, Wp: 1})
    assert check_charge_conservation(reg, fermion_table=good).ok

    bad = {(Ll.bar(nuL)[i], gamma, eL[i]): {1: {(Zp,): sp.Symbol("c")}}}
    reg_bad = ChargeRegistry({nuL: 0, eL: -1, Zp: 5})
    assert not check_charge_conservation(reg_bad, fermion_table=bad).ok


def test_unknown_leg_raises(smlite):
    reg = _registry(smlite)
    with pytest.raises(KeyError):
        reg.charge_of(sp.Symbol("mystery"))


def test_charge_breaking_vacuum_raises():
    """VEVs in BOTH doublet components leave no unbroken U(1) — there is no
    electric charge, so the derivation raises."""
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Hp", "H0"])
    H.expand_vev({H.components[0]: ExternalParameter("u", 1.0),
                  H.components[1]: ExternalParameter("w", 1.0)})
    model = Model("bad", gauge_groups=[SU2L, U1Y], fields=[H])
    with pytest.raises(ValueError, match="no unbroken electric charge"):
        derive_charge_operator(model)


def test_ambiguous_charge_raises():
    """A spectator U(1) that the vacuum leaves unbroken makes electric charge
    ambiguous (rank-2 unbroken abelian) — flagged, not silently guessed."""
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    gx = ExternalParameter("gx", 0.3, positive=True)
    SU2L = SU2("SU2L", coupling=gw)
    U1Y, U1X = U1("U1Y", coupling=g1), U1("U1X", coupling=gx)
    # One doublet charged under BOTH U(1)s with a single VEV leaves two
    # independent unbroken combinations (T3+Y and T3+X), so electric charge
    # is not uniquely defined.
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2), U1X: sp.Rational(1, 2)},
               component_names=["Hp", "H0"])
    H.expand_vev({H.components[1]: ExternalParameter("v", 1.0)})
    model = Model("2u1", gauge_groups=[SU2L, U1Y, U1X], fields=[H])
    with pytest.raises(ValueError, match="ambiguous"):
        derive_charge_operator(model)
