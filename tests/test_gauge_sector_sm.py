"""Phase 3 validation: SM electroweak gauge sector.

Pinned physics (Peskin/Romão conventions, CONVENTIONS.md signs):
- m_W = g v / 2;  m_Z² = (g² + g'²) v²/4;  photon massless
- Weinberg rotation: tan θ_W = g'/g diagonalizes the (W3, B) block
- hWW vertex = i g m_W = i g² v/2  (× g^{μν})
- hZZ vertex = i (g² + g'²) v / 2  (× g^{μν})
- hhWW vertex = i g²/2;  no photon–Higgs couplings
- Z(h ∂G0) VSS vertex ∝ (p(h) − p(G0)), coefficient √(g²+g'²)/2
- γW⁺W⁻ cubic coupling magnitude = e = g sinθ_W; ZW⁺W⁻ = g cosθ_W
- W⁺W⁻W⁺W⁻ quartic ∝ g²; γγW⁺W⁻ ∝ e²
"""

import sympy as sp
import pytest

from feynlag import (
    Dmu, ExternalParameter, InternalParameter, Lagrangian, Model, Rotation,
    SU2, Scalar, U1, conjugate_pair, cubic_couplings, dag,
    diagonalize_orthogonal_2x2, momentum, quartic_couplings, rotation_2x2,
)


@pytest.fixture(scope="module")
def sm():
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
    L_kin = (dag(DH) * DH)[0]

    L = Lagrangian()
    L.add(L_kin, sector="kinetic")
    L.add(-V, sector="potential")

    model = Model("SM-EW", gauge_groups=[SU2L, U1Y], fields=[H, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, lam, mu2], lagrangian=L)
    model.solve_tadpoles([mu2])
    return model, SU2L, U1Y, H, v, gw, g1


def _w_components(SU2L):
    return SU2L.bosons().components


def test_gauge_mass_matrix(sm):
    model, SU2L, U1Y, H, v, gw, g1 = sm
    W1, W2, W3 = _w_components(SU2L)
    B = U1Y.bosons().components[0]
    g, gp = gw.s, g1.s

    M2 = model.gauge_mass_matrix([W1, W2, W3, B])

    # m_W² = g²v²/4 for W1, W2, diagonal
    assert sp.simplify(M2[0, 0] - g**2 * v.s**2 / 4) == 0
    assert sp.simplify(M2[1, 1] - g**2 * v.s**2 / 4) == 0
    assert M2[0, 1] == 0 and M2[0, 2] == 0 and M2[0, 3] == 0

    # (W3, B) block: singular, trace = m_Z²
    block = M2[2:4, 2:4]
    assert sp.simplify(block.det()) == 0
    assert sp.simplify(sp.trace(block) - (g**2 + gp**2) * v.s**2 / 4) == 0
    # off-diagonal −g g' v²/4
    assert sp.simplify(block[0, 1] + g * gp * v.s**2 / 4) == 0


def test_weinberg_rotation(sm):
    model, SU2L, U1Y, H, v, gw, g1 = sm
    _, _, W3 = _w_components(SU2L)
    B = U1Y.bosons().components[0]
    g, gp = gw.s, g1.s

    block = model.gauge_mass_matrix([W3, B])
    Z, A = sp.symbols("Z A", real=True)
    # standard convention: Z = cw W3 − sw B, A = sw W3 + cw B
    thetaW = sp.atan(gp / g)
    rot = Rotation([W3, B], [Z, A], rotation_2x2(-thetaW))

    ok, residuals = rot.check(block)
    assert ok, residuals
    masses = rot.masses_squared(block)
    assert sp.simplify(masses[0] - (g**2 + gp**2) * v.s**2 / 4) == 0  # m_Z²
    assert sp.simplify(masses[1]) == 0                                # photon


@pytest.fixture(scope="module")
def physical(sm):
    """SM with all rotations to the physical basis registered."""
    model, SU2L, U1Y, H, v, gw, g1 = sm
    g, gp = gw.s, g1.s
    W1, W2, W3 = _w_components(SU2L)
    B = U1Y.bosons().components[0]

    Z, A = sp.symbols("Z A", real=True)
    thetaW = sp.atan(gp / g)
    model.rotate(Rotation([W3, B], [Z, A], rotation_2x2(-thetaW)))

    # W± = (W1 ∓ i W2)/√2  (unitary change of basis)
    Wp, Wm = sp.symbols("Wp Wm")
    U = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], U, kind="unitary"))

    h = sp.Symbol("H0_r", real=True)
    G0 = sp.Symbol("H0_i", real=True)
    Gp = H.components[0]
    Gm, cmap = conjugate_pair(Gp, "Gm")
    fields = [h, G0, Gp, Gm, Z, A, Wp, Wm]
    return model, fields, cmap, dict(h=h, G0=G0, Gp=Gp, Gm=Gm, Z=Z, A=A,
                                     Wp=Wp, Wm=Wm, g=g, gp=gp, v=v.s)


def _rule(rules, *fields):
    key = tuple(sorted(fields, key=lambda s: s.sort_key()))
    assert key in rules, f"missing vertex {key}; have {list(rules)}"
    return rules[key]


@pytest.fixture(scope="module")
def kinetic_rules(physical):
    model, fields, cmap, s = physical
    return model.feynman_rules(fields, sector="kinetic", conjugate_map=cmap,
                               simplifier=sp.simplify)


def test_hWW_coupling(kinetic_rules, physical):
    _, _, _, s = physical
    rule = _rule(kinetic_rules, s["h"], s["Wp"], s["Wm"])
    # i g m_W = i g² v / 2   (g^{μν} implicit)
    assert sp.simplify(rule - sp.I * s["g"] ** 2 * s["v"] / 2) == 0


def test_hZZ_coupling(kinetic_rules, physical):
    _, _, _, s = physical
    rule = _rule(kinetic_rules, s["h"], s["Z"], s["Z"])
    assert sp.simplify(rule - sp.I * (s["g"] ** 2 + s["gp"] ** 2) * s["v"] / 2) == 0


def test_hhWW_and_hhZZ_couplings(kinetic_rules, physical):
    _, _, _, s = physical
    rule = _rule(kinetic_rules, s["h"], s["h"], s["Wp"], s["Wm"])
    assert sp.simplify(rule - sp.I * s["g"] ** 2 / 2) == 0
    rule = _rule(kinetic_rules, s["h"], s["h"], s["Z"], s["Z"])
    assert sp.simplify(rule - sp.I * (s["g"] ** 2 + s["gp"] ** 2) / 2) == 0


def test_no_photon_higgs_couplings(kinetic_rules, physical):
    _, _, _, s = physical
    A, h = s["A"], s["h"]
    for key in kinetic_rules:
        if A in key and h in key and s["Gp"] not in key and s["Gm"] not in key:
            pytest.fail(f"forbidden photon–Higgs vertex {key}: "
                        f"{kinetic_rules[key]}")


def test_photon_charged_goldstone_is_QED(kinetic_rules, physical):
    """γ G⁺ G⁻ must couple with e = g g'/√(g²+g'²) times (p(G+) − p(G−))."""
    _, _, _, s = physical
    rule = _rule(kinetic_rules, s["A"], s["Gp"], s["Gm"])
    e = s["g"] * s["gp"] / sp.sqrt(s["g"] ** 2 + s["gp"] ** 2)
    pGp, pGm = momentum(s["Gp"]), momentum(s["Gm"])
    # scalar-QED vertex: ± i e (p(G+) − p(G−)) — antisymmetric in momenta
    expected = sp.I * e * (pGp - pGm)
    ratio = sp.simplify(rule / expected)
    assert ratio in (1, -1), f"γG+G- rule {rule} vs QED {expected}"


def test_ZhG0_vss_structure(kinetic_rules, physical):
    """Z h G0 vertex: antisymmetric (p(h) − p(G0)) with |coeff| = √(g²+g'²)/2."""
    _, _, _, s = physical
    rule = _rule(kinetic_rules, s["Z"], s["h"], s["G0"])
    gz = sp.sqrt(s["g"] ** 2 + s["gp"] ** 2)
    expected = gz / 2 * (momentum(s["h"]) - momentum(s["G0"]))
    ratio = sp.simplify(rule / expected)
    assert ratio in (1, -1, sp.I, -sp.I), f"ZhG0 rule: {rule}"


def test_cubic_gauge_couplings(physical):
    """γW⁺W⁻ = e, ZW⁺W⁻ = g cosθ_W (magnitudes of the F-tensor couplings)."""
    model, fields, cmap, s = physical
    SU2L = model.gauge_groups[0]
    g, gp = s["g"], s["gp"]

    # A^a = Σ_i U[a,i] V_i over physical (Wp, Wm, Z, A)
    cw, sw = g / sp.sqrt(g**2 + gp**2), gp / sp.sqrt(g**2 + gp**2)
    U = sp.Matrix([
        [1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0],     # W1
        [sp.I / sp.sqrt(2), -sp.I / sp.sqrt(2), 0, 0],  # W2 (W± = (W1∓iW2)/√2)
        [0, 0, cw, sw],                              # W3
    ])
    physical_bosons = [s["Wp"], s["Wm"], s["Z"], s["A"]]
    cubic = cubic_couplings(SU2L, physical=physical_bosons, U=U)

    e = g * sw
    gAWW = cubic.get((s["A"], s["Wp"], s["Wm"]), 0)
    gZWW = cubic.get((s["Z"], s["Wp"], s["Wm"]), 0)
    assert sp.simplify(sp.Abs(gAWW) - e) == 0, gAWW
    assert sp.simplify(sp.Abs(gZWW) - g * cw) == 0, gZWW
    # no γγ or ZZ pairs with a single W
    assert (s["A"], s["A"], s["Wp"]) not in cubic

    # total antisymmetry under leg exchange
    assert sp.simplify(cubic[(s["Wp"], s["A"], s["Wm"])] + gAWW) == 0


def test_quartic_gauge_couplings(physical):
    model, fields, cmap, s = physical
    SU2L = model.gauge_groups[0]
    g, gp = s["g"], s["gp"]
    cw, sw = g / sp.sqrt(g**2 + gp**2), gp / sp.sqrt(g**2 + gp**2)
    U = sp.Matrix([
        [1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0],
        [sp.I / sp.sqrt(2), -sp.I / sp.sqrt(2), 0, 0],
        [0, 0, cw, sw],
    ])
    physical_bosons = [s["Wp"], s["Wm"], s["Z"], s["A"]]
    quartic = quartic_couplings(SU2L, physical=physical_bosons, U=U)

    # WWWW present with strength ∝ g², γγWW ∝ e² = g² sw²
    wwww = quartic.get((s["Wp"], s["Wm"], s["Wp"], s["Wm"]), 0)
    assert sp.simplify(wwww / g**2).is_number and wwww != 0
    aaww = quartic.get((s["A"], s["Wp"], s["A"], s["Wm"]), 0)
    assert sp.simplify(aaww / (g * sw) ** 2).is_number and aaww != 0
    # no 4-photon vertex
    assert (s["A"], s["A"], s["A"], s["A"]) not in quartic


def test_vertex_objects_classified(physical):
    model, fields, cmap, s = physical
    verts = model.vertices(fields, sector="kinetic", conjugate_map=cmap,
                           simplifier=sp.simplify)
    types = {v.vertex_type for v in verts}
    assert types <= {"SSS", "SSSS", "VSS", "VVS", "VVSS"}
    by_type = {t: [v for v in verts if v.vertex_type == t] for t in types}
    assert "VVS" in types and "VVSS" in types and "VSS" in types
    # every VVS coupling is momentum-free; every VSS coupling carries momenta
    for vert in by_type["VVS"]:
        assert not vert.coupling.has(momentum)
    for vert in by_type["VSS"]:
        assert vert.coupling.has(momentum)
