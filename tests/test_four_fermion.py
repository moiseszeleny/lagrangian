"""Four-fermion (FFFF) effective operators — Phase C2.

Pins the dim-6 four-fermion track that lifts the one-`Bilinear`-per-term
restriction, using the muon-decay Fermi operator

    L ⊃ −(4 G_F/√2) (ν̄_μ γ^μ P_L μ)(ē γ_μ P_L ν_e) + h.c.

as the running example.  The design (see `vertices/bilinear.py` docstring) is
the **as-written bilinear basis** (no Fierz) restricted to **four distinct
fermion components**; repeated components raise.  Every check that already
handled one bilinear (charge conservation, hermiticity pairing, gauge
invariance, mass dimension) is re-pinned for the two-bilinear case.
"""

import importlib
import sys

import pytest
import sympy as sp

from feynlag import (
    Bilinear, DiracGamma, DiracGammaLower, ExternalParameter, InternalParameter,
    Lagrangian, Model, ParameterSet, Rotation, SU2, Scalar, U1, WeylFermion,
    check_gauge_invariance, check_hermiticity, check_mass_dimension,
    classify_spins, diracPL, diracPR, dag, expand_bilinear,
    extract_fermion_vertices, four_fermion_feynman_rule, rotation_2x2,
    verify_ufo_numeric,
)
from feynlag.charges import (
    ChargeRegistry, check_charge_conservation, check_hermiticity_pairing,
)
from feynlag.export.ufo import UFOParticle, write_ufo


# --------------------------------------------------------------------------
# shared fixtures: the two lepton doublets and the Fermi operator
# --------------------------------------------------------------------------

@pytest.fixture
def leptons():
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.35, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    Lmu = WeylFermion("Lmu", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                      chirality="L", component_names=["numuL", "muL"])
    Le = WeylFermion("Le", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", component_names=["nueL", "eL"])
    return dict(SU2L=SU2L, U1Y=U1Y, Lmu=Lmu, Le=Le)


@pytest.fixture
def fermi(leptons):
    """The muon-decay operator and its ingredients."""
    Lmu, Le = leptons["Lmu"], leptons["Le"]
    GF = ExternalParameter("G_F", 1.16637e-5, positive=True, unit_dim=-2)
    i = sp.Symbol("i", integer=True)
    mu = sp.Symbol("mu", integer=True)
    gL_up = DiracGamma(mu) * diracPL       # γ^μ P_L
    gL_dn = DiracGammaLower(mu) * diracPL   # γ_μ P_L (shared μ ⇒ contraction)
    numuLbar = Lmu.bar_components[0]
    muL = Lmu.components[1]
    eLbar = Le.bar_components[1]
    nueL = Le.components[0]
    B1 = Bilinear(numuLbar[i], gL_up, muL[i])
    B2 = Bilinear(eLbar[i], gL_dn, nueL[i])
    op = -(4 * GF.s / sp.sqrt(2)) * B1 * B2
    return dict(GF=GF, op=op, B1=B1, B2=B2, i=i, gL_up=gL_up, gL_dn=gL_dn,
                numuLbar=numuLbar, muL=muL, eLbar=eLbar, nueL=nueL, **leptons)


# --------------------------------------------------------------------------
# extraction + Feynman rule
# --------------------------------------------------------------------------

def test_fermi_extraction_nested_key_and_coefficient(fermi):
    """Two-bilinear term → one nested key with coefficient −4 G_F/√2."""
    table = extract_fermion_vertices(fermi["op"], [])
    assert len(table) == 1
    key = next(iter(table))
    assert isinstance(key[0], tuple), "FFFF key must be a nested pair of subkeys"
    assert len(key) == 2
    coeff = table[key][0][()]                # n_bosons=0, no boson legs
    assert sp.simplify(coeff - (-4 * fermi["GF"].s / sp.sqrt(2))) == 0


def test_fermi_key_order_is_canonical(fermi):
    """B₁·B₂ and B₂·B₁ group under the same canonically sorted key."""
    t12 = extract_fermion_vertices(fermi["B1"] * fermi["B2"], [])
    t21 = extract_fermion_vertices(fermi["B2"] * fermi["B1"], [])
    assert set(t12) == set(t21)


def test_four_fermion_feynman_rule_scalar(fermi):
    """The rule is the scalar i·coeff·∏mult!, Dirac chains carried separately."""
    table = extract_fermion_vertices(fermi["op"], [])
    coeff = next(iter(table.values()))[0][()]
    rule = four_fermion_feynman_rule(coeff, (fermi["gL_up"], fermi["gL_dn"]))
    assert sp.simplify(rule - sp.I * (-4 * fermi["GF"].s / sp.sqrt(2))) == 0
    # gammas are validated but never folded into the scalar
    assert not rule.has(DiracGamma)


def test_four_fermion_rule_rejects_wrong_chain_count(fermi):
    with pytest.raises(ValueError):
        four_fermion_feynman_rule(fermi["GF"].s, (fermi["gL_up"],))


# --------------------------------------------------------------------------
# distinct-legs guard (the Fierz boundary)
# --------------------------------------------------------------------------

def test_repeated_component_raises(fermi):
    """(ēΓe)(ēΓ′e): a repeated fermion component needs Fierz → raise."""
    eLbar, nueL = fermi["eLbar"], fermi["nueL"]
    i = fermi["i"]
    B = Bilinear(eLbar[i], fermi["gL_up"], nueL[i])
    Bp = Bilinear(eLbar[i], fermi["gL_dn"], nueL[i])   # same two components
    with pytest.raises(NotImplementedError, match="repeated fermion component"):
        extract_fermion_vertices(B * Bp, [])


def test_squared_bilinear_raises(fermi):
    """A squared bilinear B² is two identical legs → raise (atoms() dedups)."""
    with pytest.raises(NotImplementedError, match="repeated fermion component"):
        extract_fermion_vertices(fermi["B1"] ** 2, [])


def test_three_bilinears_raise(fermi):
    """Six-fermion (three bilinears) is outside the catalog."""
    eLbar, nueL, i = fermi["eLbar"], fermi["nueL"], fermi["i"]
    B3 = Bilinear(eLbar[i], diracPL, nueL[i])
    with pytest.raises(NotImplementedError):
        extract_fermion_vertices(fermi["B1"] * fermi["B2"] * B3, [])


# --------------------------------------------------------------------------
# expand_bilinear distributes a two-bilinear product with a rotated leg
# --------------------------------------------------------------------------

def test_expand_bilinear_two_bilinears_rotated(fermi):
    """A mass-basis rotation on one leg leaves an Add inside a Bilinear slot;
    expand_bilinear must split it even inside a two-bilinear product (else
    extraction groups by an unsplit composite key — the VLL silent-garbage
    mode)."""
    muL, numuLbar, eLbar, nueL, i = (fermi["muL"], fermi["numuLbar"],
                                     fermi["eLbar"], fermi["nueL"], fermi["i"])
    m1, m2 = sp.IndexedBase("m1"), sp.IndexedBase("m2")
    c, s = sp.symbols("c s")
    # rotate muL → c·m1 + s·m2 inside B1
    B1_rot = Bilinear(numuLbar[i], fermi["gL_up"], c * m1[i] + s * m2[i])
    B2 = Bilinear(eLbar[i], fermi["gL_dn"], nueL[i])
    out = expand_bilinear(B1_rot * B2)
    # the product must split into two clean two-bilinear monomials
    terms = sp.Add.make_args(sp.expand(out))
    assert len(terms) == 2
    for t in terms:
        bils = t.atoms(Bilinear)
        assert len(bils) == 2
        for b in bils:
            assert not b.field.has(sp.Add) and not b.bar.has(sp.Add)


# --------------------------------------------------------------------------
# hermiticity
# --------------------------------------------------------------------------

def test_hermiticity_sector_passes_with_hc(fermi):
    """The Fermi operator + h.c. is hermitian as a Lagrangian sector."""
    L = fermi["op"] + sp.conjugate(fermi["op"])
    ok, residual = check_hermiticity(L)
    assert ok, residual


def test_hermiticity_pairing_finds_partner(fermi):
    """With + h.c., the two-bilinear key finds its conjugate partner."""
    L = fermi["op"] + sp.conjugate(fermi["op"])
    table = extract_fermion_vertices(L, [])
    report = check_hermiticity_pairing(fermion_table=table)
    assert report.ok, report.failures
    assert not report.skipped


def test_hermiticity_pairing_flags_missing_hc(fermi):
    """Without the + h.c., the partner key is absent → flagged."""
    table = extract_fermion_vertices(fermi["op"], [])
    report = check_hermiticity_pairing(fermion_table=table)
    assert not report.ok
    assert report.failures


# --------------------------------------------------------------------------
# charge conservation
# --------------------------------------------------------------------------

def _charge_registry(fermi):
    Lmu, Le = fermi["Lmu"], fermi["Le"]
    numuL, muL = Lmu.components
    nueL, eL = Le.components
    return ChargeRegistry({numuL: 0, muL: -1, nueL: 0, eL: -1})


def test_charge_conservation_fermi(fermi):
    """Σ Q = 0 across the four fermion legs of the muon-decay vertex."""
    table = extract_fermion_vertices(fermi["op"], [])
    report = check_charge_conservation(_charge_registry(fermi),
                                       fermion_table=table)
    assert report.ok, report.failures
    assert report.n_checked == 1


def test_charge_conservation_flags_violation(fermi):
    """A charge-violating four-fermion operator is caught."""
    # (ν̄_μ Γ μ)(ν̄_e Γ′ μ): net charge −1 + −1 = −2 ≠ 0 (bad)
    Lmu, Le, i = fermi["Lmu"], fermi["Le"], fermi["i"]
    numuLbar, muL = Lmu.bar_components[0], Lmu.components[1]
    nueLbar, muL2 = Le.bar_components[0], Lmu.components[1]
    # distinct components: use e-doublet's nu as second bar, mu twice would
    # repeat — instead pair with the electron field for distinctness
    eL = Le.components[1]
    B1 = Bilinear(numuLbar[i], fermi["gL_up"], muL[i])
    B2 = Bilinear(nueLbar[i], fermi["gL_dn"], eL[i])
    op = B1 * B2                    # charge: (0−1)+(0−1) = −2
    table = extract_fermion_vertices(op, [])
    report = check_charge_conservation(_charge_registry(fermi),
                                       fermion_table=table)
    assert not report.ok
    assert report.failures


# --------------------------------------------------------------------------
# mass dimension + the max_dim EFT flag
# --------------------------------------------------------------------------

def test_operator_is_dimension_six(fermi):
    """The four-fermion operator (dimensionless coefficient) reads as dim 6."""
    C = sp.Symbol("C")           # dimensionless Wilson coefficient
    op = C * fermi["B1"] * fermi["B2"]
    ok4, worst = check_mass_dimension(op, [fermi["Lmu"], fermi["Le"]],
                                      max_dim=4)
    assert worst == 6 and not ok4
    ok6, _ = check_mass_dimension(op, [fermi["Lmu"], fermi["Le"]], max_dim=6)
    assert ok6


def test_dimensionful_coefficient_makes_term_dim_four(fermi):
    """With G_F carrying its physical dim −2 the *term* is a complete dim-4
    Lagrangian term (both bookkeepings are supported and pinned)."""
    ok, worst = check_mass_dimension(
        fermi["op"], [fermi["Lmu"], fermi["Le"]], [fermi["GF"]], max_dim=4)
    assert worst == 4 and ok


def test_check_invariance_max_dim_threading(fermi):
    """Model.check_invariance(max_dim=…) gates the four-fermion operator."""
    C = ExternalParameter("C", 1.0)        # dimensionless (unit_dim=0)
    op = C.s * fermi["B1"] * fermi["B2"]
    L = Lagrangian()
    L.add(op + sp.conjugate(op), sector="other")
    model = Model("fourfermi", gauge_groups=[fermi["U1Y"]],
                  fields=[fermi["Lmu"], fermi["Le"]],
                  parameters=[C], lagrangian=L)
    rep4 = model.check_invariance(max_dim=4, hermiticity=False)
    assert not rep4.ok                      # dim-6 operator exceeds 4
    rep6 = model.check_invariance(max_dim=6, hermiticity=False)
    assert rep6.ok, rep6.failures


# --------------------------------------------------------------------------
# gauge invariance (tree-wide, unchanged by bilinear count)
# --------------------------------------------------------------------------

def test_gauge_invariance_tree_wide(fermi):
    """A U(1)_Y-invariant four-fermion operator passes check_gauge_invariance
    with no FFFF-specific code (every leg is transformed regardless of count)."""
    ok, viol = check_gauge_invariance(fermi["op"], [fermi["Lmu"], fermi["Le"]],
                                      fermi["U1Y"])
    assert ok, viol


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------

def test_classify_spins_accepts_ffff():
    fs = sp.symbols("a b c d")
    spins = {f: sp.Rational(1, 2) for f in fs}
    assert classify_spins(fs, spins) == "FFFF"


# --------------------------------------------------------------------------
# UFO export + round-trip
# --------------------------------------------------------------------------

_UFO_SUBMODULES = ["object_library", "function_library", "parameters",
                   "couplings", "particles", "lorentz", "vertices"]


def _import_ufo(path):
    sys.path.insert(0, str(path))
    for mod in _UFO_SUBMODULES:
        sys.modules.pop(mod, None)
    try:
        import object_library
        for mod in _UFO_SUBMODULES[1:]:
            importlib.import_module(mod)
        return object_library
    finally:
        sys.path.pop(0)


@pytest.fixture
def fermi_ufo(fermi, tmp_path):
    GF = fermi["GF"]
    MMU = ExternalParameter("MMU", 0.10566, positive=True, unit_dim=1)
    numu, numub, mm, mp, em, ep, nue, nueb = sp.symbols(
        "numu numub mm mp em ep nue nueb")
    particles = [
        UFOParticle(numu, 14, "vm", antiname="vm~", antisymbol=numub, spin=2),
        UFOParticle(mm, 13, "mu-", antiname="mu+", antisymbol=mp, spin=2,
                    mass="MMU", charge=-1),
        UFOParticle(em, 11, "e-", antiname="e+", antisymbol=ep, spin=2,
                    charge=-1),
        UFOParticle(nue, 12, "ve", antiname="ve~", antisymbol=nueb, spin=2),
    ]
    params = ParameterSet(GF, MMU)
    ffv = [{"bar1": numub, "field1": mm, "bar2": ep, "field2": nue,
            "couplings": {("VL", "VL"): -4 * GF.s / sp.sqrt(2)}}]
    out = tmp_path / "Fermi"
    write_ufo(out, "Fermi", params, particles, four_fermion_vertices=ffv)
    return out


def test_ufo_ffff_lorentz_emitted(fermi_ufo):
    lorentz = (fermi_ufo / "lorentz.py").read_text()
    assert "FFFFVLL" in lorentz
    assert "Gamma(-1,2,-2)*ProjM(-2,1)*Gamma(-1,4,-3)*ProjM(-3,3)" in lorentz


def test_ufo_ffff_vertex_has_four_legs(fermi_ufo):
    vertices = (fermi_ufo / "vertices.py").read_text()
    assert "L.FFFFVLL" in vertices
    # all four flavor-resolved legs present
    for name in ("P.vm", "P.mu__minus__", "P.e__plus__", "P.ve"):
        assert name in vertices


def test_ufo_ffff_uses_absolute_imports(fermi_ufo):
    for fname in ("lorentz.py", "vertices.py", "couplings.py"):
        assert "from ." not in (fermi_ufo / fname).read_text()


def test_ufo_ffff_roundtrip_and_feynman_i(fermi_ufo):
    """Round-trip evaluates the coupling to i·(−4 G_F/√2) — the Feynman-rule i
    must be carried (as in add_fermion_vertex)."""
    report = verify_ufo_numeric(fermi_ufo)
    assert report.ok, report.failures
    assert report.couplings
    gf = 1.16637e-5
    expected = 1j * (-4 * gf / sp.sqrt(2))
    values = list(report.couplings.values())
    assert any(abs(complex(v) - complex(expected)) < 1e-12 for v in values), \
        values
