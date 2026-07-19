"""Majorana infrastructure + the dim-5 Weinberg operator — Phase D.2.

Pins the charge-conjugation / Majorana-bilinear machinery and the neutrino-mass
physics it exists for: the Weinberg operator

    O_5 = (c_ij / Λ) (L_iᵀ C ε L_j)(Hᵀ ε H) + h.c.

which, after EWSB, gives the symmetric Majorana mass ``m_ν = −c v²/Λ`` (Takagi-
diagonalized) plus ``ν̄νh`` / ``ν̄νhh`` couplings.  The same machinery is reused
for a bare type-I-seesaw ``½ M_R ν_Rᵀ C ν_R``.
"""

import sympy as sp
import pytest

from feynlag import (
    Bilinear, DiracGamma, ExternalParameter, InternalParameter, Lagrangian,
    MajoranaBilinear, Model, SU2, Scalar, U1, WeylFermion, Dmu, dag, diracC,
    diracPL, diracPR, check_gauge_invariance, check_hermiticity,
    check_mass_dimension, diagonalize_takagi, extract_majorana_vertices,
    majorana_feynman_rule, majorana_mass_matrix, majorana_symmetry_sign,
    numeric_equal, suggest_yukawa,
)
from feynlag.dirac import _dirac_rep


# --------------------------------------------------------------------------
# §1  charge conjugation
# --------------------------------------------------------------------------

def test_C_matrix_identities():
    """C = iγ²γ⁰ satisfies Cᵀ = C† = C⁻¹ = −C and the transpose rules."""
    rep = _dirac_rep()
    C, PL, PR = rep["C"], rep["PL"], rep["PR"]
    Z = sp.zeros(4)
    assert sp.simplify(C.T + C) == Z
    assert sp.simplify(C.conjugate().T + C) == Z
    assert sp.simplify(C.inv() + C) == Z
    for m in range(4):
        assert sp.simplify(C * rep[("g", m)].T * C.inv() + rep[("g", m)]) == Z
    assert sp.simplify(C * PL.T * C.inv() - PL) == Z
    assert sp.simplify(C * PR.T * C.inv() - PR) == Z


def test_majorana_symmetry_sign_CPL():
    """C P_L / C P_R are antisymmetric matrices ⟹ symmetric Majorana bilinear
    (sign +1) ⟹ the mass matrix comes out symmetric."""
    assert majorana_symmetry_sign(diracC * diracPL) == 1
    assert majorana_symmetry_sign(diracC * diracPR) == 1
    with pytest.raises(NotImplementedError):        # a current Cγ^μP is out of scope
        majorana_symmetry_sign(diracC * DiracGamma(sp.Symbol("mu")) * diracPL)


# --------------------------------------------------------------------------
# §2  MajoranaBilinear atom
# --------------------------------------------------------------------------

def test_majorana_bilinear_symmetry():
    """ψ_iᵀ C P_L ψ_j = ψ_jᵀ C P_L ψ_i (canonicalised at construction)."""
    nu = sp.IndexedBase("nu")
    i, j = sp.symbols("i j", integer=True)
    CPL = diracC * diracPL
    assert MajoranaBilinear(nu[i], CPL, nu[j]) == MajoranaBilinear(nu[j], CPL, nu[i])


def test_majorana_bilinear_conjugate():
    """(ψᵀ C P_L ψ)† = ψ̄ C P_R ψ̄ (bar legs, projector flips, C invariant)."""
    _, _, Ll, _, _ = _sm_leptons()
    nuL = Ll.components[0]
    nuLbar = Ll.bar_components[0]
    i, j = sp.symbols("i j", integer=True)
    conj = sp.conjugate(MajoranaBilinear(nuL[i], diracC * diracPL, nuL[j]))
    assert conj == MajoranaBilinear(nuLbar[i], diracC * diracPR, nuLbar[j])


# --------------------------------------------------------------------------
# shared SM-lepton + Weinberg fixture
# --------------------------------------------------------------------------

def _sm_leptons(nflavors=1):
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.35, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=nflavors,
                     component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=nflavors,
                     component_names=["eR"])
    return gw, g1, Ll, eR, (SU2L, U1Y)


def _weinberg_operator(Ll, H, kappa_over_lambda, indices):
    """O = κ/Λ Σ_{a,c} (εH)_a (εH)_c MB(L^a[i], C P_L, L^c[j])."""
    i, j = indices
    Gp, H0 = H.components
    epsH = [H0, -Gp]
    comp = Ll.components
    CPL = diracC * diracPL
    O = sp.S.Zero
    for a in range(2):
        for c in range(2):
            O += kappa_over_lambda * epsH[a] * epsH[c] * \
                MajoranaBilinear(comp[a][i], CPL, comp[c][j])
    return sp.expand(O)


@pytest.fixture
def weinberg():
    gw, g1, Ll, eR, (SU2L, U1Y) = _sm_leptons(nflavors=1)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    Lam = ExternalParameter("Lam", 1e14, positive=True, unit_dim=1)
    kap = ExternalParameter("kap", 0.5)
    lam = ExternalParameter("lam", 0.13)
    mu2 = InternalParameter("mu2", unit_dim=2)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    i, j = sp.symbols("i j", integer=True)
    O = _weinberg_operator(Ll, H, kap.s / Lam.s, (i, j))
    HdH = (dag(H) * H.mat)[0]
    V = -mu2.s * HdH + lam.s * HdH**2
    L = Lagrangian()
    L.add((dag(Dmu(H)) * Dmu(H))[0], sector="kinetic")
    L.add(-V, sector="potential")
    L.add(O, sector="other")
    model = Model("SM-Weinberg", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, Lam, kap, lam, mu2], lagrangian=L)
    model.solve_tadpoles([mu2])
    return dict(model=model, O=O, Ll=Ll, H=H, v=v, Lam=Lam, kap=kap,
                SU2L=SU2L, U1Y=U1Y, i=i, j=j)


# --------------------------------------------------------------------------
# gauge invariance + mass dimension
# --------------------------------------------------------------------------

def test_weinberg_gauge_invariant(weinberg):
    """LᵀCεL HH is invariant under SU(2)_L and U(1)_Y (the ε contraction)."""
    O, Ll, H = weinberg["O"], weinberg["Ll"], weinberg["H"]
    for group in (weinberg["SU2L"], weinberg["U1Y"]):
        ok, viol = check_gauge_invariance(O, [Ll, H], group)
        assert ok, (group.name, viol)


def test_weinberg_is_dimension_five(weinberg):
    """The *operator* is dimension 5 (dimensionless-coefficient convention):
    rejected at max_dim=4, admitted at 5."""
    O, Ll, H = weinberg["O"], weinberg["Ll"], weinberg["H"]
    fields = [Ll, H]
    ok4, worst = check_mass_dimension(O, fields, max_dim=4)
    assert worst == 5 and not ok4
    ok5, _ = check_mass_dimension(O, fields, max_dim=5)
    assert ok5
    # Model.check_invariance admits it at max_dim=5.  (With Λ carrying dim +1,
    # the 1/Λ makes the *term* dim-4, so max_dim=4 also passes — the same
    # operator-vs-term distinction as the Fermi G_F case.)
    assert weinberg["model"].check_invariance(max_dim=5, hermiticity=False).ok


# --------------------------------------------------------------------------
# §3  Majorana mass matrix + Takagi
# --------------------------------------------------------------------------

def test_weinberg_neutrino_mass(weinberg):
    """m_ν = −κ v²/Λ (the seesaw formula)."""
    Ll, model = weinberg["Ll"], weinberg["model"]
    nuL = Ll.components[0]
    M = majorana_mass_matrix(weinberg["O"], nuL, model.vacuum, 1,
                             (weinberg["i"], weinberg["j"]),
                             gamma=diracC * diracPL)
    expected = -weinberg["kap"].s * weinberg["v"].s**2 / weinberg["Lam"].s
    assert sp.simplify(M[0, 0] - expected) == 0


def test_weinberg_mass_matrix_symmetric_and_takagi():
    """A 2-flavour Weinberg mass is symmetric = −c v²/Λ and Takagi-diagonalizes
    to non-negative masses."""
    gw, g1, Ll, eR, (SU2L, U1Y) = _sm_leptons(nflavors=2)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    Lam = ExternalParameter("Lam", 1e14, positive=True, unit_dim=1)
    mu2 = InternalParameter("mu2", unit_dim=2)
    lam = ExternalParameter("lam", 0.13)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    i, j = sp.symbols("i j", integer=True)
    cK = sp.IndexedBase("cK")
    O = _weinberg_operator(Ll, H, cK[i, j] / Lam.s, (i, j))
    HdH = (dag(H) * H.mat)[0]
    L = Lagrangian()
    L.add((dag(Dmu(H)) * Dmu(H))[0], sector="kinetic")
    L.add(-(-mu2.s * HdH + lam.s * HdH**2), sector="potential")
    L.add(O, sector="other")
    model = Model("W2", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, Lam, lam, mu2], lagrangian=L)
    model.solve_tadpoles([mu2])
    M = majorana_mass_matrix(O, Ll.components[0], model.vacuum, 2, (i, j),
                             gamma=diracC * diracPL)
    assert sp.simplify(M - M.T) == sp.zeros(2, 2)
    # m_ν = −v²/Λ · (symmetric part of c)
    for a in range(2):
        for b in range(2):
            expected = -v.s**2 / (2 * Lam.s) * (cK[a, b] + cK[b, a])
            assert sp.simplify(M[a, b] - expected) == 0
    # Takagi on a symmetric numeric instance
    Mnum = sp.Matrix([[sp.Rational(-1, 10), sp.Rational(-1, 20)],
                      [sp.Rational(-1, 20), sp.Rational(-1, 5)]])
    U, D = diagonalize_takagi(Mnum)
    assert all(D[k, k] >= 0 for k in range(2))
    assert sp.simplify(U * D * U.T - Mnum) == sp.zeros(2, 2)
    assert sp.simplify(U * U.conjugate().T - sp.eye(2)) == sp.zeros(2, 2)


def test_seesaw_majorana_mass_reuse():
    """The machinery also handles a bare type-I seesaw ½ M_R ν_Rᵀ C ν_R."""
    SU2L = SU2("SU2L", coupling=ExternalParameter("gw", 0.65, positive=True))
    U1Y = U1("U1Y", coupling=ExternalParameter("g1", 0.35, positive=True))
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=2,
                      component_names=["nuR"])
    i, j = sp.symbols("i j", integer=True)
    MR = sp.IndexedBase("MR")
    nu = nuR.components[0]
    CPL = diracC * diracPL
    # L ⊃ −½ M_R,ij ν_Rᵀ C ν_R  (symmetric M_R, no scalar dependence)
    L = -sp.Rational(1, 2) * (MR[i, j]) * MajoranaBilinear(nu[i], CPL, nu[j])
    # a spectator VEV'd singlet just so a vacuum exists (the bare mass is
    # scalar-independent, so at_vacuum leaves it untouched)
    vX = ExternalParameter("vX", 1e15, positive=True, unit_dim=1)
    S = Scalar("S", reps={}, component_names=["S0"])
    S.expand_vev({S.components[0]: vX})
    model = Model("seesaw", gauge_groups=[SU2L, U1Y], fields=[nuR, S],
                  parameters=[vX], lagrangian=Lagrangian())
    M = majorana_mass_matrix(L, nu, model.vacuum, 2, (i, j), gamma=CPL)
    assert sp.simplify(M - M.T) == sp.zeros(2, 2)
    # recovers the symmetric part of M_R (a Majorana mass is symmetric)
    for a in range(2):
        for b in range(2):
            assert sp.simplify(M[a, b] - (MR[a, b] + MR[b, a]) / 2) == 0


# --------------------------------------------------------------------------
# §4  Majorana vertices
# --------------------------------------------------------------------------

def test_weinberg_vertices(weinberg):
    """ν̄νh = i κ v/Λ · C P_L,  ν̄νhh = i κ/Λ · C P_L (dual-verified)."""
    model, Ll = weinberg["model"], weinberg["Ll"]
    nuL = Ll.components[0]
    CPL = diracC * diracPL
    i, j = weinberg["i"], weinberg["j"]
    Lphys = sp.expand(model.physical_lagrangian(sector="other"))
    h = sp.Symbol("H0_r", real=True)
    G0 = sp.Symbol("H0_i", real=True)
    tab = extract_majorana_vertices(Lphys, [h, G0])
    key = (nuL[i], CPL, nuL[j])
    c1 = tab[key][1][(h,)]
    c2 = tab[key][2][(h, h)]
    rule_h = majorana_feynman_rule(c1, CPL, (h,))
    rule_hh = majorana_feynman_rule(c2, CPL, (h, h))
    kap, v, Lam = weinberg["kap"].s, weinberg["v"].s, weinberg["Lam"].s
    assert sp.simplify(rule_h - sp.I * kap * v / Lam * CPL) == 0
    assert sp.simplify(rule_hh - sp.I * kap / Lam * CPL) == 0
    # dual numeric check on the scalar coefficients
    syms = [weinberg["kap"].s, weinberg["v"].s, weinberg["Lam"].s]
    ok, _ = numeric_equal(c1, kap * v / Lam, syms)
    assert ok
    ok, _ = numeric_equal(c2, kap / (2 * Lam), syms)
    assert ok


# --------------------------------------------------------------------------
# hermiticity
# --------------------------------------------------------------------------

def test_weinberg_hermiticity(weinberg):
    """O + h.c. is hermitian; O alone is not."""
    O = weinberg["O"]
    ok, res = check_hermiticity(O + sp.conjugate(O))
    assert ok, res
    ok_alone, _ = check_hermiticity(O)
    assert not ok_alone


# --------------------------------------------------------------------------
# §5  suggest.py enumeration
# --------------------------------------------------------------------------

def test_suggest_weinberg_dim5():
    """suggest_yukawa returns the Weinberg operator only at max_dim >= 5."""
    gw, g1, Ll, eR, (SU2L, U1Y) = _sm_leptons(nflavors=1)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    at4 = suggest_yukawa([Ll, eR], [H], [SU2L, U1Y], max_dim=4)
    at5 = suggest_yukawa([Ll, eR], [H], [SU2L, U1Y], max_dim=5)
    labels4 = " ".join(t.label for t in at4)
    labels5 = " ".join(t.label for t in at5)
    assert "Weinberg" not in labels4
    assert "Weinberg" in labels5
    assert len(at5) == len(at4) + 1
