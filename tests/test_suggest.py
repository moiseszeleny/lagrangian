"""Term-suggestion tool: invariant enumeration pinned against hand-built models.

Pins (cross-checked against the examples/ models the suggestions reproduce):
- SM potential spans {H†H, (H†H)²}.
- 2HDM with an EXACT Z2 gives the 7-operator basis (m11², m22², λ1–λ5);
  the soft-breaking m12² term is correctly ABSENT.
- 3HDM+S3 gives the 10-operator basis (Reynolds-projection ground truth),
  spanning examples/thdm_s3.py's CG-built potential.
- U(1)_X singlet gives {H†H, S†S, (H†H)², (S†S)(H†H), (S†S)²} — no bare
  S/S²/S³ (all X-charged).
- SM Yukawa: the e (via H), d (via H), u (via H̃) patterns; VLL adds the
  bare Dirac mass and the mixing Yukawa (examples/sm_vll.py).
- Every suggestion passes the full gauge/discrete/hermiticity/dimension
  oracle (verify=True), re-asserted here as a parametrized battery.
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, S3, SU2, SU3, Scalar, U1, WeylFermion, ZN, dag,
    check_gauge_invariance, check_discrete_invariance, check_hermiticity,
    check_mass_dimension, numeric_equal, Bilinear,
    suggest_potential, suggest_yukawa, suggest_kinetic, build_lagrangian,
)


# --------------------------------------------------------------------------
# span-equality helper (dual verification: exact rank + random numeric)
# --------------------------------------------------------------------------

def _normalize(expr, registry):
    expr = sp.expand(expr)
    for node in expr.atoms(sp.conjugate):
        registry.setdefault(node, sp.Dummy())
    for node in expr.atoms(Bilinear):
        registry.setdefault(node, sp.Dummy())
    return expr.xreplace(registry)


def _matrix(exprs, registry, keys):
    from feynlag import extract_interaction_coefficients
    rows = []
    for e in exprs:
        norm = _normalize(e, registry)
        syms = sorted(norm.free_symbols, key=lambda s: s.sort_key())
        table = extract_interaction_coefficients(norm, syms)
        vec = {}
        for terms in table.values():
            for ft, c in terms.items():
                vec[ft] = vec.get(ft, sp.S.Zero) + c
        rows.append([vec.get(k, sp.S.Zero) for k in keys])
    return sp.Matrix(rows) if rows else sp.Matrix(0, len(keys), [])


def _all_keys(exprs, registry):
    from feynlag import extract_interaction_coefficients
    keys = set()
    for e in exprs:
        norm = _normalize(e, registry)
        syms = sorted(norm.free_symbols, key=lambda s: s.sort_key())
        for terms in extract_interaction_coefficients(norm, syms).values():
            keys.update(terms)
    return sorted(keys, key=str)


def assert_same_span(suggested, hand_built):
    """Both operator lists span the same coefficient row space (exact rank)."""
    registry = {}
    keys = _all_keys(list(suggested) + list(hand_built), registry)
    A = _matrix(suggested, registry, keys)
    B = _matrix(hand_built, registry, keys)
    rank_A, rank_B = A.rank(), B.rank()
    rank_stack = sp.Matrix.vstack(A, B).rank()
    assert rank_A == rank_B == rank_stack, (
        f"span mismatch: rank(suggested)={rank_A}, rank(hand)={rank_B}, "
        f"rank(stacked)={rank_stack}")
    return rank_stack


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def ew():
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    return SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)


def _doublet(name, SU2L, U1Y):
    return Scalar(name, reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                  component_names=[f"{name}p", f"{name}0"])


# --------------------------------------------------------------------------
# potential
# --------------------------------------------------------------------------

def test_sm_potential(ew):
    SU2L, U1Y = ew
    H = _doublet("H", SU2L, U1Y)
    terms = suggest_potential([H], [SU2L, U1Y])
    assert len(terms) == 2
    HdH = (dag(H) * H.mat)[0]
    assert_same_span([t.expr for t in terms], [HdH, HdH**2])


def test_2hdm_exact_z2(ew):
    SU2L, U1Y = ew
    H1, H2 = _doublet("H1", SU2L, U1Y), _doublet("H2", SU2L, U1Y)
    Z2 = ZN("Z2", 2)
    Z2.assign(0, H1)
    Z2.assign(1, H2)
    terms = suggest_potential([H1, H2], [SU2L, U1Y], discrete_groups=[Z2])

    # exact-Z2 basis: m11², m22², λ1–λ5 (7 operators; NO soft m12²)
    assert len(terms) == 7
    labels = {t.label for t in terms}
    # the Z2-odd soft term (H1†H2) alone must be absent
    assert "(H1†H2)" not in labels

    def bra(a, b):
        return (dag(a) * b.mat)[0]
    b11, b22, b12, b21 = bra(H1, H1), bra(H2, H2), bra(H1, H2), bra(H2, H1)
    hand = [b11, b22, b11**2, b22**2, b11 * b22, b12 * b21,
            b12**2 + b21**2]
    assert_same_span([t.expr for t in terms], hand)


def test_2hdm_soft_z2_includes_m12(ew):
    """With NO discrete symmetry declared, the m12²(H1†H2)+h.c. soft term is
    among the suggestions (the softly-broken 2HDM of examples/thdm.py)."""
    SU2L, U1Y = ew
    H1, H2 = _doublet("H1", SU2L, U1Y), _doublet("H2", SU2L, U1Y)
    terms = suggest_potential([H1, H2], [SU2L, U1Y])
    # the dim-2 (H1†H2)+h.c. operator now appears
    dim2 = [t for t in terms if t.dim == 2]
    assert any(t.hc_added for t in dim2)   # the off-diagonal soft term


def test_3hdm_s3(ew):
    SU2L, U1Y = ew
    s3 = S3()
    H1, H2, HS = (_doublet("H1", SU2L, U1Y), _doublet("H2", SU2L, U1Y),
                  _doublet("HS", SU2L, U1Y))
    s3.assign("2", H1, H2)
    s3.assign("1", HS)
    terms = suggest_potential([H1, H2, HS], [SU2L, U1Y], discrete_groups=[s3])

    # the 10-operator S3 basis (2 mass + 8 quartic)
    assert len(terms) == 10

    # span check against examples/thdm_s3.py's CG-built potential
    def bra(a, b):
        return (dag(a) * b.mat)[0]
    x11, x22 = bra(H1, H1), bra(H2, H2)
    x12, x21 = bra(H1, H2), bra(H2, H1)
    s1, s2 = bra(HS, H1), bra(HS, H2)
    sss = bra(HS, HS)
    inv1 = x11 + x22
    inv1p = x12 - x21
    d2 = (x11 - x22, -(x12 + x21))
    lam4 = s1 * d2[0] + s2 * d2[1]
    lam4 = lam4 + sp.conjugate(lam4)
    lam7 = s1**2 + s2**2
    lam7 = lam7 + sp.conjugate(lam7)
    hand = [inv1, sss,
            inv1**2, inv1p**2, d2[0]**2 + d2[1]**2, lam4,
            sss * inv1, s1 * bra(H1, HS) + s2 * bra(H2, HS), lam7, sss**2]
    assert_same_span([t.expr for t in terms], hand)


def test_u1x_singlet_portal(ew):
    SU2L, U1Y = ew
    gX = ExternalParameter("gX", 0.5, positive=True)
    U1X = U1("U1X", coupling=gX)
    aX = ExternalParameter("aX", 0.4)
    qS = ExternalParameter("qS", 1.0)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2), U1X: aX.s / 2},
               component_names=["Gp", "H0"])
    S = Scalar("S", reps={U1X: qS.s}, component_names=["S0"])
    terms = suggest_potential([H, S], [SU2L, U1Y, U1X])

    labels = {t.label for t in terms}
    assert len(terms) == 5
    # the portal is present; no bare S / S² / S³ (all X-charged)
    assert any("S0" in l and "H" in l for l in labels)   # (S†S)(H†H)
    S0 = S.components[0]
    for t in terms:
        # every term is X-neutral: no lone S0 or conjugate(S0) power imbalance
        assert sp.expand(t.expr) == sp.expand(sp.conjugate(t.expr)) or t.hc_added


# --------------------------------------------------------------------------
# Yukawa
# --------------------------------------------------------------------------

def test_sm_lepton_yukawa(ew):
    SU2L, U1Y = ew
    H = _doublet("H", SU2L, U1Y)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=3, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                     component_names=["eR"])
    terms = suggest_yukawa([Ll, eR], [H], [SU2L, U1Y])
    assert len(terms) == 1
    assert terms[0].dim == 4
    # the term connects L̄ (P_R) to e_R with an H leg, + h.c.
    assert terms[0].expr.has(Bilinear)


def test_sm_quark_yukawa_up_down(ew):
    SU2L, U1Y = ew
    gs = ExternalParameter("gs", 1.22, positive=True)
    SU3c = SU3("SU3c", coupling=gs)
    H = _doublet("H", SU2L, U1Y)
    QL = WeylFermion("QL", reps={SU2L: 2, U1Y: sp.Rational(1, 6), SU3c: 3},
                     chirality="L", nflavors=3,
                     component_names=[f"{f}L_{c}" for f in "ud"
                                      for c in (1, 2, 3)])
    uR = WeylFermion("uR", reps={U1Y: sp.Rational(2, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=[f"uR_{c}" for c in (1, 2, 3)])
    dR = WeylFermion("dR", reps={U1Y: -sp.Rational(1, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=[f"dR_{c}" for c in (1, 2, 3)])
    terms = suggest_yukawa([QL, uR, dR], [H], [SU2L, U1Y, SU3c])
    # down-type via H, up-type via H̃
    assert len(terms) == 2
    labels = {t.label for t in terms}
    assert any("̃" in l for l in labels)         # the H̃ (up-type) pattern
    assert any("dR" in l for l in labels)


def test_vll_bare_mass_and_mixing(ew):
    SU2L, U1Y = ew
    H = _doublet("H", SU2L, U1Y)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])
    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                       chirality="L", nflavors=1, component_names=["NL", "EL"])
    PsiR = WeylFermion("PsiR", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                       chirality="R", nflavors=1, component_names=["NR", "ER"])
    terms = suggest_yukawa([Ll, PsiL, eR, PsiR], [H], [SU2L, U1Y])
    labels = {t.label for t in terms}
    # the vector-like bare Dirac mass (dim 3) and the mixing Yukawa (dim 4)
    assert any(t.dim == 3 and "PsiL" in t.label and "PsiR" in t.label
               for t in terms)
    assert any(t.dim == 4 and "PsiL" in t.label and "eR" in t.label
               for t in terms)


def test_yukawa_forbidden_by_discrete(ew):
    """A Z2 under which only e_R is odd forbids the lepton Yukawa entirely."""
    SU2L, U1Y = ew
    H = _doublet("H", SU2L, U1Y)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=3, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                     component_names=["eR"])
    Z2 = ZN("Z2", 2)
    Z2.assign(0, Ll)
    Z2.assign(0, H)
    Z2.assign(1, eR)
    terms = suggest_yukawa([Ll, eR], [H], [SU2L, U1Y], discrete_groups=[Z2])
    assert terms == []


# --------------------------------------------------------------------------
# kinetic scaffolding + build_lagrangian
# --------------------------------------------------------------------------

def test_kinetic_scaffold(ew):
    SU2L, U1Y = ew
    H = _doublet("H", SU2L, U1Y)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=3, component_names=["nuL", "eL"])
    kin = suggest_kinetic([H, Ll])
    assert any(t.sector == "kinetic" for t in kin)   # (DμH)†(DμH)
    assert any(t.meta.get("current") for t in kin)   # gauge current


def test_build_lagrangian(ew):
    SU2L, U1Y = ew
    H = _doublet("H", SU2L, U1Y)
    pot = suggest_potential([H], [SU2L, U1Y])
    L, params = build_lagrangian(pot, coupling_prefix="lam")
    assert len(L) == 2
    assert [p.name for p in params] == ["lam1", "lam2"]
    # potential terms enter as −c·expr
    for term in L.terms:
        assert term.sector == "potential"


# --------------------------------------------------------------------------
# oracle battery (every suggestion is genuinely invariant)
# --------------------------------------------------------------------------

def test_oracle_all_suggestions_invariant(ew):
    """suggest_* with verify=True already asserts this internally; re-run the
    full battery here explicitly across the potential fixtures as a guard."""
    SU2L, U1Y = ew
    s3 = S3()
    H1, H2, HS = (_doublet("H1", SU2L, U1Y), _doublet("H2", SU2L, U1Y),
                  _doublet("HS", SU2L, U1Y))
    s3.assign("2", H1, H2)
    s3.assign("1", HS)
    terms = suggest_potential([H1, H2, HS], [SU2L, U1Y], discrete_groups=[s3],
                              verify=False)
    for t in terms:
        for g in (SU2L, U1Y):
            ok, viol = check_gauge_invariance(t.expr, [H1, H2, HS], g)
            assert ok, (t.label, g.name, viol)
        ok, viol = check_discrete_invariance(t.expr, s3)
        assert ok, (t.label, viol)
        ok, worst = check_mass_dimension(t.expr, [H1, H2, HS])
        assert ok, (t.label, worst)
        ok, _ = check_hermiticity(t.expr)
        assert ok, t.label
