"""Phase 4 validation: fermion sector (SM leptons, SVD/Takagi, seesaw).

Pinned physics:
- h ℓℓ Yukawa vertex = −i m_ℓ / v   (with Y_ℓ = √2 m_ℓ / v)
- W⁺ ν̄ℓ current    = i g/√2 γ^μ P_L
- Z ℓ̄ℓ current: coefficient ∝ (T³ − Q sin²θ_W) pattern
- photon coupling of the charged lepton = −e (electric charge)
- toy seesaw: Takagi of [[0, m_D], [m_D, M_R]] reproduces the seesaw light
  mass at leading order (cross-check with verify.seesaw_light_mass)
"""

import sympy as sp
import pytest

from feynlag import (
    Bilinear, DiracGamma, ExternalParameter, Rotation, SU2, Scalar, U1,
    Vacuum, WeylFermion, diagonalize_svd, diagonalize_takagi, diracPL,
    diracPR, extract_fermion_vertices, fermion_feynman_rule,
    fermion_gauge_current, fermion_mass_matrix, rotation_2x2,
    seesaw_light_mass,
)

i, j = sp.symbols("fl_i fl_j", integer=True)


@pytest.fixture(scope="module")
def sm_leptons():
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=3,
                     component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                     component_names=["eR"])
    return SU2L, U1Y, H, Ll, eR, v, gw, g1


class TestYukawa:
    def _yukawa(self, sm_leptons):
        """L_Yuk = − Y[i,j] L̄_i H e_R[j] + h.c. in components."""
        SU2L, U1Y, H, Ll, eR, v, gw, g1 = sm_leptons
        Y = sp.IndexedBase("Ye")
        nuLbar, eLbar = Ll.bar_components
        Gp, H0 = H.components
        # ψ̄_L (…) χ_R sandwiches carry P_R
        LYuk = -(Y[i, j] * Gp * Bilinear(nuLbar[i], diracPR, eR.components[0][j])
                 + Y[i, j] * H0 * Bilinear(eLbar[i], diracPR, eR.components[0][j]))
        # + h.c. (χ̄_R (…) ψ_L carries P_L, conjugated coupling/scalars)
        eRbar = eR.bar_components[0]
        LYuk += -(sp.conjugate(Y[i, j]) * sp.conjugate(Gp)
                  * Bilinear(eRbar[j], diracPL, Ll.components[0][i])
                  + sp.conjugate(Y[i, j]) * sp.conjugate(H0)
                  * Bilinear(eRbar[j], diracPL, Ll.components[1][i]))
        return LYuk, Y

    def test_mass_matrix(self, sm_leptons):
        SU2L, U1Y, H, Ll, eR, v, gw, g1 = sm_leptons
        LYuk, Y = self._yukawa(sm_leptons)
        vac = Vacuum([H])
        eLbar = Ll.bar_components[1]
        M = fermion_mass_matrix(LYuk, eLbar, eR.components[0], vac, 3, (i, j),
                                gamma=diracPR)
        # M[a,b] = Y[a,b] v / sqrt(2)
        for a in range(3):
            for b in range(3):
                assert sp.simplify(M[a, b] - Y[a, b] * v.s / sp.sqrt(2)) == 0

    def test_hll_coupling_pinned(self, sm_leptons):
        """With diagonal Y = √2 m_ℓ/v: h ℓℓ vertex = −i m_ℓ/v P_R (+ h.c.)."""
        SU2L, U1Y, H, Ll, eR, v, gw, g1 = sm_leptons
        LYuk, Y = self._yukawa(sm_leptons)
        vac = Vacuum([H])

        # physical basis: expand H0 around the vacuum
        L_shifted = sp.expand(LYuk.xreplace(vac.shift_map))
        h = sp.Symbol("H0_r", real=True)
        table = extract_fermion_vertices(L_shifted, [h])

        eLbar = Ll.bar_components[1]
        key = (eLbar[i], diracPR, eR.components[0][j])
        coeff_h = table[key][1][(h,)]
        assert sp.simplify(coeff_h + Y[i, j] / sp.sqrt(2)) == 0

        # rule with m_ell = Y v/√2: −i m_ℓ/v (γ-slot P_R)
        m_ell = sp.Symbol("m_ell", positive=True)
        rule = fermion_feynman_rule(coeff_h, diracPR, (h,))
        rule_mass = rule.subs(Y[i, j], sp.sqrt(2) * m_ell / v.s)
        assert sp.simplify(rule_mass + sp.I * m_ell / v.s * diracPR) == 0


class TestGaugeCurrents:
    def test_w_lepton_current(self, sm_leptons):
        """W⁺ ν̄_L γ^μ P_L e_L with coupling g/√2 (from T± = (T1 ± iT2))."""
        SU2L, U1Y, H, Ll, eR, v, gw, g1 = sm_leptons
        g = gw.s
        current = fermion_gauge_current(Ll, i, gauge_groups=[SU2L])

        # rotate W1, W2 → W± ;  W3 stays
        W1, W2, W3 = SU2L.bosons().components
        Wp, Wm = sp.symbols("Wp Wm")
        U = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
        sub = Rotation([W1, W2], [Wp, Wm], U, kind="unitary").substitution()
        current = sp.expand(current.xreplace(sub))

        table = extract_fermion_vertices(current, [Wp, Wm, W3])
        nuLbar, eLbar = Ll.bar_components
        nuL, eL = Ll.components
        mu = sp.Symbol("mu", integer=True)
        gamma_L = DiracGamma(mu) * diracPL

        # ν̄ γ P_L e with W+: coupling g/√2
        key = (nuLbar[i], gamma_L, eL[i])
        coeff = table[key][1][(Wp,)]
        rule = fermion_feynman_rule(coeff, gamma_L, (Wp,))
        assert sp.simplify(rule - sp.I * g / sp.sqrt(2) * gamma_L) == 0
        # and the conjugate current with W−
        key = (eLbar[i], gamma_L, nuL[i])
        coeff = table[key][1][(Wm,)]
        assert sp.simplify(coeff - g / sp.sqrt(2)) == 0

    def test_neutral_current_charges(self, sm_leptons):
        """Z ℓ̄ℓ ∝ (T³ − Q s²_W); photon couples with Q e."""
        SU2L, U1Y, H, Ll, eR, v, gw, g1 = sm_leptons
        g, gp = gw.s, g1.s
        current = (fermion_gauge_current(Ll, i)
                   + fermion_gauge_current(eR, i))

        W1, W2, W3 = SU2L.bosons().components
        B = U1Y.bosons().components[0]
        Z, A = sp.symbols("Z A", real=True)
        thetaW = sp.atan(gp / g)
        sub = Rotation([W3, B], [Z, A],
                       rotation_2x2(-thetaW)).substitution()
        current = sp.expand(current.xreplace(sub))

        table = extract_fermion_vertices(current, [Z, A, W1, W2])
        mu = sp.Symbol("mu", integer=True)
        gamma_L = DiracGamma(mu) * diracPL
        gamma_R = DiracGamma(mu) * diracPR

        nuLbar, eLbar = Ll.bar_components
        nuL, eL = Ll.components
        eRbar, eRc = eR.bar_components[0], eR.components[0]

        gz = sp.sqrt(g**2 + gp**2)
        sw2 = gp**2 / (g**2 + gp**2)
        e_em = g * gp / gz

        # photon: charged lepton couples with −e (Q = −1), both chiralities
        assert sp.simplify(table[(eLbar[i], gamma_L, eL[i])][1][(A,)]
                           + e_em) == 0
        assert sp.simplify(table[(eRbar[i], gamma_R, eRc[i])][1][(A,)]
                           + e_em) == 0
        # photon–neutrino: absent
        assert (A,) not in table[(nuLbar[i], gamma_L, nuL[i])].get(1, {})

        # Z couplings: gz (T³ − Q s²_W)
        for key, T3, Q, gamma in (
                ((nuLbar[i], gamma_L, nuL[i]), sp.Rational(1, 2), 0, gamma_L),
                ((eLbar[i], gamma_L, eL[i]), -sp.Rational(1, 2), -1, gamma_L),
                ((eRbar[i], gamma_R, eRc[i]), 0, -1, gamma_R)):
            coeff = table[key][1][(Z,)]
            expected = gz * (T3 - Q * sw2)
            assert sp.simplify(coeff - expected) == 0, (key, coeff, expected)


class TestDiagonalization:
    def test_svd_2x2(self):
        M = sp.Matrix([[3, 1], [1, sp.Rational(3, 2)]])
        eL = sp.symbols("e1L e2L")
        eRs = sp.symbols("e1R e2R")
        eLp = sp.symbols("e1Lp e2Lp")
        eRp = sp.symbols("e1Rp e2Rp")
        rotL, rotR = diagonalize_svd(M, eL, eRs, eLp, eRp)
        D = rotL.matrix * M * rotR.matrix.T
        assert sp.simplify(D[0, 1]) == 0
        assert sp.simplify(D[1, 0]) == 0
        assert D[0, 0] >= 0 and D[1, 1] >= 0
        # singular values match M Mᵀ eigenvalues
        eigs = sorted((M * M.T).eigenvals())
        svals = sorted([D[0, 0] ** 2, D[1, 1] ** 2])
        for s2, e in zip(svals, eigs):
            assert sp.simplify(s2 - e) == 0

    def test_takagi_toy_seesaw(self):
        mD, MR = sp.Rational(1, 10), sp.Integer(100)
        M = sp.Matrix([[0, mD], [mD, MR]])
        U, D = diagonalize_takagi(M)
        # M = U D Uᵀ reconstruction
        assert sp.simplify(U * D * U.T - M) == sp.zeros(2, 2)
        # non-negative masses
        assert D[0, 0] >= 0 and D[1, 1] >= 0
        # light eigenvalue ≈ seesaw formula mD²/MR
        light = min(D[0, 0], D[1, 1])
        seesaw = -seesaw_light_mass(sp.Matrix([[mD]]), sp.Matrix([[MR]]))[0, 0]
        assert abs(float(light - seesaw)) < float(seesaw) * (mD / MR) ** 2 * 10

    def test_takagi_requires_symmetric(self):
        with pytest.raises(ValueError):
            diagonalize_takagi(sp.Matrix([[0, 1], [2, 0]]))
