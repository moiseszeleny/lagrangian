"""Invariance-check tests: SM doublet, wrong hypercharge, Z2, S3 3HDM."""

import sympy as sp
import pytest

from feynlag import (
    Bilinear, Dmu, ExternalParameter, Lagrangian, Model, S3, Scalar, SU2,
    U1, WeylFermion, ZN, check_discrete_invariance, check_gauge_invariance,
    check_hermiticity, check_mass_dimension, dag, diracPL, diracPR,
    fermion_gauge_current,
)


@pytest.fixture
def ew():
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    return SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)


def sm_higgs(ew, y=sp.Rational(1, 2), name="H"):
    SU2L, U1Y = ew
    return Scalar(name, reps={SU2L: 2, U1Y: y},
                  component_names=[f"{name}p", f"{name}0"])


class TestGaugeInvariance:
    def test_sm_potential_invariant(self, ew):
        SU2L, U1Y = ew
        H = sm_higgs(ew)
        mu2, lam = sp.symbols("mu2 lam", real=True)
        HdH = (dag(H) * H.mat)[0]
        V = -mu2 * HdH + lam * HdH**2

        for group in (SU2L, U1Y):
            ok, violations = check_gauge_invariance(V, [H], group)
            assert ok, f"{group.name}: {violations}"

    def test_wrong_hypercharge_fails(self, ew):
        """H1†H2 with different hypercharges is not U(1) invariant."""
        SU2L, U1Y = ew
        H1 = sm_higgs(ew, y=sp.Rational(1, 2), name="H1")
        H2 = sm_higgs(ew, y=sp.Rational(3, 2), name="H2")
        term = (dag(H1) * H2.mat)[0]
        ok, violations = check_gauge_invariance(term, [H1, H2], U1Y)
        assert not ok

    def test_non_invariant_su2_term_fails(self, ew):
        """A single doublet component squared breaks SU(2)."""
        SU2L, U1Y = ew
        H = sm_higgs(ew)
        term = H.components[0] * sp.conjugate(H.components[0])  # |H+|^2 alone
        ok, violations = check_gauge_invariance(term, [H], SU2L)
        assert not ok

    def test_two_doublet_mixed_term_invariant(self, ew):
        """(H1†H2)(H2†H1) is invariant for equal hypercharges."""
        SU2L, U1Y = ew
        H1 = sm_higgs(ew, name="H1")
        H2 = sm_higgs(ew, name="H2")
        H1dH2 = (dag(H1) * H2.mat)[0]
        term = H1dH2 * sp.conjugate(H1dH2)
        for group in (SU2L, U1Y):
            ok, violations = check_gauge_invariance(term, [H1, H2], group)
            assert ok, f"{group.name}: {violations}"


class TestDiscreteInvariance:
    def test_z2_forbids_odd_term(self, ew):
        Z2 = ZN("Z2", 2)
        H1 = sm_higgs(ew, name="H1")
        H2 = sm_higgs(ew, name="H2")
        Z2.assign(0, H1)
        Z2.assign(1, H2)

        odd = (dag(H1) * H2.mat)[0]                 # H1†H2: odd, forbidden
        even = odd * sp.conjugate(odd)              # |H1†H2|^2: even, allowed

        ok_odd, _ = check_discrete_invariance(odd, Z2)
        ok_even, _ = check_discrete_invariance(even, Z2)
        assert not ok_odd
        assert ok_even

    def test_s3_invariant_terms_pass(self, ew):
        """Representative S3-invariant 3HDM terms (doublet (H1,H2) + singlet HS)."""
        s3 = S3()
        H1 = sm_higgs(ew, name="H1")
        H2 = sm_higgs(ew, name="H2")
        HS = sm_higgs(ew, name="HS")
        s3.assign("2", H1, H2)
        s3.assign("1", HS)

        H1dH1 = (dag(H1) * H1.mat)[0]
        H2dH2 = (dag(H2) * H2.mat)[0]
        HSdHS = (dag(HS) * HS.mat)[0]

        # mu^2 (H1†H1 + H2†H2): the S3-invariant quadratic
        inv_quadratic = H1dH1 + H2dH2
        # singlet self-coupling
        inv_singlet = HSdHS**2

        for term in (inv_quadratic, inv_singlet):
            ok, violations = check_discrete_invariance(term, s3)
            assert ok, violations

    def test_s3_forbidden_term_fails(self, ew):
        s3 = S3()
        H1 = sm_higgs(ew, name="H1")
        H2 = sm_higgs(ew, name="H2")
        HS = sm_higgs(ew, name="HS")
        s3.assign("2", H1, H2)
        s3.assign("1", HS)

        # HS†H1 alone picks the first member of the doublet: forbidden
        term = (dag(HS) * H1.mat)[0]
        ok, _ = check_discrete_invariance(term, s3)
        assert not ok

    def test_s3_cubic_invariant(self, ew):
        """HS†(H1†H1 + H2†H2)HS is trivially invariant; check a CG-built one:
        the '1' contraction of the (2 x 2) product with real doublet fields."""
        s3 = S3()
        x1, x2 = sp.symbols("s3x1 s3x2", real=True)
        s3.assign("2", x1, x2)
        invariant = x1**2 + x2**2
        cubic_invariant = x1**3 - 3 * x1 * x2**2  # Re[(x1 + i x2)^3]
        for term in (invariant, cubic_invariant):
            ok, violations = check_discrete_invariance(term, s3)
            assert ok, violations
        # x1^3 alone is not invariant
        ok, _ = check_discrete_invariance(x1**3, s3)
        assert not ok


class TestHermiticityAndDimension:
    def test_hermitian_potential(self, ew):
        H = sm_higgs(ew)
        lam = sp.Symbol("lam", real=True)
        HdH = (dag(H) * H.mat)[0]
        ok, _ = check_hermiticity(lam * HdH**2)
        assert ok

    def test_non_hermitian_fails(self, ew):
        H1 = sm_higgs(ew, name="H1")
        H2 = sm_higgs(ew, name="H2")
        term = (dag(H1) * H2.mat)[0]  # needs + h.c.
        ok, _ = check_hermiticity(term)
        assert not ok

    def test_mass_dimension(self, ew):
        H = sm_higgs(ew)
        lam = sp.Symbol("lam", real=True)
        HdH = (dag(H) * H.mat)[0]
        ok, worst = check_mass_dimension(lam * HdH**2, [H])
        assert ok and worst == 4
        ok, worst = check_mass_dimension(lam * HdH**3, [H])
        assert not ok and worst == 6


class TestModelReport:
    def test_model_check_invariance(self, ew):
        SU2L, U1Y = ew
        H = sm_higgs(ew)
        mu2, lam = sp.symbols("mu2 lam", real=True)
        HdH = (dag(H) * H.mat)[0]

        L = Lagrangian()
        L.add(mu2 * HdH - lam * HdH**2, sector="potential")

        m = Model("SMHiggs", gauge_groups=[SU2L, U1Y], fields=[H],
                  lagrangian=L)
        report = m.check_invariance()
        assert report.ok, report.failures

    def test_model_report_failure_and_raise(self, ew):
        SU2L, U1Y = ew
        H1 = sm_higgs(ew, name="H1")
        H2 = sm_higgs(ew, y=sp.Rational(3, 2), name="H2")
        L = Lagrangian()
        L.add((dag(H1) * H2.mat)[0], sector="potential", name="bad-term")

        m = Model("Broken", gauge_groups=[SU2L, U1Y], fields=[H1, H2],
                  lagrangian=L)
        report = m.check_invariance()
        assert not report.ok
        with pytest.raises(ValueError, match="invariance"):
            report.raise_on_failure()

    def test_dmu_kinetic_term_invariant(self, ew):
        """Regression: a Dmu-built (Leibniz-compound PartialMu) kinetic term
        must pass gauge invariance, not just non-derivative potential terms.

        Previously `check_gauge_invariance` differentiated the transformed
        term symbolically w.r.t. the infinitesimal parameter; since `Dmu`
        wraps `PartialMu` around a component that the transform makes
        alpha-dependent, this produced an unevaluated, non-cancelling
        `Subs(Derivative(PartialMu(...), ...))` residue and reported false
        invariance failures on every gauged kinetic term."""
        SU2L, U1Y = ew
        H = sm_higgs(ew)
        DH = Dmu(H)
        L_kin = (dag(DH) * DH)[0]

        L = Lagrangian().add(L_kin, sector="kinetic")
        m = Model("SM-kinetic", gauge_groups=[SU2L, U1Y],
                  fields=[H, SU2L.bosons("W"), U1Y.bosons("B")], lagrangian=L)
        report = m.check_invariance()
        assert report.ok, report.failures


class TestFermionGaugeInvariance:
    """SM lepton Yukawa + gauge-current terms: check_gauge_invariance and
    check_mass_dimension now cover the fermion Bilinear track (previously
    they crashed / silently no-opped on it — see invariance.py's
    _fermion_transform and vertices/bilinear.py's expand_bilinear)."""

    fl_i, fl_j = sp.symbols("fl_i fl_j", integer=True)

    def _leptons(self, ew):
        SU2L, U1Y = ew
        H = sm_higgs(ew)
        Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                         chirality="L", nflavors=3,
                         component_names=["nuL", "eL"])
        eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                         component_names=["eR"])
        W, B = SU2L.bosons(), U1Y.bosons()
        return H, Ll, eR, W, B

    def _yukawa(self, H, Ll, eR):
        i, j = self.fl_i, self.fl_j
        Y = sp.IndexedBase("Ye")
        nuLbar, eLbar = Ll.bar_components
        eRc, eRbar = eR.components[0], eR.bar_components[0]
        Gp, H0 = H.components
        LYuk = -(Y[i, j] * Gp * Bilinear(nuLbar[i], diracPR, eRc[j])
                 + Y[i, j] * H0 * Bilinear(eLbar[i], diracPR, eRc[j]))
        LYuk += -(sp.conjugate(Y[i, j]) * sp.conjugate(Gp)
                  * Bilinear(eRbar[j], diracPL, Ll.components[0][i])
                  + sp.conjugate(Y[i, j]) * sp.conjugate(H0)
                  * Bilinear(eRbar[j], diracPL, Ll.components[1][i]))
        return LYuk

    def test_yukawa_gauge_invariant(self, ew):
        SU2L, U1Y = ew
        H, Ll, eR, W, B = self._leptons(ew)
        LYuk = self._yukawa(H, Ll, eR)
        fields = [H, Ll, eR, W, B]
        assert check_gauge_invariance(LYuk, fields, SU2L)[0]
        assert check_gauge_invariance(LYuk, fields, U1Y)[0]

    def test_yukawa_mass_dimension(self, ew):
        H, Ll, eR, W, B = self._leptons(ew)
        LYuk = self._yukawa(H, Ll, eR)
        ok, dim = check_mass_dimension(LYuk, [H, Ll, eR])
        assert ok and dim == 4

    def test_gauge_current_invariant(self, ew):
        SU2L, U1Y = ew
        H, Ll, eR, W, B = self._leptons(ew)
        current = (fermion_gauge_current(Ll, self.fl_i)
                  + fermion_gauge_current(eR, self.fl_i))
        fields = [Ll, eR, W, B]
        assert check_gauge_invariance(current, fields, SU2L)[0]
        assert check_gauge_invariance(current, fields, U1Y)[0]

    def test_gauge_current_mass_dimension(self, ew):
        H, Ll, eR, W, B = self._leptons(ew)
        current = (fermion_gauge_current(Ll, self.fl_i)
                  + fermion_gauge_current(eR, self.fl_i))
        ok, dim = check_mass_dimension(current, [Ll, eR, W, B])
        assert ok and dim == 4

    def test_bare_bilinear_without_higgs_fails(self, ew):
        """A bilinear connecting the SU2 doublet directly to the SU2
        singlet, with no scalar to soak up the representation mismatch, is
        not gauge invariant under either group (proves the check
        discriminates, not just trivially passes)."""
        SU2L, U1Y = ew
        H, Ll, eR, W, B = self._leptons(ew)
        i, j = self.fl_i, self.fl_j
        nuLbar = Ll.bar_components[0]
        bad = Bilinear(nuLbar[i], diracPR, eR.components[0][j])
        fields = [Ll, eR, W, B]
        assert not check_gauge_invariance(bad, fields, SU2L)[0]
        assert not check_gauge_invariance(bad, fields, U1Y)[0]

    def test_overdimensioned_fermion_operator_fails(self, ew):
        H, Ll, eR, W, B = self._leptons(ew)
        i, j = self.fl_i, self.fl_j
        HdH = (dag(H) * H.mat)[0]
        eLbar, eRc = Ll.bar_components[1], eR.components[0]
        dim5 = HdH * Bilinear(eLbar[i], diracPR, eRc[j])   # 2 + 3 = 5 > 4
        ok, dim = check_mass_dimension(dim5, [H, Ll, eR])
        assert not ok and dim == 5

    def test_model_check_invariance_end_to_end(self, ew):
        SU2L, U1Y = ew
        H, Ll, eR, W, B = self._leptons(ew)
        LYuk = self._yukawa(H, Ll, eR)
        current = (fermion_gauge_current(Ll, self.fl_i)
                  + fermion_gauge_current(eR, self.fl_i))

        L = Lagrangian()
        L.add(LYuk, sector="yukawa")
        L.add(current, sector="yukawa")

        m = Model("SM-leptons", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, W, B], lagrangian=L)
        report = m.check_invariance()
        assert report.ok, report.failures

    def test_fermion_bilinear_conjugate(self, ew):
        """(psibar1 Gamma psi2)^dagger = psibar2 Gammabar psi1, with flavor
        indices carried over unchanged."""
        H, Ll, eR, W, B = self._leptons(ew)
        i, j = self.fl_i, self.fl_j
        nuLbar = Ll.bar_components[0]
        eRc, eRbar = eR.components[0], eR.bar_components[0]

        bil = Bilinear(nuLbar[i], diracPR, eRc[j])
        expected = Bilinear(eRbar[j], diracPL, Ll.components[0][i])
        assert sp.conjugate(bil) == expected

    def test_gauge_current_hermitian(self, ew):
        """i psibar gamma^mu D_mu psi is hermitian by construction, with no
        hand-written +h.c. needed."""
        SU2L, U1Y = ew
        H, Ll, eR, W, B = self._leptons(ew)
        current = (fermion_gauge_current(Ll, self.fl_i)
                  + fermion_gauge_current(eR, self.fl_i))
        ok, residual = check_hermiticity(current)
        assert ok, residual

    def test_yukawa_with_hc_is_hermitian(self, ew):
        """The full Yukawa Lagrangian (mass term + its hand-written h.c.
        partner) is now genuinely verified hermitian, not merely skipped."""
        H, Ll, eR, W, B = self._leptons(ew)
        SU2L, U1Y = ew
        LYuk = self._yukawa(H, Ll, eR)
        L = Lagrangian().add(LYuk, sector="yukawa")
        m = Model("SM-leptons-herm", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, W, B], lagrangian=L)
        report = m.check_invariance()
        assert report.ok, report.failures

    def test_yukawa_without_hc_fails_hermiticity(self, ew):
        """A Yukawa mass term written WITHOUT its h.c. partner is not
        hermitian — proves the check discriminates, not just passes."""
        H, Ll, eR, W, B = self._leptons(ew)
        SU2L, U1Y = ew
        i, j = self.fl_i, self.fl_j
        Y = sp.IndexedBase("Ye")
        nuLbar, eLbar = Ll.bar_components
        Gp, H0 = H.components
        eRc = eR.components[0]
        LYuk_no_hc = -(Y[i, j] * Gp * Bilinear(nuLbar[i], diracPR, eRc[j])
                      + Y[i, j] * H0 * Bilinear(eLbar[i], diracPR, eRc[j]))

        L = Lagrangian().add(LYuk_no_hc, sector="yukawa")
        m = Model("SM-leptons-no-hc", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, W, B], lagrangian=L)
        report = m.check_invariance()
        assert any(name == "hermiticity" for _, name, _ in report.failures)


class TestFermionDiscreteInvariance:
    """S3/ZN discrete symmetry acting on fermion multiplets — the same
    physics as thdm_s3.py's scalar S3 doublet/singlet structure, but for
    lepton generations (the user's own 3HDM+S3/LFV research domain)."""

    fl_i = sp.Symbol("dfl_i", integer=True)

    def _s3_leptons(self):
        """Two WeylFermion 'generations' forming an S3 doublet, a third
        forming the S3 singlet — mirroring thdm_s3.py's (H1,H2)+HS pattern."""
        s3 = S3()
        psi1 = WeylFermion("dpsi1", reps={}, chirality="L", nflavors=1,
                           component_names=["dp1"])
        psi2 = WeylFermion("dpsi2", reps={}, chirality="L", nflavors=1,
                           component_names=["dp2"])
        psiS = WeylFermion("dpsiS", reps={}, chirality="L", nflavors=1,
                           component_names=["dpS"])
        s3.assign("2", psi1, psi2)
        s3.assign("1", psiS)
        return s3, psi1, psi2, psiS

    def test_doublet_singlet_contraction_invariant(self):
        """The S3 '1' contraction of the doublet with itself,
        Sum_i psibar_i Gamma psi_i, is S3-invariant (mirrors
        S3.doublet_product's known-invariant '1' structure for scalars)."""
        s3, psi1, psi2, psiS = self._s3_leptons()
        i = self.fl_i
        p1bar, = psi1.bar_components
        p2bar, = psi2.bar_components
        p1c, = psi1.components
        p2c, = psi2.components

        invariant = (Bilinear(p1bar[i], diracPR, p1c[i])
                    + Bilinear(p2bar[i], diracPR, p2c[i]))
        ok, violations = check_discrete_invariance(invariant, s3)
        assert ok, violations

    def test_singlet_alone_invariant(self):
        s3, psi1, psi2, psiS = self._s3_leptons()
        i = self.fl_i
        pSbar, = psiS.bar_components
        pSc, = psiS.components
        term = Bilinear(pSbar[i], diracPR, pSc[i])
        ok, violations = check_discrete_invariance(term, s3)
        assert ok, violations

    def test_single_doublet_member_alone_fails(self):
        """Picking out only psi1 (not the full doublet contraction) breaks
        S3 — proves the check discriminates, not just trivially passes."""
        s3, psi1, psi2, psiS = self._s3_leptons()
        i = self.fl_i
        p1bar, = psi1.bar_components
        p1c, = psi1.components
        term = Bilinear(p1bar[i], diracPR, p1c[i])
        ok, violations = check_discrete_invariance(term, s3)
        assert not ok

    def test_cross_multiplet_bilinear_fails(self):
        """A bilinear connecting the doublet to the singlet (with no CG
        contraction structure) is not S3-invariant."""
        s3, psi1, psi2, psiS = self._s3_leptons()
        i = self.fl_i
        p1bar, = psi1.bar_components
        pSc, = psiS.components
        term = Bilinear(p1bar[i], diracPR, pSc[i])
        ok, violations = check_discrete_invariance(term, s3)
        assert not ok

    def test_zn_matching_charges_invariant(self):
        """ZN case: net charge 0 (bar charge -1, field charge +1) is
        invariant — exercises the complex-phase X=conjugate(M) branch."""
        Z3 = ZN("Z3", 3)
        psi = WeylFermion("dzpsi", reps={}, chirality="L", nflavors=1,
                          component_names=["dzp"])
        Z3.assign(1, psi)
        i = self.fl_i
        pbar, = psi.bar_components
        pc, = psi.components
        term = Bilinear(pbar[i], diracPR, pc[i])
        ok, violations = check_discrete_invariance(term, Z3)
        assert ok, violations

    def test_zn_mismatched_charges_fails(self):
        Z3 = ZN("Z3", 3)
        psiA = WeylFermion("dzpsiA", reps={}, chirality="L", nflavors=1,
                           component_names=["dzpA"])
        psiB = WeylFermion("dzpsiB", reps={}, chirality="L", nflavors=1,
                           component_names=["dzpB"])
        Z3.assign(1, psiA)
        Z3.assign(2, psiB)
        i = self.fl_i
        pAbar, = psiA.bar_components
        pBc, = psiB.components
        term = Bilinear(pAbar[i], diracPR, pBc[i])   # charge -1 + 2 = 1 != 0
        ok, violations = check_discrete_invariance(term, Z3)
        assert not ok
