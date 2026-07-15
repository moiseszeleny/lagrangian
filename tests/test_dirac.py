"""Dirac algebra tests, ported from the DLRSM1 dirac.py __main__ cases."""

import pytest
import sympy as sp

from feynlag import (
    DiracGamma, dirac0, diracI, diracPL, diracPR, dirac_conjugate,
    gamma_simplify, minkowski_metric,
)

mu, nu = sp.symbols("mu nu")


class TestMinkowskiMetric:
    def test_concrete_diagonal(self):
        assert minkowski_metric(0, 0) == 1
        assert minkowski_metric(1, 1) == -1
        assert minkowski_metric(2, 2) == -1
        assert minkowski_metric(3, 3) == -1

    def test_concrete_off_diagonal(self):
        assert minkowski_metric(0, 1) == 0
        assert minkowski_metric(2, 3) == 0

    def test_symbolic_diagonal(self):
        g_mumu = minkowski_metric(mu, mu)
        assert g_mumu.subs(mu, 0) == 1
        assert g_mumu.subs(mu, 1) == -1

    def test_other_convention(self):
        assert minkowski_metric(0, 0, convention="-+++") == -1
        assert minkowski_metric(1, 1, convention="-+++") == 1


class TestCliffordAlgebra:
    # gamma_simplify normalizes I4 -> 1 (unit of the Dirac-space algebra)

    def test_gamma0_squared(self):
        assert gamma_simplify(DiracGamma(0) * DiracGamma(0)) == 1

    def test_gamma1_squared(self):
        assert gamma_simplify(DiracGamma(1) * DiracGamma(1)) == -1

    def test_anticommutator_symbolic(self):
        g_mu, g_nu = DiracGamma(mu), DiracGamma(nu)
        anti = g_mu * g_nu + g_nu * g_mu
        result = gamma_simplify(anti)
        expected = 2 * minkowski_metric(mu, nu)
        # result and expected may differ by KroneckerDelta identities that
        # SymPy does not contract symbolically (delta(0,mu)*delta(mu,nu) ==
        # delta(0,nu)*delta(mu,nu)); compare at every concrete index pair.
        for m in range(4):
            for n in range(4):
                subs = {mu: m, nu: n}
                assert sp.expand(result.subs(subs) - expected.subs(subs)) == 0, \
                    f"mismatch at (mu, nu) = ({m}, {n})"

    def test_anticommutator_concrete_off_diagonal(self):
        g0, g1 = DiracGamma(0), DiracGamma(1)
        result = gamma_simplify(g0 * g1 + g1 * g0)
        assert result == 0 or result == dirac0

    def test_identity_absorption(self):
        g = DiracGamma(mu)
        assert gamma_simplify(g * diracI) == g
        assert gamma_simplify(diracI * g) == g

    def test_zero_absorption(self):
        g = DiracGamma(mu)
        assert dirac0 * g == dirac0


class TestChiralProjectors:
    def test_idempotent(self):
        assert diracPL * diracPL == diracPL
        assert diracPR * diracPR == diracPR

    def test_orthogonal(self):
        assert diracPL * diracPR == dirac0
        assert diracPR * diracPL == dirac0

    def test_completeness(self):
        assert diracPL + diracPR == diracI
        assert diracPR + diracPL == diracI

    def test_identity_action(self):
        assert diracI * diracPL == diracPL
        assert diracPR * diracI == diracPR

    def test_zero_action(self):
        assert dirac0 * diracPL == dirac0
        assert diracPR * dirac0 == dirac0

    def test_scalar_coefficients_survive(self):
        expr = 2 * diracPL + 3 * diracPL
        assert expr == 5 * diracPL


class TestDiracConjugate:
    """Gammabar = gamma^0 Gamma^dagger gamma^0, for the Bilinear middle slot."""

    def test_identity_self_conjugate(self):
        assert dirac_conjugate(diracI) == diracI
        assert dirac_conjugate(sp.S.One) == diracI

    def test_bare_projectors_swap(self):
        assert dirac_conjugate(diracPL) == diracPR
        assert dirac_conjugate(diracPR) == diracPL

    def test_bare_gamma_self_conjugate(self):
        assert dirac_conjugate(DiracGamma(mu)) == DiracGamma(mu)

    def test_vector_current_self_conjugate(self):
        """gamma^mu P_L = P_R gamma^mu, so the two conjugation effects
        cancel: a V-A/V+A current conjugates to itself."""
        assert dirac_conjugate(DiracGamma(mu) * diracPL) == DiracGamma(mu) * diracPL
        assert dirac_conjugate(DiracGamma(mu) * diracPR) == DiracGamma(mu) * diracPR

    def test_double_conjugate_is_identity(self):
        for gamma in (diracI, diracPL, diracPR, DiracGamma(mu),
                     DiracGamma(mu) * diracPL, DiracGamma(mu) * diracPR):
            assert dirac_conjugate(dirac_conjugate(gamma)) == gamma

    def test_unsupported_structure_raises(self):
        with pytest.raises(NotImplementedError):
            dirac_conjugate(DiracGamma(mu) * DiracGamma(nu))
