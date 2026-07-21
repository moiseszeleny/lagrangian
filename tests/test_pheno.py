"""Tests for the pheno (decay calculator) submodule."""

import pytest
import sympy as sp

from feynlag.pheno.trace import dirac_trace
from feynlag.pheno.kinematics import kallen, two_body_phase_space
from feynlag.dirac import DiracGamma, PL, PR, minkowski_metric

def test_kallen():
    # lambda(M^2, m1^2, m2^2) for M=10, m1=2, m2=3
    # 100^2 + 4^2 + 9^2 - 2(400) - 2(36) - 2(900)
    # kallen(100, 4, 9) = 10000 + 16 + 81 - 800 - 72 - 1800 = 7425
    assert kallen(100, 4, 9) == 7425

def test_trace_two_gammas():
    mu = sp.Symbol('mu')
    nu = sp.Symbol('nu')
    expr = DiracGamma(mu) * DiracGamma(nu)
    res = dirac_trace(expr)
    assert res == 4 * minkowski_metric(mu, nu)

def test_trace_two_gammas_PL():
    mu = sp.Symbol('mu')
    nu = sp.Symbol('nu')
    expr = DiracGamma(mu) * DiracGamma(nu) * PL()
    res = dirac_trace(expr)
    assert res == 2 * minkowski_metric(mu, nu)

def test_trace_four_gammas():
    mu = sp.Symbol('mu')
    nu = sp.Symbol('nu')
    alpha = sp.Symbol('alpha')
    beta = sp.Symbol('beta')
    expr = DiracGamma(mu) * DiracGamma(nu) * DiracGamma(alpha) * DiracGamma(beta)
    res = dirac_trace(expr)
    
    g = minkowski_metric
    expected = 4 * (g(mu, nu)*g(alpha, beta) - g(mu, alpha)*g(nu, beta) + g(mu, beta)*g(nu, alpha))
    assert res == expected
