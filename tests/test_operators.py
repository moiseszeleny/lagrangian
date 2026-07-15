"""Derivative operator and momentum-space substitution tests."""

import sympy as sp

from feynlag import (
    D_linear, PartialMu, expand_derivatives, momentum, to_momentum_space,
)

phi1, phi2, h = sp.symbols("phi1 phi2 h", real=True)
g = sp.Symbol("g", real=True)
FIELDS = [phi1, phi2, h]


def test_partialmu_doit_momentum_rule():
    """Pinned convention: PartialMu(phi).doit() == I * p(phi) * phi."""
    assert PartialMu(phi1).doit() == sp.I * momentum(phi1) * phi1


def test_leibniz_two_fields():
    result = D_linear(phi1 * phi2, FIELDS)
    expected = PartialMu(phi1) * phi2 + phi1 * PartialMu(phi2)
    assert sp.expand(result - expected) == 0


def test_leibniz_with_coefficient():
    result = D_linear(g * phi1 * phi2, FIELDS)
    expected = g * (PartialMu(phi1) * phi2 + phi1 * PartialMu(phi2))
    assert sp.expand(result - expected) == 0


def test_linearity_over_sums():
    result = D_linear(phi1 + g * phi2, FIELDS)
    expected = PartialMu(phi1) + g * PartialMu(phi2)
    assert sp.expand(result - expected) == 0


def test_single_field_with_power():
    """Leibniz on phi1**2 (expanded to phi1*phi1)."""
    result = D_linear(phi1**2, FIELDS)
    # after expand: phi1*phi1 -> 2 phi1 PartialMu(phi1)? The Leibniz path sees
    # Pow, which expand() keeps as Pow — verify via momentum space instead.
    ms = result.replace(PartialMu, lambda arg: sp.I * momentum(arg) * arg)
    # d(phi1^2) = 2 phi1 dphi1 -> 2 I p(phi1) phi1^2
    # (whatever internal form, momentum space must agree)
    assert sp.expand(ms - 2 * sp.I * momentum(phi1) * phi1**2) == 0


def test_expand_derivatives_pipeline():
    expr = PartialMu(phi1 * phi2)
    result = expand_derivatives(expr, FIELDS)
    expected = PartialMu(phi1) * phi2 + phi1 * PartialMu(phi2)
    assert sp.expand(result - expected) == 0


def test_to_momentum_space():
    expr = PartialMu(phi1) * PartialMu(phi2)
    result = to_momentum_space(expr, FIELDS)
    expected = (sp.I * momentum(phi1) * phi1) * (sp.I * momentum(phi2) * phi2)
    assert sp.expand(result - expected) == 0


def test_kinetic_term_momentum_space():
    """(d phi)^2 -> -p(phi)^2 phi^2."""
    L_kin = PartialMu(phi1) * PartialMu(phi1)
    result = to_momentum_space(L_kin, FIELDS)
    assert sp.expand(result + momentum(phi1) ** 2 * phi1**2) == 0
