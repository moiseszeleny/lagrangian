"""Vertex-extractor tests, including the pinned symmetry-factor convention."""

import sympy as sp
import pytest

from feynlag import (
    extract_interaction_coefficients, feynman_rule, vertex_multiplicity,
    round_trip_reconstruct,
)

phi, h, a = sp.symbols("phi h a", real=True)
lam, mu2, g = sp.symbols("lambda mu2 g", real=True)


def test_phi4_coefficient():
    L = -lam / sp.factorial(4) * phi**4
    d = extract_interaction_coefficients(L, [phi])
    assert d[4][(phi, phi, phi, phi)] == -lam / 24


def test_pinned_symmetry_factor_phi4():
    """CONVENTIONS.md pin: L = -lambda/4! phi^4  =>  vertex = -i*lambda."""
    L = -lam / sp.factorial(4) * phi**4
    d = extract_interaction_coefficients(L, [phi])
    coeff = d[4][(phi, phi, phi, phi)]
    assert feynman_rule(coeff, (phi, phi, phi, phi)) == -sp.I * lam


def test_pinned_symmetry_factor_phi3():
    """L = -lambda/3! phi^3  =>  vertex = -i*lambda."""
    L = -lam / sp.factorial(3) * phi**3
    d = extract_interaction_coefficients(L, [phi])
    assert feynman_rule(d[3][(phi, phi, phi)], (phi, phi, phi)) == -sp.I * lam


def test_mixed_field_vertex():
    """L = -g h a^2: vertex = i * (-g) * 2! = -2ig."""
    L = -g * h * a**2
    d = extract_interaction_coefficients(L, [h, a])
    key = tuple(sorted([h, a, a], key=lambda s: s.sort_key()))
    assert feynman_rule(d[3][key], key) == -2 * sp.I * g


def test_vertex_multiplicity():
    assert vertex_multiplicity((phi, phi, phi, phi)) == 24
    assert vertex_multiplicity((h, a, a)) == 2
    assert vertex_multiplicity((h, a)) == 1


def test_quadratic_and_constant_terms():
    L = mu2 * phi**2 + 7
    d = extract_interaction_coefficients(L, [phi])
    assert d[2][(phi, phi)] == mu2
    assert d[0][()] == 7


def test_coefficient_accumulation():
    """Identical monomials written twice must sum."""
    L = g * h * a + 2 * g * a * h
    d = extract_interaction_coefficients(L, [h, a])
    key = tuple(sorted([h, a], key=lambda s: s.sort_key()))
    assert d[2][key] == 3 * g


def test_round_trip_toy_potential():
    """Round-trip reconstruction on a 2HDM-flavored toy potential."""
    h1, h2 = sp.symbols("h1 h2", real=True)
    l1, l2, l3 = sp.symbols("l1 l2 l3", real=True)
    V = (mu2 * (h1**2 + h2**2) + l1 * h1**4 + l2 * h2**4
         + l3 * h1**2 * h2**2 + g * h1**3 * h2)
    ok, residual = round_trip_reconstruct(-V, [h1, h2])
    assert ok, f"round trip failed, residual: {residual}"


def test_fallback_path_agrees_with_poly():
    """Force the fallback and compare against the Poly fast path."""
    from feynlag.vertices.extract import _extract_fallback
    L = -lam / 24 * phi**4 + mu2 * phi**2 - g * h * phi**2
    fast = extract_interaction_coefficients(L, [phi, h])
    slow = _extract_fallback(L, {phi, h})
    assert set(fast.keys()) == set(slow.keys())
    for n in fast:
        assert set(fast[n]) == set(slow[n])
        for k in fast[n]:
            assert sp.simplify(fast[n][k] - slow[n][k]) == 0
