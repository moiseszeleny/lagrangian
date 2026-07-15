"""Verification-utility tests (ported invariants + round trip)."""

import sympy as sp

from feynlag import (
    check_dimension, is_hermitian, is_symmetric, numeric_equal,
    seesaw_light_mass,
)

x, y = sp.symbols("x y", positive=True)
m, v = sp.symbols("m v", positive=True)
Y = sp.Symbol("Y", real=True)


def test_is_hermitian():
    a = sp.Symbol("a", real=True)
    M = sp.Matrix([[1, sp.I * a], [-sp.I * a, 2]])
    assert is_hermitian(M)
    assert not is_hermitian(sp.Matrix([[0, 1], [2, 0]]))


def test_is_symmetric():
    assert is_symmetric(sp.Matrix([[1, 5], [5, 2]]))
    assert not is_symmetric(sp.Matrix([[1, 5], [-5, 2]]))


def test_check_dimension():
    # m^2 v^2 has mass dimension 4
    ok, found = check_dimension(m**2 * v**2, {m: 1, v: 1}, 4)
    assert ok, found
    # Yukawa coupling is dimensionless
    ok, found = check_dimension(Y * m / v, {Y: 0, m: 1, v: 1}, 0)
    assert ok, found
    # wrong target fails
    ok, _ = check_dimension(m**2, {m: 1}, 3)
    assert not ok


def test_numeric_equal_true_identity():
    ok, diff = numeric_equal(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1),
                             [x], seed=42)
    assert ok, diff


def test_numeric_equal_detects_difference():
    ok, diff = numeric_equal(x**2, x**2 + sp.Rational(1, 1000), [x], seed=42)
    assert not ok


def test_seesaw_light_mass():
    mD = sp.Symbol("m_D", positive=True)
    MR = sp.Symbol("M_R", positive=True)
    result = seesaw_light_mass(sp.Matrix([[mD]]), sp.Matrix([[MR]]))
    assert sp.simplify(result[0, 0] + mD**2 / MR) == 0
