"""Jupyter ``_repr_latex_`` smoke tests for the core display-facing classes."""

import sympy as sp

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, ParameterSet, Rotation,
    Scalar, SU2, Vertex, rotation_2x2,
)


def _assert_latex(s):
    assert isinstance(s, str)
    assert s.startswith("$")
    assert s.endswith("$")
    return s


def test_parameter_repr_latex_uses_tex_override():
    theta = ExternalParameter("theta_c", 0.1, tex=r"\theta_c")
    assert _assert_latex(theta._repr_latex_()) == r"$\displaystyle \theta_c$"


def test_parameter_repr_latex_falls_back_to_sympy_printer():
    mu1 = InternalParameter("mu1")
    tex = _assert_latex(mu1._repr_latex_())
    assert r"\mu_{1}" in tex


def test_parameter_set_repr_latex():
    v = ExternalParameter("v", 246.0, positive=True)
    lam = InternalParameter("lam")
    lam.define(v.s / 2)
    ps = ParameterSet(v, lam)
    tex = _assert_latex(ps._repr_latex_())
    assert r"\begin{array}" in tex
    assert "v" in tex and "lam" in tex


def test_field_repr_latex_singlet_and_multiplet():
    s = Scalar("sigma")
    _assert_latex(s._repr_latex_())

    gw = sp.Symbol("g_w")
    SU2L = SU2("SU2L", coupling=gw)
    H = Scalar("H", reps={SU2L: 2}, component_names=["Hp", "H0"])
    tex = _assert_latex(H._repr_latex_())
    assert "Hp" in tex and "H_{0}" in tex


def test_vertex_repr_latex():
    a, b, c = sp.symbols("a b c")
    vtx = Vertex((a, b, c), sp.I * sp.Symbol("g"), vertex_type="VVS")
    tex = _assert_latex(vtx._repr_latex_())
    assert "g" in tex


def test_lagrangian_repr_latex():
    phi = sp.Symbol("phi")
    lam = sp.Symbol("lambda")
    L = Lagrangian()
    L.add(-lam * phi**4 / 24, sector="potential")
    tex = _assert_latex(L._repr_latex_())
    assert r"\mathcal{L}" in tex


def test_rotation_repr_latex():
    h, H = sp.symbols("h H")
    phi1, phi2 = sp.symbols("phi_1 phi_2")
    theta = sp.Symbol("theta")
    rot = Rotation([phi1, phi2], [h, H], rotation_2x2(theta))
    tex = _assert_latex(rot._repr_latex_())
    assert "=" in tex
