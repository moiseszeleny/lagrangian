"""UFO color-tensor export for the QCD (SU(3)) sector.

Checks that add_fermion_vertex/add_vvv_vertex/add_vvvv_vertex emit real
color-tensor strings (not the color-singlet '1' default) for a
quark-gluon (qqg), 3-gluon (ggg), and 4-gluon (gggg) vertex, with numeric
coupling values matching the already-pinned physics in tests/test_qcd.py
and tests/test_yangmills.py.
"""

import sympy as sp
import pytest

from feynlag import ExternalParameter, ParameterSet, SU3, quartic_couplings
from feynlag.export.ufo import UFOParticle, assemble_vvvv, write_ufo

import importlib
import sys


def _import_ufo(path):
    sys.path.insert(0, str(path.parent))
    try:
        return importlib.import_module(path.name)
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def qcd_ufo(tmp_path_factory):
    gs = ExternalParameter("gs", 1.22, positive=True)

    q = sp.Symbol("q")
    qbar = sp.Symbol("qbar")
    g = sp.Symbol("g")

    particles = [
        UFOParticle(q, 1, "q", antiname="q~", spin=2, color=3,
                    antisymbol=qbar),
        UFOParticle(g, 21, "g", spin=3, color=8),
    ]

    params = ParameterSet(gs)

    # qqg: T^1_{0,1} = 1/2 (Gell-Mann lambda^1/2) => coefficient +gs/2,
    # matching tests/test_qcd.py::test_qqg_coupling_pinned.
    fermion_vertices = [
        dict(bar=qbar, field=q, bosons=(g,), left=gs.s / 2,
             color="T(3,1,2)"),
    ]
    # ggg: f^123=1 => coupling exactly -gs, matching
    # tests/test_qcd.py::test_ggg_coupling_pinned. One physical gluon
    # particle repeated three times — NOT the 8-component weak-basis dict.
    vvv = {(g, g, g): -gs.s}
    vvv_colors = {(g, g, g): "f(1,2,3)"}

    # gggg: couplings come from the actual group's weak-basis adjoint
    # components (G_1..G_8) via assemble_vvvv — the color-tensor sum is
    # what carries the adjoint index, so the UFO vertex itself is still
    # ONE physical gluon repeated four times, matching
    # tests/test_qcd.py::test_gggg_coupling_pinned's (G_1,G_2,G_4,G_5).
    SU3c = SU3("SU3c", coupling=gs)
    G = SU3c.bosons("G")
    G1, G2, G4, G5 = (G.components[0], G.components[1], G.components[3],
                      G.components[4])
    gggg_couplings = assemble_vvvv(quartic_couplings(SU3c), (G1, G2, G4, G5))
    vvvv = {(g, g, g, g): gggg_couplings}
    vvvv_colors = {(g, g, g, g): {
        "VVVV1": "f(1,2,-1)*f(3,4,-1)",
        "VVVV2": "f(1,3,-1)*f(2,4,-1)",
        "VVVV3": "f(1,4,-1)*f(2,3,-1)",
    }}

    out = tmp_path_factory.mktemp("ufo") / "QCD_UFO"
    write_ufo(out, "QCD", params, particles, vvv=vvv, vvv_colors=vvv_colors,
              vvvv=vvvv, vvvv_colors=vvvv_colors,
              fermion_vertices=fermion_vertices)
    return out, dict(gs=1.22)


def test_qqg_color_string(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    names = {p.name for p in ufo.all_particles}
    assert {"q", "q~", "g"} <= names
    for vert in ufo.all_vertices:
        pnames = sorted(p.name for p in vert.particles)
        if pnames == sorted(["q", "q~", "g"]):
            assert vert.color == ["T(3,1,2)"]
            break
    else:
        pytest.fail("qqg vertex not found")


def test_ggg_color_string_and_coupling(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    for vert in ufo.all_vertices:
        if [p.name for p in vert.particles] == ["g", "g", "g"]:
            assert vert.color == ["f(1,2,3)"]
            coupling = list(vert.couplings.values())[0]
            value = complex(eval(coupling.value, {"gs": num["gs"]}))
            assert abs(value - (-num["gs"])) < 1e-9
            break
    else:
        pytest.fail("ggg vertex not found")


def test_gggg_color_strings_and_couplings(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    expected_colors = {
        "VVVV1": "f(1,2,-1)*f(3,4,-1)",
        "VVVV2": "f(1,3,-1)*f(2,4,-1)",
        "VVVV3": "f(1,4,-1)*f(2,3,-1)",
    }
    expected_values = {
        "VVVV1": 1.5 * num["gs"] ** 2,
        "VVVV2": 0.75 * num["gs"] ** 2,
        "VVVV3": -0.75 * num["gs"] ** 2,
    }
    for vert in ufo.all_vertices:
        if [p.name for p in vert.particles] == ["g", "g", "g", "g"]:
            lorentz_names = [l.name for l in vert.lorentz]
            assert set(lorentz_names) == set(expected_colors)
            assert vert.color == [expected_colors[n] for n in lorentz_names]
            for slot, lname in enumerate(lorentz_names):
                coupling = vert.couplings[(slot, slot)]
                value = complex(eval(coupling.value, {"gs": num["gs"]}))
                assert abs(value - expected_values[lname]) < 1e-9
            break
    else:
        pytest.fail("gggg vertex not found")


def test_gluon_particle_is_color_octet_self_conjugate(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    (gluon,) = [p for p in ufo.all_particles if p.name == "g"]
    assert gluon.color == 8
    assert gluon.selfconjugate


def test_quark_antiquark_color_conjugate(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    quark = next(p for p in ufo.all_particles if p.name == "q")
    antiquark = next(p for p in ufo.all_particles if p.name == "q~")
    assert quark.color == 3
    assert antiquark.color == -3
