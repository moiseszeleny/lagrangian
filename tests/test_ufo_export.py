"""Phase 5 validation: UFO export of an SM-lite electroweak model.

Checks:
- the generated directory is a valid python package importable with the
  shipped object_library,
- internal parameters resolve numerically in dependency order,
- coupling values evaluate to the pinned SM numbers (i g m_W for hWW, ...),
- vertices reference existing particles/couplings/lorentz objects.
"""

import importlib
import sys

import pytest
import sympy as sp

from feynlag import (
    Dmu, ExternalParameter, InternalParameter, Lagrangian, Model,
    ParameterSet, Rotation, SU2, Scalar, U1, conjugate_pair, cubic_couplings,
    dag, rotation_2x2,
)
from feynlag.export.ufo import UFOParticle, write_ufo
from feynlag import verify_ufo_numeric


@pytest.fixture(scope="module")
def sm_ufo(tmp_path_factory):
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)
    MH = InternalParameter("MH", positive=True, unit_dim=1)
    MW = InternalParameter("MW", positive=True, unit_dim=1)
    MZ = InternalParameter("MZ", positive=True, unit_dim=1)
    MH.define(sp.sqrt(2 * lam.s) * v.s)
    MW.define(gw.s * v.s / 2)
    MZ.define(sp.sqrt(gw.s**2 + g1.s**2) * v.s / 2)

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})

    HdH = (dag(H) * H.mat)[0]
    V = -mu2.s * HdH + lam.s * HdH**2
    DH = Dmu(H)
    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-V, sector="potential")

    W = SU2L.bosons("W")
    B = U1Y.bosons("B")
    model = Model("SMlite", gauge_groups=[SU2L, U1Y], fields=[H, W, B],
                  parameters=[gw, g1, v, lam, mu2, MH, MW, MZ],
                  lagrangian=L)
    model.solve_tadpoles([mu2])

    g, gp = gw.s, g1.s
    W1, W2, W3 = W.components
    Bc = B.components[0]
    Z, A = sp.symbols("Z A", real=True)
    thetaW = sp.atan(gp / g)
    model.rotate(Rotation([W3, Bc], [Z, A], rotation_2x2(-thetaW)))
    Wp, Wm = sp.symbols("Wp Wm")
    U = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], U, kind="unitary"))

    h = sp.Symbol("H0_r", real=True)
    G0 = sp.Symbol("H0_i", real=True)
    Gp = H.components[0]
    Gm, cmap = conjugate_pair(Gp, "Gm")
    fields = [h, G0, Gp, Gm, Z, A, Wp, Wm]

    verts = (model.vertices(fields, sector="potential",
                            conjugate_map=cmap, simplifier=sp.simplify)
             + model.vertices(fields, sector="kinetic",
                              conjugate_map=cmap, simplifier=sp.simplify))

    cw, sw = g / sp.sqrt(g**2 + gp**2), gp / sp.sqrt(g**2 + gp**2)
    Umix = sp.Matrix([
        [1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0],
        [sp.I / sp.sqrt(2), -sp.I / sp.sqrt(2), 0, 0],
        [0, 0, cw, sw],
    ])
    cubic = cubic_couplings(SU2L, physical=[Wp, Wm, Z, A], U=Umix)
    # one representative ordering per boson triple (the tensor is totally
    # antisymmetric — UFO wants a single vertex per particle set)
    seen = set()
    vvv = {}
    for t, coupling in cubic.items():
        key = frozenset((s, list(t).count(s)) for s in t)
        if key not in seen:
            seen.add(key)
            vvv[t] = coupling

    particles = [
        UFOParticle(h, 25, "h", spin=1, mass="MH"),
        UFOParticle(G0, 250, "G0", spin=1, mass="MZ", goldstone=True),
        UFOParticle(Gp, 251, "G+", antiname="G-", spin=1, mass="MW",
                    charge=1, antisymbol=Gm, goldstone=True),
        UFOParticle(Z, 23, "Z", spin=3, mass="MZ"),
        UFOParticle(A, 22, "a", spin=3),
        UFOParticle(Wp, 24, "W+", antiname="W-", spin=3, mass="MW",
                    charge=1, antisymbol=Wm),
    ]

    out = tmp_path_factory.mktemp("ufo") / "SMlite_UFO"
    params = ParameterSet(gw, g1, v, lam, mu2, MH, MW, MZ)
    write_ufo(out, "SMlite", params, particles, bosonic_vertices=verts,
              vvv=vvv)
    return out, model, dict(g=0.6535, gp=0.3580, v=246.0, lam=0.129)


_UFO_SUBMODULES = ("object_library", "function_library", "coupling_orders",
                   "parameters", "couplings", "lorentz", "particles",
                   "vertices")


def _import_ufo(path):
    """Load a UFO directory the MadGraph way (dir on sys.path, absolute
    imports); return object_library holding the all_* registries."""
    sys.path.insert(0, str(path))
    for mod in _UFO_SUBMODULES:
        sys.modules.pop(mod, None)
    try:
        import object_library
        for mod in _UFO_SUBMODULES[1:]:
            importlib.import_module(mod)
        return object_library
    finally:
        sys.path.pop(0)


def test_ufo_roundtrip_evaluates_cleanly(sm_ufo):
    """Every exported parameter and coupling evaluates to a finite number —
    the self-verification FeynRules-style generators cannot do."""
    path, model, num = sm_ufo
    report = verify_ufo_numeric(path)
    assert report.ok, report.failures
    assert report.couplings          # at least one coupling was evaluated
    assert report.parameters


def test_ufo_uses_absolute_imports(sm_ufo):
    """UFO submodules must use absolute imports (``from object_library …``) so
    MadGraph can run them as standalone scripts (param-card generation); a
    relative ``from .object_library`` breaks with 'no known parent package'."""
    path, model, num = sm_ufo
    for fname in ("parameters.py", "couplings.py", "particles.py",
                  "vertices.py", "lorentz.py", "function_library.py",
                  "__init__.py"):
        text = (path / fname).read_text()
        assert "from ." not in text and "from . import" not in text, \
            f"{fname} has a relative import"


def test_ufo_roundtrip_reproduces_pinned_hWW(sm_ufo):
    """Some evaluated coupling equals the pinned hWW value i g² v/2."""
    path, model, num = sm_ufo
    report = verify_ufo_numeric(path)
    g, v = num["g"], num["v"]
    target = complex(0, 1) * g**2 * v / 2
    assert any(abs(val - target) < 1e-9 for val in report.couplings.values()), \
        sorted(report.couplings.items())


def test_ufo_roundtrip_flags_nonfinite(sm_ufo):
    """A non-finite external input propagates to a reported failure rather
    than a silent NaN — the check has teeth."""
    path, model, num = sm_ufo
    report = verify_ufo_numeric(path, external_values={"v": float("inf")})
    assert not report.ok


def test_validate_umbrella_includes_ufo_roundtrip(sm_ufo):
    """Model.validate(ufo_path=...) runs invariance, skips anomalies (no
    fermions), and round-trips the exported UFO — all in one report."""
    path, model, num = sm_ufo
    report = model.validate(ufo_path=path)
    assert report.ok, report.summary()
    assert report.checks["invariance"].ok
    assert report.checks["anomalies"] is None          # scalar+gauge only
    assert report.checks["ufo_roundtrip"].ok


def test_ufo_imports(sm_ufo):
    path, model, num = sm_ufo
    ufo = _import_ufo(path)
    assert ufo.all_particles
    assert ufo.all_vertices
    assert ufo.all_couplings
    assert ufo.all_lorentz
    assert ufo.all_parameters


def test_ufo_parameters_resolve(sm_ufo):
    path, model, num = sm_ufo
    ufo = _import_ufo(path)
    import cmath  # noqa: F401  (UFO value strings use cmath)

    namespace = {"cmath": cmath, "complexconjugate":
                 lambda z: complex(z).conjugate(), "abs": abs}
    values = {}
    for p in ufo.all_parameters:
        if p.nature == "external":
            values[p.name] = float(p.value)
        else:
            values[p.name] = complex(eval(p.value, namespace, dict(values)))
    # pinned numbers
    g, gp, v, lam = num["g"], num["gp"], num["v"], num["lam"]
    assert abs(values["MW"] - g * v / 2) < 1e-10
    assert abs(values["MZ"] - ((g**2 + gp**2) ** 0.5) * v / 2) < 1e-10
    assert abs(values["MH"] - (2 * lam) ** 0.5 * v) < 1e-10
    assert abs(values["mu2"] - lam * v**2) < 1e-10


def test_ufo_couplings_pinned(sm_ufo):
    """The hWW coupling value must equal i g² v/2 = i g m_W numerically."""
    path, model, num = sm_ufo
    ufo = _import_ufo(path)
    import cmath

    namespace = {"cmath": cmath, "complexconjugate":
                 lambda z: complex(z).conjugate(), "abs": abs,
                 "complex": complex}
    # resolve parameters into the namespace
    for p in ufo.all_parameters:
        namespace[p.name] = (float(p.value) if p.nature == "external"
                             else complex(eval(p.value, namespace)))

    g, v = num["g"], num["v"]
    target = complex(0, 1) * g**2 * v / 2

    # find the vertex with particles {h, W+, W-}
    for vert in ufo.all_vertices:
        names = sorted(p.name for p in vert.particles)
        if names == ["W+", "W-", "h"]:
            coupling = list(vert.couplings.values())[0]
            value = complex(eval(coupling.value, namespace))
            assert abs(value - target) < 1e-9, (value, target)
            break
    else:
        pytest.fail("hWW vertex not found in UFO output")


def test_ufo_vertices_reference_valid_objects(sm_ufo):
    path, model, num = sm_ufo
    ufo = _import_ufo(path)
    particle_names = {p.name for p in ufo.all_particles}
    coupling_names = {c.name for c in ufo.all_couplings}
    lorentz_names = {l.name for l in ufo.all_lorentz}
    for vert in ufo.all_vertices:
        for p in vert.particles:
            assert p.name in particle_names
        for l in vert.lorentz:
            assert l.name in lorentz_names
        for c in vert.couplings.values():
            assert c.name in coupling_names
        # spins of the lorentz structures match the particles
        for l in vert.lorentz:
            assert l.spins == [p.spin for p in vert.particles], \
                (vert.name, l.spins, [p.spin for p in vert.particles])
