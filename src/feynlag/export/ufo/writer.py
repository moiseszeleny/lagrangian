"""UFO model directory writer.

v1 scope (see plan): unitary-gauge tree-level UFO with the closed vertex
catalog; no form factors, NLO, ghosts or propagators.py. Colored (SU(3))
vertices emit real color-tensor strings (T/f) — see add_vvv_vertex /
add_vvvv_vertex / add_fermion_vertex; the SSS/SSSS/VSS/VVS/VVSS catalog
(add_bosonic_vertex) remains color-singlet only. Coupling orders use a
simple heuristic (``QED`` power = #legs − 2) — refine per-model if a
generator needs the exact hierarchy.

The bosonic vertices come straight from :meth:`Model.vertices`.  Fermion
vertices must be flavor-resolved by the caller (each entry names concrete
particles) since UFO particles are per flavor eigenstate.
"""

import datetime
import keyword
import re
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from ...operators import momentum
from .lorentz_map import UFO_LORENTZ, structures_for
from .pycode import ufo_expr
from .static import FUNCTION_LIBRARY, INIT_TEMPLATE, OBJECT_LIBRARY

__all__ = ["UFOParticle", "write_ufo"]


@dataclass
class UFOParticle:
    """Specification of one UFO particle.

    Attributes:
        symbol: physical SymPy symbol used in the vertex tables.
        pdg: PDG code.
        name / antiname: UFO names (equal → self-conjugate).
        spin: UFO ``2s+1`` (1 scalar, 2 fermion, 3 vector).
        mass / width: parameter names, or ``'ZERO'``.
        color: 1 (singlet), 3, 8.
        charge: electric charge.
        antisymbol: the symbol used for the conjugate in vertex tables
            (e.g. the ``Gm`` partner of ``Gp``); None if self-conjugate.
        goldstone: mark Goldstone bosons.
        texname / antitexname: LaTeX (defaults to names).
    """

    symbol: object
    pdg: int
    name: str
    antiname: str = None
    spin: int = 1
    mass: str = "ZERO"
    width: str = "ZERO"
    color: int = 1
    charge: float = 0
    antisymbol: object = None
    goldstone: bool = False
    texname: str = None
    antitexname: str = None

    def __post_init__(self):
        if self.antiname is None:
            self.antiname = self.name
        if self.texname is None:
            self.texname = self.name
        if self.antitexname is None:
            self.antitexname = self.antiname

    @property
    def self_conjugate(self):
        return self.name == self.antiname


def _pyname(name):
    """Valid python identifier for a particle name (FeynRules-style)."""
    replacements = [("+", "__plus__"), ("-", "__minus__"), ("~", "__tilde__"),
                    ("@", "__at__"), ("!", "__exclam__"), ("?", "__quest__"),
                    ("*", "__star__")]
    for orig, sub in replacements:
        name = name.replace(orig, sub)
    name = re.sub(r"\W", "_", name)
    if keyword.iskeyword(name) or name[0].isdigit():
        name = "_" + name
    return name


def _vss_split(coupling, scalars):
    """Split a VSS rule ``c·(p(a) − p(b))`` into (ordered scalars, c).

    Returns ``((a, b), c)`` such that the rule equals ``c*(p(a) − p(b))``.
    """
    a, b = scalars
    pa, pb = momentum(a), momentum(b)
    coupling = sp.expand(coupling)
    c_a = sp.simplify(coupling.coeff(pa))
    c_b = sp.simplify(coupling.coeff(pb))
    if sp.simplify(c_a + c_b) != 0:
        raise ValueError(f"VSS coupling is not antisymmetric in momenta: "
                         f"{coupling}")
    residual = sp.simplify(coupling - c_a * pa - c_b * pb)
    if residual != 0:
        raise ValueError(f"VSS coupling has non-momentum part: {coupling}")
    return (a, b), c_a


class _UFOBuilder:
    def __init__(self, model_name, parameters, particles):
        self.model_name = model_name
        self.parameters = parameters
        self.specs = {p.symbol: p for p in particles}
        for p in particles:
            if p.antisymbol is not None:
                self.specs[p.antisymbol] = p
        self.particles = particles
        self.couplings = {}          # value string -> coupling name
        # (particles, lorentz names, couplings, color strings) — colors has
        # either 1 entry (broadcast to every lorentz/coupling slot, the
        # color-singlet default) or exactly len(lorentz names) entries (one
        # color tensor per independent structure, e.g. the 3-way f*f
        # decomposition of a 4-gluon vertex).
        self.vertex_entries = []
        self.used_lorentz = set()

    # -------------------------------------------------------------- helpers

    def _particle_ref(self, symbol):
        spec = self.specs.get(symbol)
        if spec is None:
            raise KeyError(f"no UFOParticle registered for symbol {symbol}")
        if symbol == spec.antisymbol and not spec.self_conjugate:
            return f"P.{_pyname(spec.antiname)}"
        return f"P.{_pyname(spec.name)}"

    def _coupling(self, expr, n_legs):
        value = ufo_expr(sp.nsimplify(expr, rational=False)
                         if expr.is_number else expr)
        if value not in self.couplings:
            cname = f"GC_{len(self.couplings) + 1}"
            order = {"QED": max(n_legs - 2, 1)}
            self.couplings[value] = (cname, order)
        return self.couplings[value][0]

    # -------------------------------------------------------------- vertices

    def add_bosonic_vertex(self, vertex):
        """Add a feynlag Vertex (SSS/SSSS/VSS/VVS/VVSS).

        This catalog is color-singlet only in v1 (Higgs/EW self-couplings
        after EWSB) — raises if a non-singlet (color-charged) particle is
        passed in, rather than silently emitting a wrong ``color=['1']`` for
        a vertex this adder doesn't actually know how to color-tensor
        (colored self-couplings go through :meth:`add_vvv_vertex` /
        :meth:`add_vvvv_vertex`, colored fermion currents through
        :meth:`add_fermion_vertex`).
        """
        vtype = vertex.vertex_type
        particles = list(vertex.particles)
        coupling = vertex.coupling
        n = len(particles)
        for p in particles:
            if self.specs[p].color != 1:
                raise ValueError(
                    f"add_bosonic_vertex is color-singlet only; {p} has "
                    f"color {self.specs[p].color} — use add_vvv_vertex/"
                    f"add_vvvv_vertex for colored self-couplings")

        if vtype in ("SSS", "SSSS", "VVS", "VVSS"):
            spins = {p: self.specs[p].spin for p in particles}
            ordered = sorted(particles, key=lambda p: -spins[p])
            lorentz = structures_for(vtype)[0]
            cname = self._coupling(coupling, n)
        elif vtype == "VSS":
            vector = [p for p in particles if self.specs[p].spin == 3]
            scalars = [p for p in particles if self.specs[p].spin == 1]
            if len(vector) != 1 or len(scalars) != 2:
                raise ValueError(f"bad VSS content {particles}")
            (a, b), c = _vss_split(coupling, scalars)
            ordered = [vector[0], a, b]
            lorentz = "VSS1"
            cname = self._coupling(c, n)
        else:
            raise ValueError(f"add_bosonic_vertex cannot handle {vtype}; "
                             f"use the dedicated adders")
        self.used_lorentz.add(lorentz)
        self.vertex_entries.append(
            ([self._particle_ref(p) for p in ordered], [lorentz], [cname],
             ["1"]))

    def add_vvv_vertex(self, triple, coupling, color="1"):
        """Yang–Mills cubic vertex: ``coupling`` multiplies the VVV1
        structure (one representative ordering per set of three bosons).

        Args:
            color: UFO color-tensor string (default ``'1'``, singlet —
                e.g. EW self-couplings after EWSB). An unbroken non-abelian
                self-coupling (e.g. ggg) is ONE physical particle repeated
                three times with ``color='f(1,2,3)'`` — never built by
                iterating the group's full weak-basis component dict (that
                dict is for internal verification only, see yangmills.py).
        """
        self.used_lorentz.add("VVV1")
        cname = self._coupling(coupling, 3)
        self.vertex_entries.append(
            ([self._particle_ref(p) for p in triple], ["VVV1"], [cname],
             [color]))

    def add_vvvv_vertex(self, quadruple, couplings, colors=None):
        """Quartic gauge vertex: dict ``{VVVV structure name: coupling}``.

        Args:
            colors: optional ``{VVVV structure name: color string}``, one
                color tensor per independent Lorentz structure (e.g. the
                three ``f*f`` color factors of a 4-gluon vertex, paired with
                VVVV1/2/3 respectively — see export/ufo/vvvv.py). Defaults
                to broadcasting the singlet ``'1'`` to every structure.
        """
        names, cnames, clist = [], [], []
        for lname, coupling in couplings.items():
            self.used_lorentz.add(lname)
            names.append(lname)
            cnames.append(self._coupling(coupling, 4))
            clist.append((colors or {}).get(lname, "1"))
        self.vertex_entries.append(
            ([self._particle_ref(p) for p in quadruple], names, cnames,
             clist))

    def add_fermion_vertex(self, bar_symbol, field_symbol, bosons,
                           left_coupling=0, right_coupling=0, color="1"):
        """Flavor-resolved FFS/FFV vertex.

        Args:
            bar_symbol / field_symbol: particle-spec symbols of the fermions
                (the ψ̄ leg first).
            bosons: tuple with the boson leg symbol(s) (length 1 in v1).
            left_coupling / right_coupling: coefficients of P_L / P_R
                (or γ^μ P_L / γ^μ P_R).
            color: UFO color-tensor string (default ``'1'``, singlet — e.g.
                a lepton current). A qqg vertex uses ``'T(3,1,2)'`` (gluon
                at the boson position, matching this adder's own
                ``[bar, field, boson]`` leg order and
                ``fermion_gauge_current``'s ``T[r,c]`` convention, where
                ``r``=bar-leg index=position 1, ``c``=field-leg
                index=position 2).
        """
        if len(bosons) != 1:
            raise ValueError("v1 fermion vertices have exactly one boson leg")
        boson_spin = self.specs[bosons[0]].spin
        base = "FFV" if boson_spin == 3 else "FFS"
        ordered = [bar_symbol, field_symbol, bosons[0]]
        names, cnames = [], []
        # ``left/right`` are Lagrangian coefficients; the UFO coupling value
        # carries the Feynman-rule ``i`` (as the bosonic/VVV couplings do), so
        # apply it here — its omission is invisible to a single vertex but
        # breaks FFV↔VVV interference (e.g. e+e- → W+W- gauge cancellation).
        for suffix, coupling in (("L", sp.I * left_coupling),
                                 ("R", sp.I * right_coupling)):
            if coupling == 0:
                continue
            lname = base + suffix
            self.used_lorentz.add(lname)
            names.append(lname)
            cnames.append(self._coupling(coupling, 3))
        if not names:
            return
        self.vertex_entries.append(
            ([self._particle_ref(p) for p in ordered], names, cnames,
             [color]))

    # ----------------------------------------------------------------- write

    def write(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        (path / "object_library.py").write_text(OBJECT_LIBRARY)
        (path / "function_library.py").write_text(FUNCTION_LIBRARY)
        (path / "__init__.py").write_text(INIT_TEMPLATE.format(
            author="feynlag", date=datetime.date.today().isoformat(),
            version="0.1"))
        (path / "coupling_orders.py").write_text(self._coupling_orders_py())
        (path / "parameters.py").write_text(self._parameters_py())
        (path / "particles.py").write_text(self._particles_py())
        (path / "lorentz.py").write_text(self._lorentz_py())
        (path / "couplings.py").write_text(self._couplings_py())
        (path / "vertices.py").write_text(self._vertices_py())
        return path

    def _coupling_orders_py(self):
        return (
            "from object_library import all_orders, CouplingOrder\n\n"
            "QCD = CouplingOrder(name='QCD', expansion_order=99, "
            "hierarchy=1)\n"
            "QED = CouplingOrder(name='QED', expansion_order=99, "
            "hierarchy=2)\n")

    def _parameters_py(self):
        lines = ["# This file was automatically created by feynlag",
                 "import cmath",
                 "from object_library import all_parameters, Parameter",
                 "from function_library import (complexconjugate, re, im, "
                 "csc, sec, acsc, asec, cot)", "", ""]
        lines.append("ZERO = Parameter(name='ZERO', nature='internal', "
                     "type='real', value='0.0', texname='0')")
        lines.append("")
        for idx, p in enumerate(self.parameters.externals, start=1):
            if p.value is None:
                raise ValueError(f"external parameter {p.name} needs a "
                                 f"numeric value for UFO export")
            value = float(sp.sympify(p.value))
            lines.append(
                f"{p.name} = Parameter(name='{p.name}', nature='external', "
                f"type='real', value={value!r}, texname='{p.tex}', "
                f"lhablock='FEYNLAG', lhacode=[{idx}])")
        lines.append("")
        for p in self.parameters.dependency_order():
            value = ufo_expr(p.expr)
            lines.append(
                f"{p.name} = Parameter(name='{p.name}', nature='internal', "
                f"type='complex', value={value!r}, texname='{p.tex}')")
        lines.append("")
        return "\n".join(lines)

    def _particles_py(self):
        lines = ["# This file was automatically created by feynlag",
                 "from object_library import all_particles, Particle",
                 "import parameters as Param", "", ""]
        for spec in self.particles:
            mass = "Param.ZERO" if spec.mass == "ZERO" else f"Param.{spec.mass}"
            width = ("Param.ZERO" if spec.width == "ZERO"
                     else f"Param.{spec.width}")
            lines.append(
                f"{_pyname(spec.name)} = Particle(pdg_code={spec.pdg}, "
                f"name='{spec.name}', antiname='{spec.antiname}', "
                f"spin={spec.spin}, color={spec.color}, mass={mass}, "
                f"width={width}, texname='{spec.texname}', "
                f"antitexname='{spec.antitexname}', charge={spec.charge}, "
                f"goldstoneboson={spec.goldstone})")
            if not spec.self_conjugate:
                lines.append(f"{_pyname(spec.antiname)} = "
                             f"{_pyname(spec.name)}.anti()")
            lines.append("")
        return "\n".join(lines)

    def _lorentz_py(self):
        lines = ["# This file was automatically created by feynlag",
                 "from object_library import all_lorentz, Lorentz", "", ""]
        for name in sorted(self.used_lorentz):
            spins, structure = UFO_LORENTZ[name]
            lines.append(f"{name} = Lorentz(name='{name}', spins={spins}, "
                         f"structure='{structure}')")
        lines.append("")
        return "\n".join(lines)

    def _couplings_py(self):
        lines = ["# This file was automatically created by feynlag",
                 "import cmath",
                 "from object_library import all_couplings, Coupling",
                 "from function_library import (complexconjugate, re, im, "
                 "csc, sec, acsc, asec, cot)", "", ""]
        for value, (cname, order) in self.couplings.items():
            lines.append(f"{cname} = Coupling(name='{cname}', "
                         f"value={value!r}, order={order})")
        lines.append("")
        return "\n".join(lines)

    def _vertices_py(self):
        lines = ["# This file was automatically created by feynlag",
                 "from object_library import all_vertices, Vertex",
                 "import particles as P",
                 "import couplings as C",
                 "import lorentz as L", "", ""]
        for n, (parts, lnames, cnames, colors) in enumerate(
                self.vertex_entries, start=1):
            lorentz = ", ".join(f"L.{ln}" for ln in lnames)
            if len(colors) == 1:
                # single shared color structure — every lorentz/coupling
                # slot pairs with color-tensor index 0 (UFO's (0,i) key).
                couplings = ", ".join(f"(0,{i}):C.{cn}"
                                      for i, cn in enumerate(cnames))
            elif len(colors) == len(cnames):
                # one color tensor per lorentz structure (e.g. the 3-way
                # f*f decomposition of a 4-gluon vertex) — paired diagonally.
                couplings = ", ".join(f"({i},{i}):C.{cn}"
                                      for i, cn in enumerate(cnames))
            else:
                raise ValueError(
                    f"vertex V_{n}: {len(colors)} color structures for "
                    f"{len(cnames)} couplings — must be 1 (broadcast) or "
                    f"match exactly")
            color = "[" + ", ".join(f"'{c}'" for c in colors) + "]"
            lines.append(
                f"V_{n} = Vertex(name='V_{n}',\n"
                f"    particles=[{', '.join(parts)}],\n"
                f"    color={color},\n"
                f"    lorentz=[{lorentz}],\n"
                f"    couplings={{{couplings}}})")
            lines.append("")
        return "\n".join(lines)


def write_ufo(path, model_name, parameters, particles, bosonic_vertices=(),
              vvv=None, vvvv=None, fermion_vertices=(), vvv_colors=None,
              vvvv_colors=None):
    """Write a UFO model directory.

    Args:
        path: output directory.
        model_name: UFO model name.
        parameters: :class:`~feynlag.parameters.ParameterSet` (externals need
            numeric values; internals need defined ``expr``).
        particles: iterable of :class:`UFOParticle`.
        bosonic_vertices: feynlag ``Vertex`` objects
            (SSS/SSSS/VSS/VVS/VVSS) — color-singlet only (see
            :meth:`_UFOBuilder.add_bosonic_vertex`).
        vvv: dict ``{(V1,V2,V3): coupling}`` (one representative ordering
            per boson triple), e.g. from ``cubic_couplings`` for an
            EWSB-rotated, color-singlet physical basis (Wp/Wm/Z/A-style).
            An unbroken non-abelian self-coupling (e.g. ggg) is instead ONE
            physical particle repeated three times, still expressible as a
            single-entry ``{(g,g,g): coupling}`` dict — pair it with
            ``vvv_colors``.
        vvvv: dict ``{(V1,V2,V3,V4): {lorentz-name: coupling}}``.
        vvv_colors / vvvv_colors: optional ``{key: color-string}`` /
            ``{key: {lorentz-name: color-string}}`` matching ``vvv``/
            ``vvvv``'s keys, for non-singlet (color-tensor) structures —
            defaults to broadcasting the singlet ``'1'`` when a key is
            absent (see :meth:`_UFOBuilder.add_vvv_vertex`/
            :meth:`_UFOBuilder.add_vvvv_vertex`).
        fermion_vertices: iterable of dicts with keys ``bar``, ``field``,
            ``bosons``, ``left``, ``right`` (flavor-resolved symbols), and
            optionally ``color`` (default ``'1'``; e.g. ``'T(3,1,2)'`` for a
            qqg vertex).

    Returns:
        the written path.
    """
    builder = _UFOBuilder(model_name, parameters, list(particles))
    for v in bosonic_vertices:
        builder.add_bosonic_vertex(v)
    for triple, coupling in (vvv or {}).items():
        builder.add_vvv_vertex(triple, coupling,
                               color=(vvv_colors or {}).get(triple, "1"))
    for quad, couplings in (vvvv or {}).items():
        builder.add_vvvv_vertex(quad, couplings,
                                colors=(vvvv_colors or {}).get(quad))
    for fv in fermion_vertices:
        builder.add_fermion_vertex(fv["bar"], fv["field"], fv["bosons"],
                                   fv.get("left", 0), fv.get("right", 0),
                                   color=fv.get("color", "1"))
    return builder.write(path)
