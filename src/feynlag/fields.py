"""Particle field classes.

A :class:`Field` is declared with its gauge representations and expands into
explicit **component symbols** (the representation the whole pipeline works
in — commuting SymPy Symbols for bosons, ``IndexedBase`` for fermion flavor
structure).

- :class:`Scalar` — complex (default) or real; supports VEV registration
  ``φ⁰ → (v + h + i a)/√2`` (explicit 1/√2 per CONVENTIONS.md).
- :class:`Fermion` and its Weyl/Dirac/Majorana subclasses — chirality-aware,
  flavor-indexed (dynamics arrive in Phase 4).
- :class:`GaugeBoson` — created by gauge groups, one component per generator.

Discrete-symmetry membership is registered on the group
(``S3.assign('2', H1, H2)``), not in ``reps``, because discrete multiplets
usually span several fields.
"""

import sympy as sp

from .conventions import SQRT2
from .groups.base import GaugeGroup

__all__ = ["Field", "Scalar", "Fermion", "WeylFermion", "DiracFermion",
           "MajoranaFermion", "GaugeBoson", "dag", "hc", "conjugate_pair",
           "bar_partner"]

#: {component IndexedBase: bar IndexedBase} and its inverse, populated by
#: every Fermion at construction time — lets Bilinear's hermitian conjugate
#: (vertices/bilinear.py) find "the bar of this field" / "the field of this
#: bar" without threading a Fermion reference through every call site.
_BAR_PARTNER = {}


def bar_partner(indexed_base):
    """The paired IndexedBase for a Fermion component or bar component.

    ``bar_partner(psi) -> psibar`` and ``bar_partner(psibar) -> psi``.
    """
    return _BAR_PARTNER[indexed_base]


class Field:
    """Base class for all fields.

    Args:
        name: field name; used as prefix for auto-generated components.
        reps: dict ``{GaugeGroup: representation}`` (charge for a U(1),
            dimension label for SU(2)/SU(3)).
        component_names: explicit component names (physics-friendly, e.g.
            ``['H1p', 'H10']`` for an SU(2) doublet).  Default:
            ``name_1 … name_n`` (or just ``name`` for a singlet).
        real: whether components are real symbols.
        self_conjugate: particle is its own antiparticle.
        tex: LaTeX name.
    """

    def __init__(self, name, reps=None, component_names=None, real=False,
                 self_conjugate=False, tex=None):
        self.name = name
        self.tex = tex if tex is not None else name
        self.reps = dict(reps) if reps else {}
        self.self_conjugate = self_conjugate
        self.real = real

        for group in self.reps:
            if not isinstance(group, GaugeGroup):
                raise TypeError(
                    f"reps keys must be gauge groups; register discrete "
                    f"multiplets with group.assign(...): got {group!r}")

        self.components = self._make_components(component_names)
        #: column matrix of components — use in matrix expressions.
        self.mat = sp.Matrix(len(self.components), 1, self.components)

    # ------------------------------------------------------------------ setup

    @property
    def dim(self):
        """Total number of gauge components (product over non-abelian reps)."""
        d = 1
        for group, rep in self.reps.items():
            d *= group.rep_dim(rep)
        return d

    def _component_symbol(self, cname):
        assumptions = {"real": True} if self.real else {}
        return sp.Symbol(cname, **assumptions)

    def _make_components(self, component_names):
        n = self.dim
        if component_names is None:
            component_names = ([self.name] if n == 1 else
                               [f"{self.name}_{i + 1}" for i in range(n)])
        if len(component_names) != n:
            raise ValueError(f"{self.name}: expected {n} component names, "
                             f"got {len(component_names)}")
        return [self._component_symbol(c) for c in component_names]

    # ------------------------------------------------------------- interface

    def dag(self):
        """Hermitian conjugate of the component column: a row Matrix."""
        return self.mat.conjugate().T

    def generators(self, group):
        """Generator matrices of ``group`` acting on this field's FULL
        component space (Kronecker product over all non-abelian reps).
        """
        if group not in self.reps:
            # singlet: transforms trivially
            return [sp.zeros(self.dim, self.dim)] * group.n_generators

        # Build the Kronecker factors in declaration order.
        factor_dims = [(g, g.rep_dim(r)) for g, r in self.reps.items()]
        gens = []
        for T in group.generators(self.reps[group]):
            M = sp.Matrix([[1]])
            for g, d in factor_dims:
                block = T if g == group else sp.eye(d)
                M = sp.Matrix(sp.kronecker_product(M, block))
            gens.append(M)
        return gens

    def charge(self, u1_group):
        """U(1) charge of this field (0 if not charged under it)."""
        return sp.sympify(self.reps.get(u1_group, 0))

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"

    def __getitem__(self, i):
        return self.components[i]

    def __len__(self):
        return len(self.components)


class Scalar(Field):
    """Spin-0 field.

    ``vev_expansions`` maps a component symbol to its EWSB decomposition
    ``(vev, real_part, imag_part)``: complex components expand as
    ``φ → (v + h + i a)/√2`` (explicit 1/√2, CONVENTIONS.md); real components
    shift in place as ``φ → v + φ`` (no 1/√2 — that normalization belongs to
    the complex-field decomposition).
    """

    spin = 0

    def __init__(self, name, reps=None, component_names=None, real=False,
                 self_conjugate=None, tex=None):
        if self_conjugate is None:
            self_conjugate = real
        super().__init__(name, reps=reps, component_names=component_names,
                         real=real, self_conjugate=self_conjugate, tex=tex)
        #: {component: (vev_symbol, re_symbol, im_symbol_or_None)}
        self.vev_expansions = {}

    def expand_vev(self, vev_map):
        """Register VEVs for components: ``{component_symbol: vev}``.

        For a complex component ``phi``, creates real symbols ``phi_r``,
        ``phi_i`` and registers ``phi → (v + phi_r + i·phi_i)/√2``.
        For a real component, registers ``phi → v + phi``.

        ``vev`` may be a Parameter or a SymPy symbol/expression.
        """
        for comp, vev in vev_map.items():
            if comp not in self.components:
                raise ValueError(f"{comp} is not a component of {self.name}")
            vev_expr = sp.sympify(getattr(vev, "symbol", vev))
            if self.real:
                self.vev_expansions[comp] = (vev_expr, comp, None)
            else:
                re = sp.Symbol(f"{comp.name}_r", real=True)
                im = sp.Symbol(f"{comp.name}_i", real=True)
                self.vev_expansions[comp] = (vev_expr, re, im)
        return self

    @property
    def shift_map(self):
        """Substitution dict expanding VEV'd components around the vacuum."""
        shift = {}
        for comp, (vev, re, im) in self.vev_expansions.items():
            if im is None:
                shift[comp] = vev + comp
            else:
                shift[comp] = (vev + re + sp.I * im) / SQRT2
        return shift

    @property
    def fluctuations(self):
        """All real fluctuation symbols introduced by :meth:`expand_vev`."""
        out = []
        for comp, (vev, re, im) in self.vev_expansions.items():
            out.append(re)
            if im is not None:
                out.append(im)
        return out


class Fermion(Field):
    """Spin-1/2 field with chirality and flavor structure.

    Components are ``IndexedBase`` objects: ``psi[i]`` is flavor ``i`` of a
    gauge component.  Full fermionic dynamics (bilinear vertex extraction,
    SVD/Takagi diagonalization) arrive in Phase 4; the declaration layer
    already supports invariance checking through the gauge representations.
    """

    spin = sp.Rational(1, 2)

    def __init__(self, name, reps=None, component_names=None, chirality=None,
                 nflavors=1, self_conjugate=False, tex=None):
        if chirality not in (None, "L", "R"):
            raise ValueError("chirality must be 'L', 'R' or None")
        self.chirality = chirality
        self.nflavors = nflavors
        super().__init__(name, reps=reps, component_names=component_names,
                         real=False, self_conjugate=self_conjugate, tex=tex)
        #: Dirac adjoints ψ̄, one IndexedBase per gauge component
        self.bar_components = [sp.IndexedBase(f"{c.label}bar")
                               for c in self.components]
        for comp, bar in zip(self.components, self.bar_components):
            _BAR_PARTNER[comp] = bar
            _BAR_PARTNER[bar] = comp

    def _component_symbol(self, cname):
        return sp.IndexedBase(cname)

    def bar(self, component):
        """The ψ̄ IndexedBase matching a gauge component."""
        return self.bar_components[self.components.index(component)]

    def dag(self):
        raise NotImplementedError(
            "use the fermion bilinear track (feynlag.vertices.bilinear); "
            "fermions have no bosonic dagger")


class WeylFermion(Fermion):
    """Two-component chiral fermion (chirality required)."""

    def __init__(self, name, reps=None, chirality="L", **kwargs):
        if chirality not in ("L", "R"):
            raise ValueError("a Weyl fermion needs chirality 'L' or 'R'")
        super().__init__(name, reps=reps, chirality=chirality, **kwargs)


class DiracFermion(Fermion):
    """Four-component fermion (both chiralities)."""


class MajoranaFermion(Fermion):
    """Self-conjugate fermion; mass matrix is complex symmetric (Takagi)."""

    def __init__(self, name, reps=None, **kwargs):
        kwargs["self_conjugate"] = True
        super().__init__(name, reps=reps, **kwargs)


class GaugeBoson(Field):
    """Spin-1 gauge field; one real component per group generator.

    Create through the group:  ``SU2L.bosons('W')`` → components W_1..W_3.
    """

    spin = 1

    def __init__(self, name, group, component_names=None, tex=None):
        self.group = group
        n = group.n_generators
        if component_names is None and n > 1:
            component_names = [f"{name}_{i + 1}" for i in range(n)]
        reps = {}
        if not group.abelian:
            # adjoint rep label: dimension == number of generators
            reps = {group: n}
        super().__init__(name, reps=reps,
                         component_names=component_names if n > 1 else [name],
                         real=True, self_conjugate=True, tex=tex)


def _gauge_bosons(self, name=None):
    """Create (once) and return this group's gauge boson field."""
    if self._gauge_bosons is None:
        self._gauge_bosons = GaugeBoson(name or self.name, self)
    return self._gauge_bosons


GaugeGroup.bosons = _gauge_bosons


def dag(obj):
    """Hermitian conjugate: Field → row Matrix; Matrix → ``.H``; expr →
    complex conjugate (valid for bosonic expressions)."""
    if isinstance(obj, Field):
        return obj.dag()
    if isinstance(obj, sp.MatrixBase):
        return obj.conjugate().T
    return sp.conjugate(obj)


def hc(expr):
    """Hermitian conjugate of a bosonic Lagrangian term (complex conjugate).

    Write ``term + hc(term)`` for ``term + h.c.``.  Fermion bilinears need
    the Phase-4 machinery.
    """
    return sp.conjugate(expr)


def conjugate_pair(sym, name=None):
    """Antiparticle symbol for a complex physical field.

    The vertex extractor works on plain Symbols, so ``conjugate(G⁺)`` must be
    rewritten as the physical ``G⁻`` symbol first.

    Args:
        sym: the complex field symbol (e.g. ``Gp``).
        name: name for the conjugate symbol (default ``{sym}_c``; pass the
            physics name, e.g. ``'Gm'``).

    Returns:
        ``(partner_symbol, {conjugate(sym): partner_symbol})`` — merge the
        dicts of several pairs and pass as ``conjugate_map`` to
        ``Model.interactions`` / ``Model.feynman_rules``.
    """
    partner = sp.Symbol(name if name is not None else f"{sym.name}_c")
    return partner, {sp.conjugate(sym): partner}
