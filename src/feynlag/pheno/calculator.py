"""Partial widths, total widths and branching ratios from a feynlag model."""

from dataclasses import dataclass

import sympy as sp

from .amplitudes import amplitude_squared
from .kinematics import TwoBodyKinematics, is_allowed
from .vertices import collect_decay_vertices

__all__ = ["DecayCalculator", "DecayChannel", "partial_width"]


@dataclass
class DecayChannel:
    """One two-body channel and its partial width.

    Attributes:
        parent: the decaying field symbol.
        children: the two daughter symbols.
        width: the symbolic partial width.
        vertex: the :class:`~feynlag.pheno.vertices.DecayVertex` behind it.
        color_factor: colour multiplicity applied (``N_c``, default 1).
        symmetry_factor: ``1/2`` for identical daughters, else 1.
    """

    parent: sp.Expr
    children: tuple
    width: sp.Expr
    vertex: object = None
    color_factor: sp.Expr = sp.S.One
    symmetry_factor: sp.Expr = sp.S.One

    def __repr__(self):
        kids = " ".join(str(c) for c in self.children)
        return f"DecayChannel({self.parent} -> {kids} : {self.width})"


def partial_width(vertex, M, m1, m2, color_factor=1, symmetry_factor=1):
    """``Γ = S · N_c · ⟨|M|²⟩ · √λ(M²,m₁²,m₂²)/(16πM³)``.

    Returns ``0`` when the channel is closed (``M < m₁+m₂``) and that is
    decidable; an undecidable symbolic comparison keeps the channel.
    """
    if is_allowed(M, m1, m2) is False:
        return sp.S.Zero
    kin = TwoBodyKinematics(sp.sympify(M), sp.sympify(m1), sp.sympify(m2))
    return (sp.sympify(symmetry_factor) * sp.sympify(color_factor)
            * amplitude_squared(vertex, kin) * kin.phase_space())


class DecayCalculator:
    """Two-body decay widths and branching ratios for a built model.

    Args:
        model: a :class:`~feynlag.lagrangian.Model` in its physical basis
            (tadpoles solved, rotations registered).
        masses: ``{leg: mass}``. Fermion legs are ``Indexed`` objects, so a
            mass may be keyed by the exact leg or by its ``IndexedBase`` —
            the base is the fallback.
        boson_fields: physical boson symbols to extract vertices over.
        fermion_sectors: Lagrangian sectors carrying fermion bilinears
            (e.g. ``('gauge', 'yukawa')``).
        conjugate_map: passed through to the bosonic extractor.
        color_factors: ``{leg: N_c}``; anything absent defaults to 1.
        parameters: optional :class:`~feynlag.parameters.ParameterSet` for
            :meth:`numeric`.
        simplifier: optional callable applied to bosonic couplings.
        particle_map: ``{weyl_leg: physical_particle}`` identifying the two
            Weyl legs of one Dirac fermion, so its ``P_L`` and ``P_R``
            currents merge into a single channel — see
            :func:`~feynlag.pheno.vertices.resolve_leg`. Omitting it for a
            Dirac fermion splits ``Z → e⁺e⁻`` into two wrong half-channels.
    """

    def __init__(self, model, masses, boson_fields=(), fermion_sectors=(),
                 conjugate_map=None, color_factors=None, parameters=None,
                 simplifier=None, particle_map=None):
        self.model = model
        self.masses = dict(masses)
        self.boson_fields = list(boson_fields)
        self.fermion_sectors = tuple(fermion_sectors)
        self.conjugate_map = conjugate_map
        self.color_factors = dict(color_factors or {})
        self.parameters = parameters
        self.simplifier = simplifier
        self.particle_map = dict(particle_map or {})
        self._vertices = None

    # ------------------------------------------------------------- vertices

    def vertices(self):
        """All three-leg vertices, extracted once and cached."""
        if self._vertices is None:
            self._vertices = collect_decay_vertices(
                self.model, self.boson_fields,
                fermion_sectors=self.fermion_sectors,
                conjugate_map=self.conjugate_map,
                simplifier=self.simplifier,
                particle_map=self.particle_map)
        return self._vertices

    # --------------------------------------------------------------- lookup

    def mass_of(self, leg):
        """Mass of a leg, falling back to its ``IndexedBase``.

        Raises:
            KeyError: no mass declared — better than silently assuming zero,
                which would open every channel.
        """
        if leg in self.masses:
            return sp.sympify(self.masses[leg])
        base = getattr(leg, "base", None)
        if base is not None and base in self.masses:
            return sp.sympify(self.masses[base])
        raise KeyError(
            f"no mass declared for {leg!r}; add it to the calculator's "
            f"`masses` mapping (keyed by the leg or by its IndexedBase)")

    def _color_of(self, leg):
        if leg in self.color_factors:
            return sp.sympify(self.color_factors[leg])
        base = getattr(leg, "base", None)
        if base is not None and base in self.color_factors:
            return sp.sympify(self.color_factors[base])
        return sp.S.One

    # --------------------------------------------------------------- widths

    def channels(self, parent):
        """Every open two-body channel of ``parent``.

        A vertex contributes when one of its legs is ``parent``; the other two
        are the daughters. Legs with undeclared masses are skipped rather than
        guessed at.
        """
        M = self.mass_of(parent)
        out = []
        for vertex in self.vertices():
            legs = list(vertex.particles)
            if parent not in legs:
                continue
            remaining = list(legs)
            remaining.remove(parent)
            if len(remaining) != 2:
                continue
            try:
                m1, m2 = (self.mass_of(leg) for leg in remaining)
            except KeyError:
                continue
            symmetry = (sp.Rational(1, 2) if remaining[0] == remaining[1]
                        else sp.S.One)
            color = sp.S.One
            for leg in remaining:
                color *= self._color_of(leg)
            width = partial_width(vertex, M, m1, m2, color_factor=color,
                                  symmetry_factor=symmetry)
            if width == 0:
                continue
            out.append(DecayChannel(parent=parent, children=tuple(remaining),
                                    width=width, vertex=vertex,
                                    color_factor=color,
                                    symmetry_factor=symmetry))
        return out

    def partial_widths(self, parent):
        """``{(child1, child2): Γ}`` for every open channel."""
        widths = {}
        for channel in self.channels(parent):
            widths[channel.children] = (widths.get(channel.children, sp.S.Zero)
                                        + channel.width)
        return widths

    def total_width(self, parent):
        """``Γ_tot = Σ Γ_i`` over open two-body channels."""
        return sp.Add(*self.partial_widths(parent).values())

    def branching_ratios(self, parent):
        """``{(child1, child2): Γ_i/Γ_tot}``."""
        widths = self.partial_widths(parent)
        total = sp.Add(*widths.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in widths.items()}

    # -------------------------------------------------------------- numeric

    def _substitutions(self, extra=None):
        values = {}
        if self.parameters is not None:
            values.update(self.parameters.numeric())
        if extra:
            values.update({sp.sympify(k): v for k, v in extra.items()})
        return values

    def numeric(self, expr, extra=None):
        """Evaluate ``expr`` at the model's parameter point.

        Uses :meth:`~feynlag.parameters.ParameterSet.numeric`, which already
        resolves internal parameters in dependency order keyed by symbol — no
        parameter evaluation is re-implemented here.

        Args:
            expr: a symbolic width, or a dict of them.
            extra: additional ``{symbol: value}`` substitutions applied last
                (masses, VEVs, anything outside the ``ParameterSet``).
        """
        if isinstance(expr, dict):
            return {k: self.numeric(v, extra=extra) for k, v in expr.items()}
        values = self._substitutions(extra)
        return complex(sp.sympify(expr).subs(values).evalf())

    def numeric_partial_widths(self, parent, extra=None):
        """``{(child1, child2): Γ}`` as **floats**, with closed channels at 0.

        This is not just ``numeric(partial_widths(...))``.  With symbolic
        masses :func:`~feynlag.pheno.kinematics.is_allowed` cannot decide
        whether a channel is open, so it is kept; once numbers are substituted
        a closed channel has ``λ < 0`` and ``√λ`` turns **imaginary**, which
        would silently poison the total width and every branching ratio.  Here
        the threshold is re-tested against the substituted masses and closed
        channels are set to exactly zero.
        """
        values = self._substitutions(extra)

        def as_number(expr):
            return complex(sp.sympify(expr).subs(values).evalf())

        M = as_number(self.mass_of(parent))
        out = {}
        for channel in self.channels(parent):
            m1, m2 = (as_number(self.mass_of(c)) for c in channel.children)
            if M.real < (m1.real + m2.real):
                width = 0.0            # below threshold at this parameter point
            else:
                width = as_number(channel.width).real
            out[channel.children] = out.get(channel.children, 0.0) + width
        return out

    def numeric_total_width(self, parent, extra=None):
        """``Γ_tot`` as a float, counting only channels open at this point."""
        return sum(self.numeric_partial_widths(parent, extra=extra).values())

    def numeric_branching_ratios(self, parent, extra=None):
        """``{(child1, child2): BR}`` as floats, closed channels at 0."""
        widths = self.numeric_partial_widths(parent, extra=extra)
        total = sum(widths.values())
        if total == 0:
            return {k: 0.0 for k in widths}
        return {k: w / total for k, w in widths.items()}
