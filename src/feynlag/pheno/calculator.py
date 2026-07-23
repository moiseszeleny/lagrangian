"""Partial widths, total widths and branching ratios from a feynlag model."""

import warnings
from dataclasses import dataclass

import sympy as sp

from .amplitudes import amplitude_squared
from .kinematics import TwoBodyKinematics, is_allowed
from .particles import expand_particles
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
        particles: list of :class:`~feynlag.pheno.particles.DiracParticle` —
            the recommended way to declare physical fermions. Each bundles its
            two Weyl legs, mass and colour $N_c$, so the ``particle_map`` /
            ``masses`` / ``color_factors`` triple cannot fall out of sync.
            Colour is applied **once** per channel (no per-leg double count),
            and any extracted fermion vertex claimed by no declared particle is
            surfaced in :attr:`unmatched_channels` (with a warning) instead of
            silently vanishing. Merges with ``masses`` (bosons stay there) and
            with an explicit ``particle_map``/``color_factors`` if also given.
    """

    def __init__(self, model, masses, boson_fields=(), fermion_sectors=(),
                 conjugate_map=None, color_factors=None, parameters=None,
                 simplifier=None, particle_map=None, particles=()):
        self.model = model
        self.masses = dict(masses)
        self.boson_fields = list(boson_fields)
        self.fermion_sectors = tuple(fermion_sectors)
        self.conjugate_map = conjugate_map
        self.color_factors = dict(color_factors or {})
        self.parameters = parameters
        self.simplifier = simplifier
        self.particle_map = dict(particle_map or {})
        self.particles = list(particles)
        #: {particle_symbol: N_c} from declared DiracParticles (colour once).
        self._particle_color = {}
        if self.particles:
            pmap, pmasses, pcolors = expand_particles(self.particles)
            # DiracParticle-declared entries fill in without clobbering an
            # explicit override the caller also passed.
            for leg, sym in pmap.items():
                self.particle_map.setdefault(leg, sym)
            for sym, m in pmasses.items():
                self.masses.setdefault(sym, m)
            self._particle_color = pcolors
        self._vertices = None
        self._unmatched = None

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

    def _channel_color(self, children):
        """Colour multiplicity of a two-body channel.

        When both daughters belong to the same declared
        :class:`~feynlag.pheno.particles.DiracParticle` (an ``f f̄`` pair),
        colour is that particle's $N_c$ applied **once** — never squared. This
        is what makes the per-leg over-count (the ``81×`` trap) unrepresentable
        through the ``particles=`` path.  Otherwise fall back to the legacy
        per-leg product of ``color_factors`` (backward compatible).
        """
        if self._particle_color:
            nc = {self._particle_color.get(c) for c in children}
            nc.discard(None)
            if len(nc) == 1 and all(c in self._particle_color for c in children):
                return nc.pop()
        color = sp.S.One
        for leg in children:
            color *= self._color_of(leg)
        return color

    # --------------------------------------------------------------- widths

    def channels(self, parent):
        """Every open two-body channel of ``parent``.

        A vertex contributes when one of its legs is ``parent``; the other two
        are the daughters. A daughter with an undeclared mass is skipped — but
        recorded in :attr:`unmatched_channels` when ``particles=`` is in use,
        so the skip is visible rather than silent.
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
                self._record_unmatched(vertex, parent, remaining)
                continue
            symmetry = (sp.Rational(1, 2) if remaining[0] == remaining[1]
                        else sp.S.One)
            color = self._channel_color(remaining)
            width = partial_width(vertex, M, m1, m2, color_factor=color,
                                  symmetry_factor=symmetry)
            if width == 0:
                continue
            out.append(DecayChannel(parent=parent, children=tuple(remaining),
                                    width=width, vertex=vertex,
                                    color_factor=color,
                                    symmetry_factor=symmetry))
        return out

    def _record_unmatched(self, vertex, parent, children):
        # Only fermion channels count: a bosonic leg with no declared mass
        # (e.g. an unphysical-gauge Goldstone) is a legitimate skip, not the
        # "forgot to declare a fermion" trap this surfaces.
        if vertex.vertex_type not in ("FFS", "FFV"):
            return
        if self._unmatched is None:
            self._unmatched = []
        entry = (parent, tuple(children), vertex.vertex_type)
        if entry not in self._unmatched:
            self._unmatched.append(entry)

    @property
    def unmatched_channels(self):
        """Fermion channels dropped because a daughter had no declared mass.

        A list of ``(parent, (child1, child2), vertex_type)`` populated as a
        side effect of :meth:`channels`.  When ``particles=`` was supplied and
        this is non-empty, a warning is emitted once — turning the otherwise
        silent skip (a fermion the model produces but you forgot to declare)
        into a visible one.  Call :meth:`channels` (or any width method) first;
        it is empty until then.
        """
        matched = self._unmatched or []
        if matched and self.particles:
            warnings.warn(
                f"{len(matched)} fermion channel(s) were dropped because a "
                f"daughter has no declared DiracParticle/mass: "
                f"{[ (str(p), tuple(map(str, c)), t) for p, c, t in matched] }. "
                f"Declare them (or ignore if intentional).",
                stacklevel=2)
        return list(matched)

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

    # ----------------------------------------------------------- off-shell

    def offshell_vv_width(self, parent, vector, m_parent, m_V, width_V,
                          channels, identical=False, coupling=None,
                          backend="auto"):
        """Off-shell ``parent → V (V^* → f f̄')`` width (Tier 2).

        The on-shell ``VVS`` channel is zero below threshold (``2m_V >
        m_parent``); this returns the physical $1\\to3$ width through one
        on-shell and one off-shell ``V``, via
        :func:`~feynlag.pheno.offshell.scalar_offshell_vv_width`.

        Args:
            parent, vector: the scalar and vector physical symbols (used to look
                up the ``SVV`` coupling from the extracted vertices).
            m_parent, m_V, width_V: numeric masses and the vector total width
                ``Γ_V`` (the Breit–Wigner width — itself a Tier-1 output).
            channels: ``[(g_L, g_R, multiplicity)]`` per ``V^*→f f̄'`` channel.
            identical: ``False`` for distinct ``V₁V₂`` (``WW``, overall factor
                2), ``True`` for identical (``ZZ``).
            coupling: the ``SVV`` coupling value; if ``None`` it is read from the
                extracted ``VVS`` vertex ``(parent, vector, vector)``.
            backend: numeric integration backend.

        Returns:
            the off-shell width (float).
        """
        from .offshell import scalar_offshell_vv_width
        if coupling is None:
            coupling = self._svv_coupling(parent, vector)
        return scalar_offshell_vv_width(
            float(m_parent), float(m_V), float(width_V),
            abs(complex(coupling)), channels, identical=identical,
            backend=backend)

    def _svv_coupling(self, parent, vector):
        """The ``SVV`` coupling magnitude (``i`` stripped) from the VVS vertex
        containing ``parent`` and ``vector`` — e.g. ``(h, W⁺, W⁻)`` for
        ``h→WW*`` or ``(h, Z, Z)`` for ``h→ZZ*``."""
        for v in self.vertices():
            if (v.vertex_type == "VVS" and parent in v.particles
                    and vector in v.particles):
                c = sp.simplify(v.coupling / sp.I)      # strip the Feynman i
                sub = self._substitutions()
                return complex(sp.sympify(c).subs(sub).evalf())
        raise KeyError(f"no VVS vertex containing ({parent}, {vector})")
