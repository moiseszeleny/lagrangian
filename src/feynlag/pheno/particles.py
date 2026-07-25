"""Physical (post-EWSB) Dirac particles for the decay calculator.

A Dirac fermion is *not* a declarable input field in feynlag — the SM is
chiral, so `b_L` (an SU(2) doublet, $Y=+1/6$) and `b_R` (a singlet, $Y=-1/3$)
carry different reps and no 4-component field can hold both invariantly;
:class:`~feynlag.fields.DiracFermion` raises on construction for exactly this
reason.  **After** electroweak symmetry breaking, however, the physical `b` is
one Dirac particle built from those two Weyl legs, and a decay calculation
needs to treat it as one.

:class:`DiracParticle` is that post-EWSB view.  It bundles the two Weyl leg
bases, the particle/antiparticle symbols, the mass, and the colour
multiplicity $N_c$ into a single object, so the three things the calculator
otherwise tracks in separate, separately-keyed dicts
(``particle_map``/``masses``/``color_factors``) travel together and cannot fall
out of sync.  The bar legs are derived automatically from the
:func:`~feynlag.fields.bar_partner` registry — they are not passed in.
"""

from dataclasses import dataclass, field

import sympy as sp

from ..fields import bar_partner

__all__ = ["DiracParticle", "ExternalState", "expand_particles"]


@dataclass(frozen=True)
class DiracParticle:
    """One physical Dirac fermion, assembled from its two Weyl legs.

    Args:
        particle: the particle symbol (a ``str`` is promoted to a
            :class:`~sympy.core.symbol.Symbol`), e.g. ``'b'``.
        left: the left-handed Weyl leg base — the ``IndexedBase`` returned by
            ``fermion.components[k]`` (e.g. ``QL.components[1]`` for the
            down-type member of a quark doublet).
        right: the right-handed Weyl leg base (e.g. ``bR.components[0]``).
        mass: the physical mass (symbol or number).
        color: colour multiplicity $N_c$ — ``3`` for a quark, ``1`` (default)
            for a lepton.  Applied **once** per channel by the calculator, so
            no per-leg double counting is possible.
        antiparticle: the antiparticle symbol; defaults to ``Symbol(f"{name}bar")``.

    The bar legs (``bar_left``/``bar_right``) are looked up from the
    :func:`~feynlag.fields.bar_partner` registry, not supplied.
    """

    particle: sp.Symbol
    left: sp.IndexedBase
    right: sp.IndexedBase
    mass: sp.Expr
    color: int = 1
    antiparticle: sp.Symbol = None

    def __post_init__(self):
        # frozen dataclass: normalise via object.__setattr__
        object.__setattr__(self, "particle", sp.sympify(self.particle))
        object.__setattr__(self, "mass", sp.sympify(self.mass))
        if self.antiparticle is None:
            object.__setattr__(self, "antiparticle",
                               sp.Symbol(f"{self.particle}bar"))
        else:
            object.__setattr__(self, "antiparticle",
                               sp.sympify(self.antiparticle))

    @property
    def bar_left(self):
        """The bar leg paired with ``left`` (from :func:`bar_partner`)."""
        return bar_partner(self.left)

    @property
    def bar_right(self):
        """The bar leg paired with ``right`` (from :func:`bar_partner`)."""
        return bar_partner(self.right)

    def particle_map(self):
        """``{weyl_leg_base: physical_symbol}`` for this particle.

        The two field legs map to the particle, the two bar legs to the
        antiparticle.  Keyed by ``IndexedBase`` so every flavour index is
        covered (``resolve_leg`` falls back to a leg's ``.base``).
        """
        return {
            self.left: self.particle,
            self.right: self.particle,
            self.bar_left: self.antiparticle,
            self.bar_right: self.antiparticle,
        }

    def masses(self):
        """``{particle: mass, antiparticle: mass}``."""
        return {self.particle: self.mass, self.antiparticle: self.mass}

    def colors(self):
        """``{particle: N_c, antiparticle: N_c}``."""
        nc = sp.Integer(self.color)
        return {self.particle: nc, self.antiparticle: nc}


@dataclass(frozen=True)
class ExternalState:
    """One external particle of a 2→2 scattering process.

    :class:`DiracParticle` bundles what a *decaying* fermion needs (mass,
    colour, its two Weyl legs).  A scattering initial/final state additionally
    needs its **spin**, because
    :func:`~feynlag.pheno.scattering.average_factor` must average over the
    initial-state spin/colour degrees of freedom — the anti-pattern this
    replaces is :func:`~feynlag.pheno.amplitudes.ffv_squared`'s hardcoded
    ``/3``, an initial-state spin average masquerading as part of the Lorentz
    algebra, which made that function unreusable for an *internal* vector.
    Here the averaging factor is declared data, applied exactly once by
    :func:`~feynlag.pheno.scattering.cross_section`.

    Args:
        name: the particle symbol (a ``str`` is promoted to a
            :class:`~sympy.core.symbol.Symbol`).
        mass: the physical mass.
        spin: ``0`` (scalar), ``Rational(1,2)`` (fermion) or ``1`` (vector).
        color: colour multiplicity, as in :class:`DiracParticle`.
    """

    name: sp.Symbol
    mass: sp.Expr
    spin: sp.Rational
    color: int = 1

    def __post_init__(self):
        object.__setattr__(self, "name", sp.sympify(self.name))
        object.__setattr__(self, "mass", sp.sympify(self.mass))
        object.__setattr__(self, "spin", sp.sympify(self.spin))

    def spin_states(self):
        """Number of physical spin states: ``1`` (scalar), ``2`` (fermion),
        ``2`` (massless vector) or ``3`` (massive vector)."""
        if self.spin == 0:
            return 1
        if self.spin == sp.Rational(1, 2):
            return 2
        if self.spin == 1:
            return 1 if self.mass == 0 else 3
        raise NotImplementedError(f"ExternalState: unsupported spin {self.spin}")

    def dof(self):
        """``spin_states() × color`` — the full averaging denominator unit."""
        return self.spin_states() * self.color

    @classmethod
    def from_dirac_particle(cls, dp, spin=sp.Rational(1, 2)):
        """Build from a :class:`DiracParticle`, so mass and ``N_c`` cannot
        fall out of sync between the two descriptions."""
        return cls(name=dp.particle, mass=dp.mass, spin=spin, color=dp.color)


def expand_particles(particles):
    """Merge a list of :class:`DiracParticle` into the calculator's dicts.

    Returns:
        ``(particle_map, masses, color_registry)`` — the three mappings the
        :class:`~feynlag.pheno.calculator.DecayCalculator` consumes.

    Raises:
        ValueError: two particles claim the same Weyl leg or the same
            particle symbol with a conflicting mass — a modelling mistake that
            would otherwise merge silently.
    """
    particle_map, masses, colors = {}, {}, {}
    for dp in particles:
        for leg, sym in dp.particle_map().items():
            if leg in particle_map and particle_map[leg] != sym:
                raise ValueError(
                    f"Weyl leg {leg!r} is claimed by two DiracParticles "
                    f"({particle_map[leg]} and {sym})")
            particle_map[leg] = sym
        for sym, m in dp.masses().items():
            if sym in masses and masses[sym] != m:
                raise ValueError(
                    f"particle {sym!r} declared with two masses "
                    f"({masses[sym]} and {m})")
            masses[sym] = m
        colors.update(dp.colors())
    return particle_map, masses, colors
