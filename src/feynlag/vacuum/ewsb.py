"""Electroweak symmetry breaking: the Vacuum object.

Aggregates the VEV registrations of all scalar fields
(``Scalar.expand_vev``) into the substitution maps the pipeline needs:

- ``shift_map``: weak components → vacuum + fluctuations
  (``φ⁰ → (v + h + i a)/√2``),
- ``fluctuations``: all real fluctuation symbols,
- ``at_vacuum(expr)``: evaluate an expression on the vacuum (shift, then set
  all fluctuations to zero).
"""

import sympy as sp

__all__ = ["Vacuum"]


class Vacuum:
    """The scalar vacuum configuration of a model."""

    def __init__(self, scalars):
        self.scalars = [s for s in scalars
                        if getattr(s, "vev_expansions", None)]
        if not self.scalars:
            raise ValueError("no scalar field has registered VEVs; call "
                             "Scalar.expand_vev first")

    @property
    def shift_map(self):
        """Weak components → vacuum + fluctuation expansion."""
        shift = {}
        for s in self.scalars:
            shift.update(s.shift_map)
        return shift

    @property
    def fluctuations(self):
        out = []
        for s in self.scalars:
            out.extend(s.fluctuations)
        return out

    @property
    def vevs(self):
        """All VEV symbols/expressions, in registration order."""
        out = []
        for s in self.scalars:
            for comp, (vev, re, im) in s.vev_expansions.items():
                out.append(vev)
        return out

    def shift(self, expr):
        """Expand ``expr`` around the vacuum (fluctuations kept)."""
        return sp.expand(expr.xreplace(self.shift_map))

    def at_vacuum(self, expr):
        """Evaluate ``expr`` at the vacuum point (all fluctuations → 0).

        Also sets any remaining non-VEV'd scalar components (e.g. charged
        components) to zero — they have no VEV by charge conservation.
        """
        shifted = self.shift(expr)
        zero_fluct = {f: 0 for f in self.fluctuations}
        for s in self.scalars:
            for comp in s.components:
                if comp not in s.vev_expansions:
                    zero_fluct[comp] = 0
                    zero_fluct[sp.conjugate(comp)] = 0
        return sp.expand(shifted.xreplace(zero_fluct))
