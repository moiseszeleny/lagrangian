"""Symmetry group base classes.

Design (see plan): group theory by **explicit component transformation** with
generator matrices — no abstract index gymnastics.  A gauge group knows the
generator matrices of each representation label; a discrete group knows the
finite generator matrices of each irrep and which field multiplets it acts on.
"""

__all__ = ["SymmetryGroup", "GaugeGroup"]


class SymmetryGroup:
    """Base class for all symmetry groups."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"

    def __hash__(self):
        return hash((type(self).__name__, self.name))

    def __eq__(self, other):
        return self is other


class GaugeGroup(SymmetryGroup):
    """Continuous (gauged) symmetry group.

    Args:
        name: label, e.g. ``'SU2L'``.
        coupling: the gauge coupling — a :class:`~feynlag.parameters.Parameter`
            or a SymPy symbol.
    """

    abelian = False

    def __init__(self, name, coupling=None):
        super().__init__(name)
        self.coupling = coupling
        self._gauge_bosons = None

    @property
    def g(self):
        """The coupling as a plain SymPy symbol."""
        return getattr(self.coupling, "symbol", self.coupling)

    # --- interface to be provided by concrete groups -------------------------

    def rep_dim(self, rep):
        """Dimension of the representation labelled ``rep``."""
        raise NotImplementedError

    def generators(self, rep):
        """List of generator matrices ``T^a`` for representation ``rep``.

        Conventions: hermitian generators normalized as in the PDG
        (``σ/2`` for the SU(2) fundamental, ``λ/2`` for SU(3)); for a U(1)
        the single "generator" is the charge times the 1×1 identity.
        """
        raise NotImplementedError

    @property
    def n_generators(self):
        """Number of generators (= number of gauge bosons)."""
        raise NotImplementedError
