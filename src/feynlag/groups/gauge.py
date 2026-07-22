"""Gauge groups with explicit generator matrices.

- :class:`U1`: representation label = the charge (a Rational/number).
- :class:`SUN`: any SU(N); representations are built dynamically in the
  Gelfand–Tsetlin basis (see :mod:`feynlag.groups.sun`).  Labels may be an
  integer *dimension* (``1`` → trivial, ``N`` → fundamental, ``N²−1`` →
  adjoint, other ints resolved by the Weyl dimension formula), a conjugate
  label (negative int ``−d`` or string ``"dbar"``), or an explicit Dynkin
  tuple ``(a_1, …, a_{N−1})``.  Conjugate reps have ``T̄^a = −(T^a)^*``.
- :class:`SU2`, :class:`SU3`: thin subclasses of :class:`SUN` kept for the
  familiar names and legacy dimension labels (SU2: 1/2/3, SU3: 1/3/8); their
  generators are byte-identical to the historic ``σ/2`` / ``λ/2`` / ``−i f``.

Every representation of a group shares the fundamental's structure constants
``f^{abc}``, so ``[T^a,T^b] = i f^{abc} T^c`` holds across reps automatically —
the property the covariant derivative and Yang–Mills self-couplings rely on.
Adjoint generators are ``(T^a)_{bc} = −i f^{abc}``.
"""

import sympy as sp

from .base import GaugeGroup
from . import sun as _sun

__all__ = ["U1", "SUN", "SU2", "SU3"]


class U1(GaugeGroup):
    """Abelian U(1) group; a field's representation IS its charge."""

    abelian = True
    n_generators = 1

    def rep_dim(self, rep):
        return 1

    def generators(self, rep):
        """One 1×1 generator: the charge itself."""
        return [sp.Matrix([[sp.sympify(rep)]])]


def _pauli():
    return [sp.Matrix([[0, 1], [1, 0]]),
            sp.Matrix([[0, -sp.I], [sp.I, 0]]),
            sp.Matrix([[1, 0], [0, -1]])]


class SUN(GaugeGroup):
    """SU(N) with representations built dynamically for any N and any irrep.

    Args:
        N: the rank+1 (SU(N) has ``N²−1`` generators).
        name: group label, e.g. ``'SU5'``.
        coupling: gauge coupling (a :class:`Parameter` or SymPy symbol).
    """

    def __init__(self, N, name, coupling=None):
        if int(N) < 2:
            raise ValueError(f"SU(N) needs N ≥ 2, got {N}")
        self.N = int(N)
        super().__init__(name, coupling=coupling)

    @property
    def n_generators(self):
        return self.N * self.N - 1

    def rep_dim(self, rep):
        return _sun.rep_dimension(self.N, rep)

    def generators(self, rep):
        return _sun.generators_for_label(self.N, rep)

    def structure_constants(self):
        """``f^{abc}`` as a dict ``{(a, b, c): value}`` of non-zero entries."""
        return _sun.structure_constants_dict(self.N)

    def structure_constant(self, a, b, c):
        return self.structure_constants().get((a, b, c), sp.S.Zero)


class SU2(SUN):
    """SU(2); legacy labels 1 (singlet), 2 (fundamental), 3 (adjoint)."""

    def __init__(self, name, coupling=None):
        super().__init__(2, name, coupling=coupling)

    #: structure constants f^{abc} = ε_{abc} (kept as the exact historic API)
    @staticmethod
    def structure_constant(a, b, c):
        return sp.LeviCivita(a + 1, b + 1, c + 1)

    def pauli_dot(self, vector_symbols):
        """``Σ_a V^a σ^a`` — the ``WsigmaL`` pattern for covariant derivatives."""
        sigma = _pauli()
        return sum((v * s for v, s in zip(vector_symbols, sigma)),
                   sp.zeros(2, 2))


class SU3(SUN):
    """SU(3); legacy labels 1 (singlet), 3 (fundamental), 8 (adjoint)."""

    def __init__(self, name, coupling=None):
        super().__init__(3, name, coupling=coupling)
