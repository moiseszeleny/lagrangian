"""Gauge groups with explicit generator matrices.

- :class:`U1`: representation label = the charge (a Rational/number).
- :class:`SU2`: labels 1 (singlet), 2 (fundamental, ``σ/2``), 3 (adjoint).
- :class:`SU3`: labels 1, 3 (fundamental, ``λ/2`` Gell-Mann), 8 (adjoint).

Adjoint generators are built from the structure constants:
``(T^a)_{bc} = -i f^{abc}``.
"""

import sympy as sp

from .base import GaugeGroup

__all__ = ["U1", "SU2", "SU3"]


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


class SU2(GaugeGroup):
    """SU(2) with representations 1 (singlet), 2 (fundamental), 3 (adjoint)."""

    n_generators = 3

    #: structure constants f^{abc} = ε_{abc}
    @staticmethod
    def structure_constant(a, b, c):
        return sp.LeviCivita(a + 1, b + 1, c + 1)

    def rep_dim(self, rep):
        rep = int(rep)
        if rep not in (1, 2, 3):
            raise ValueError(f"SU2 representation must be 1, 2 or 3, got {rep}")
        return rep

    def generators(self, rep):
        rep = int(rep)
        if rep == 1:
            return [sp.zeros(1, 1)] * 3
        if rep == 2:
            return [s / 2 for s in _pauli()]
        if rep == 3:
            # (T^a)_{bc} = -i f^{abc} = -i eps_{abc}
            return [sp.Matrix(3, 3, lambda b, c:
                              -sp.I * self.structure_constant(a, b, c))
                    for a in range(3)]
        raise ValueError(f"SU2 representation must be 1, 2 or 3, got {rep}")

    def pauli_dot(self, vector_symbols):
        """``Σ_a V^a σ^a`` — the ``WsigmaL`` pattern for covariant derivatives."""
        sigma = _pauli()
        return sum((v * s for v, s in zip(vector_symbols, sigma)),
                   sp.zeros(2, 2))


def _gell_mann():
    l1 = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    l2 = sp.Matrix([[0, -sp.I, 0], [sp.I, 0, 0], [0, 0, 0]])
    l3 = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    l4 = sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    l5 = sp.Matrix([[0, 0, -sp.I], [0, 0, 0], [sp.I, 0, 0]])
    l6 = sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    l7 = sp.Matrix([[0, 0, 0], [0, 0, -sp.I], [0, sp.I, 0]])
    l8 = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sp.sqrt(3)
    return [l1, l2, l3, l4, l5, l6, l7, l8]


class SU3(GaugeGroup):
    """SU(3) with representations 1, 3 (fundamental) and 8 (adjoint).

    v1 scope: representation bookkeeping and invariance checking (QCD vertex
    dynamics deferred).
    """

    n_generators = 8

    def rep_dim(self, rep):
        rep = int(rep)
        if rep not in (1, 3, 8):
            raise ValueError(f"SU3 representation must be 1, 3 or 8, got {rep}")
        return rep

    def generators(self, rep):
        rep = int(rep)
        if rep == 1:
            return [sp.zeros(1, 1)] * 8
        if rep == 3:
            return [l / 2 for l in _gell_mann()]
        if rep == 8:
            # (T^a)_{bc} = -i f^{abc} with f from [T^a, T^b] = i f^{abc} T^c,
            # computed from the fundamental: f^{abc} = -2i Tr([T^a,T^b] T^c).
            T = [l / 2 for l in _gell_mann()]
            f = {}
            for a in range(8):
                for b in range(8):
                    for c in range(8):
                        val = sp.simplify(
                            -2 * sp.I * sp.trace((T[a] * T[b] - T[b] * T[a]) * T[c]))
                        if val != 0:
                            f[(a, b, c)] = val
            return [sp.Matrix(8, 8, lambda b, c: -sp.I * f.get((a, b, c), 0))
                    for a in range(8)]
        raise ValueError(f"SU3 representation must be 1, 3 or 8, got {rep}")
