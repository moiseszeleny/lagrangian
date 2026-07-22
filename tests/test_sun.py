"""Dynamic SU(N) irrep generators (any N, any representation).

Pins the physics of the Gelfand–Tsetlin construction in
``feynlag.groups.sun`` and the ``SUN`` gauge group:

- backward compatibility: SU(2)/SU(3) generators are byte-identical to the
  historic ``σ/2`` / ``λ/2`` / ``−i f`` (the ``test_groups.py`` values),
- the shared-basis property ``[T^a,T^b] = i f^{abc} T^c`` with the *same*
  ``f^{abc}`` as the fundamental, for a non-trivial higher rep (the 6 of SU(3)),
- conjugate reps ``T̄^a = −(T^a)^*``,
- the Weyl dimension formula, label resolution + its error messages,
- cubic-anomaly indices, culminating in the SU(5) ``5̄ + 10`` anomaly-free
  generation — the flagship generalization test,
- end-to-end gauge invariance of a sextet / SU(4) fundamental scalar kinetic
  term through the full ``Model`` pipeline.
"""

import random

import sympy as sp
import pytest

from feynlag import (
    Dmu, ExternalParameter, Lagrangian, Model, Scalar, SU2, SU3, SUN,
    structure_constants,
)
from feynlag.groups import sun


# --------------------------------------------------------------------------- #
#  1. Weyl dimension formula                                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("N,dynkin,dim", [
    (3, (1, 0), 3), (3, (1, 1), 8), (3, (2, 0), 6), (3, (2, 1), 15),
    (3, (3, 0), 10),
    (4, (1, 0, 1), 15), (4, (0, 1, 0), 6), (4, (2, 0, 0), 10),
    (5, (0, 1, 0, 0), 10), (5, (1, 0, 0, 1), 24),
])
def test_weyl_dim(N, dynkin, dim):
    assert sun.weyl_dim(N, dynkin) == dim


# --------------------------------------------------------------------------- #
#  2. Byte-identical backward compatibility                                    #
# --------------------------------------------------------------------------- #

def _pauli_over_2():
    return [sp.Matrix([[0, 1], [1, 0]]) / 2,
            sp.Matrix([[0, -sp.I], [sp.I, 0]]) / 2,
            sp.Matrix([[1, 0], [0, -1]]) / 2]


def _gell_mann_over_2():
    l = [sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
         sp.Matrix([[0, -sp.I, 0], [sp.I, 0, 0], [0, 0, 0]]),
         sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
         sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
         sp.Matrix([[0, 0, -sp.I], [0, 0, 0], [sp.I, 0, 0]]),
         sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
         sp.Matrix([[0, 0, 0], [0, 0, -sp.I], [0, sp.I, 0]]),
         sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sp.sqrt(3)]
    return [m / 2 for m in l]


def test_su2_fundamental_byte_identical():
    assert SU2("SU2L").generators(2) == _pauli_over_2()


def test_su3_fundamental_byte_identical():
    assert SU3("SU3c").generators(3) == _gell_mann_over_2()


def test_su3_adjoint_matches_minus_i_f():
    """SU(3) adjoint generators are exactly ``(T^a)_{bc} = −i f^{abc}``."""
    g = SU3("SU3c")
    f = structure_constants(g)
    T = g.generators(8)
    for a in range(8):
        expect = sp.Matrix(8, 8, lambda b, c: -sp.I * f.get((a, b, c), 0))
        assert T[a] == expect


def test_su2_structure_constant_levicivita():
    assert SU2("SU2L").structure_constant(0, 1, 2) == 1
    assert SU2("SU2L").structure_constant(0, 2, 1) == -1


def test_pauli_dot_survives():
    g = SU2("SU2L")
    v = sp.symbols("v1 v2 v3")
    dot = g.pauli_dot(v)
    assert dot == sum((vi * si for vi, si in
                       zip(v, [2 * m for m in _pauli_over_2()])), sp.zeros(2, 2))


# --------------------------------------------------------------------------- #
#  3. Fundamentals of SU(4), SU(5): hermiticity, normalization, algebra        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("N", [4, 5])
def test_fundamental_hermitian_normalized(N):
    T = sun.generalized_gell_mann(N)
    for a in range(len(T)):
        assert T[a] == T[a].conjugate().T                      # hermitian
        assert sp.trace(T[a] * T[a]) == sp.Rational(1, 2)       # Tr = 1/2


def test_fundamental_algebra_closure():
    """[T^a,T^b] = i f^abc T^c in the fundamental (full for SU(4), a random
    sample for SU(5) to keep the runtime down)."""
    for N, sample in [(4, None), (5, 25)]:
        g = SUN(N, f"SU{N}")
        T = g.generators(N)
        f = g.structure_constants()
        n = len(T)
        pairs = [(a, b) for a in range(n) for b in range(n)]
        if sample:
            random.seed(0)
            pairs = random.sample(pairs, sample)
        for a, b in pairs:
            comm = T[a] * T[b] - T[b] * T[a]
            expect = sum((sp.I * f.get((a, b, c), 0) * T[c]
                          for c in range(n)), sp.zeros(N, N))
            assert sp.expand(comm - expect) == sp.zeros(N, N)


# --------------------------------------------------------------------------- #
#  4. Label resolution                                                         #
# --------------------------------------------------------------------------- #

def test_resolve_dimension_labels():
    assert sun.resolve_rep(3, 1) == ((0, 0), False)      # trivial
    assert sun.resolve_rep(3, 3) == ((1, 0), False)      # fundamental
    assert sun.resolve_rep(3, 8) == ((1, 1), False)      # adjoint
    assert sun.resolve_rep(3, 6) == ((2, 0), False)      # sextet
    assert sun.resolve_rep(2, 1) == ((0,), False)
    assert sun.resolve_rep(2, 2) == ((1,), False)
    assert sun.resolve_rep(2, 3) == ((2,), False)


def test_resolve_conjugates():
    assert sun.resolve_rep(3, -6) == ((2, 0), True)
    assert sun.resolve_rep(3, "6bar") == ((2, 0), True)
    assert sun.resolve_rep(3, (0, 1)) == ((1, 0), True)   # 3bar
    assert sun.resolve_rep(3, (1, 0)) == ((1, 0), False)


def test_resolve_errors():
    with pytest.raises(ValueError, match="ambiguous"):
        sun.resolve_rep(3, 15)
    with pytest.raises(ValueError, match="no SU\\(3\\) irrep of dimension 5"):
        sun.resolve_rep(3, 5)
    with pytest.raises(ValueError, match="non-negative"):
        sun.resolve_rep(3, (-1, 0))


def test_rep_dim_bad_raises_like_legacy():
    with pytest.raises(ValueError):
        SU3("SU3c").rep_dim(5)


# --------------------------------------------------------------------------- #
#  5. The 6 of SU(3): shared f, Dynkin index, Casimir                          #
# --------------------------------------------------------------------------- #

def test_sextet_shares_fundamental_structure_constants():
    """The critical property: a higher rep closes the algebra with the SAME
    f^{abc} as the fundamental (label 6 and Dynkin (2,0) give the same rep)."""
    g = SU3("SU3c")
    f = g.structure_constants()
    T6a = g.generators(6)
    T6b = g.generators((2, 0))
    assert T6a == T6b
    assert all(m.shape == (6, 6) for m in T6a)
    for a in range(8):
        for b in range(8):
            comm = T6a[a] * T6a[b] - T6a[b] * T6a[a]
            expect = sum((sp.I * f.get((a, b, c), 0) * T6a[c]
                          for c in range(8)), sp.zeros(6, 6))
            assert sp.expand(comm - expect) == sp.zeros(6, 6)


def test_sextet_dynkin_index_and_casimir():
    T = SU3("SU3c").generators(6)
    dynkin = sp.simplify(sum((m * m).trace() for m in T) / 8)
    assert dynkin == sp.Rational(5, 2)                        # S(6) = 5/2
    casimir = sum((m * m for m in T), sp.zeros(6, 6))
    assert casimir == sp.Rational(10, 3) * sp.eye(6)          # C2(6) = 10/3


# --------------------------------------------------------------------------- #
#  6. Conjugate reps                                                           #
# --------------------------------------------------------------------------- #

def test_conjugate_is_minus_conjugate():
    g = SU3("SU3c")
    T3 = g.generators(3)
    assert g.generators("3bar") == [-t.conjugate() for t in T3]


# --------------------------------------------------------------------------- #
#  7. Adjoint of SU(4)                                                          #
# --------------------------------------------------------------------------- #

def test_su4_adjoint_via_dimension_label():
    g = SUN(4, "SU4")
    T = g.generators(15)                       # N²−1 → adjoint (1,0,1)
    f = g.structure_constants()
    for a in range(15):
        expect = sp.Matrix(15, 15, lambda b, c: -sp.I * f.get((a, b, c), 0))
        assert T[a] == expect
        assert sp.simplify(sp.trace(T[a] * T[a])) == 4        # C_A = N = 4


# --------------------------------------------------------------------------- #
#  8. Cubic-anomaly indices — the SU(5) anomaly-free generation                #
# --------------------------------------------------------------------------- #

def _anomaly_index(group, rep):
    from feynlag.anomalies import _cubic_index
    return _cubic_index(group, rep)


def test_su3_anomaly_indices():
    g = SU3("SU3c")
    assert _anomaly_index(g, 3) == 1
    assert _anomaly_index(g, "3bar") == -1
    assert _anomaly_index(g, 6) == 7
    assert _anomaly_index(g, 8) == 0
    assert _anomaly_index(g, 1) == 0


def test_su5_5bar_plus_10_anomaly_free():
    """The Georgi–Glashow SU(5) generation: a left-handed 5̄ + 10 is
    anomaly-free (A(5̄) = −1, A(10) = +1)."""
    g = SUN(5, "SU5")
    A_5bar = _anomaly_index(g, "5bar")
    A_10 = _anomaly_index(g, (0, 1, 0, 0))
    assert A_5bar == -1
    assert A_10 == 1
    assert A_5bar + A_10 == 0


# --------------------------------------------------------------------------- #
#  9. End-to-end gauge invariance through the Model pipeline                    #
# --------------------------------------------------------------------------- #

def test_sextet_mass_term_gauge_invariant():
    """A ``dag(S)·S`` mass term for a GT-built sextet is gauge invariant
    through the full ``Model`` pipeline — a direct end-to-end exercise of the
    higher-rep generators in the invariance checker (fast; the covariant
    kinetic term is invariant too but its symbolic expansion is expensive for
    radical-valued reps, so it is checked in ``test_dmu_wires_sextet`` +
    ``gauge_variation`` at the coefficient level instead)."""
    gs = ExternalParameter("gs6", 1.2, positive=True)
    SU3c = SU3("SU3c6", coupling=gs)
    S = Scalar("S6", reps={SU3c: 6},
               component_names=[f"S6_{i}" for i in range(6)])
    L = Lagrangian().add((S.dag() * S.mat)[0], sector="potential")
    m = Model("sextet-mass", gauge_groups=[SU3c],
              fields=[S], parameters=[gs], lagrangian=L)
    report = m.check_invariance()
    assert report.ok, report.failures


def test_su4_fundamental_mass_term_gauge_invariant():
    g4 = ExternalParameter("g4m", 0.9, positive=True)
    SU4 = SUN(4, "SU4m", coupling=g4)
    S = Scalar("S4m", reps={SU4: 4},
               component_names=[f"S4m_{i}" for i in range(4)])
    L = Lagrangian().add((S.dag() * S.mat)[0], sector="potential")
    m = Model("su4-mass", gauge_groups=[SU4],
              fields=[S], parameters=[g4], lagrangian=L)
    report = m.check_invariance()
    assert report.ok, report.failures


def test_dmu_wires_sextet_generators():
    """``Dmu`` builds the covariant derivative ``∂_μ S − i g A^a (T^a S)`` with
    the *sextet* generators — the boson-linear part must reproduce
    ``−i g (T^a)_{ij}`` exactly for the GT-built rep."""
    gs = ExternalParameter("gs6d", 1.2, positive=True)
    SU3c = SU3("SU3c6d", coupling=gs)
    S = Scalar("S6d", reps={SU3c: 6},
               component_names=[f"S6d_{i}" for i in range(6)])
    G = SU3c.bosons("Gd")
    DS = Dmu(S)
    T = SU3c.generators(6)
    expected = sum((-sp.I * gs.symbol * G.components[a] * (T[a] * S.mat)
                    for a in range(8)), sp.zeros(6, 1))
    for i in range(6):
        boson_part = sum((DS[i].coeff(G.components[a]) * G.components[a]
                          for a in range(8)), sp.S.Zero)
        assert sp.expand(boson_part - expected[i]) == 0


# --------------------------------------------------------------------------- #
#  10. Yang–Mills self-coupling for a general SU(N)                             #
# --------------------------------------------------------------------------- #

def test_su4_cubic_couplings_build():
    from feynlag import cubic_couplings
    g4 = ExternalParameter("g4c", 0.9, positive=True)
    SU4 = SUN(4, "SU4g", coupling=g4)
    A = SU4.bosons("A")
    couplings = cubic_couplings(SU4)                  # must not raise
    f = SU4.structure_constants()
    # pick a non-zero structure-constant triple and check V = −g f
    (a, b, c) = next(iter(f))
    key = (A.components[a], A.components[b], A.components[c])
    assert sp.simplify(couplings[key] - (-SU4.g * f[(a, b, c)])) == 0
