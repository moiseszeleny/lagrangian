"""Dynamic construction of SU(N) irrep generators (any N, any irrep).

The generators of an arbitrary irreducible representation are built from the
Lie algebra itself, in the **Gelfand–Tsetlin (GT) basis** — a highest-weight /
ladder construction that yields an orthonormal weight basis with exact
closed-form ``sqrt(Rational)`` matrix elements (degenerate weight spaces are
automatically orthonormal, so no Gram–Schmidt is ever needed).

Design (see CLAUDE.md "SU(N) / QCD vertex dynamics"):

* An irrep is labelled by **Dynkin labels** ``(a_1, …, a_{N−1})`` (non-negative
  ints).  Integer *dimension* labels keep working (``1`` → trivial, ``N`` →
  fundamental, ``N²−1`` → adjoint, other ints → resolved by the Weyl dimension
  formula), as do conjugate labels (a negative int ``−d`` or a string
  ``"dbar"``, and non-canonical Dynkin tuples), whose generators are
  ``T̄^a = −(T^a)^*``.

* Every irrep of one group is expressed in the **same** generalized-Gell-Mann
  basis (the same fixed linear combinations of the ``E_{ij}`` matrix units as
  the fundamental).  Because a representation ``ρ`` is a Lie-algebra
  homomorphism, ``[T^a_R, T^b_R] = i f^{abc} T^c_R`` then holds with the *same*
  structure constants ``f^{abc}`` for every ``R`` automatically — this is the
  property that ``Dmu`` / ``fermion_gauge_current`` / ``yangmills`` require, and
  it is why the normalization ``Tr(T^a_R T^b_R) = S(R) δ^{ab}`` (Dynkin index
  ``S(R)``, ``S(fund) = 1/2``) is *fixed by the algebra*, not tunable.

All arithmetic is exact SymPy (``sqrt``/``Rational``); results are cached per
``(N, dynkin)`` at module level.
"""

import itertools

import sympy as sp

__all__ = [
    "weyl_dim",
    "resolve_rep",
    "su_n_generators",
    "generators_for_label",
    "rep_dimension",
    "structure_constants_dict",
    "generalized_gell_mann",
]


# --------------------------------------------------------------------------- #
#  Dynkin labels → partition, dimension                                       #
# --------------------------------------------------------------------------- #

def _partition(N, dynkin):
    """Top GT row ``m_k = Σ_{j≥k} a_j`` (length ``N``, last entry 0)."""
    return tuple(sum(dynkin[k:]) for k in range(N - 1)) + (0,)


def weyl_dim(N, dynkin):
    """Dimension of the SU(N) irrep with Dynkin labels ``dynkin`` (Weyl
    formula, exact integer arithmetic — no matrices built)."""
    m = _partition(N, dynkin)
    l = [m[i] + (N - i) for i in range(N)]   # l_i = m_i + N − i (1-indexed i→i+1)
    num = den = 1
    for i in range(N):
        for j in range(i + 1, N):
            num *= l[i] - l[j]
            den *= j - i
    return num // den


# --------------------------------------------------------------------------- #
#  Fundamental: generalized Gell-Mann matrices (explicit matrix units)        #
# --------------------------------------------------------------------------- #

def _matrix_unit(N, i, j):
    """``E_{ij}`` in the fundamental (1 at row ``i``, col ``j``), 0-indexed."""
    M = sp.zeros(N, N)
    M[i, j] = 1
    return M


def _hermitian_basis(N, Efun):
    """Assemble the ordered hermitian generalized-Gell-Mann generators from a
    dict ``Efun[(i, j)]`` of representation matrices of the gl(N) units
    ``E_{ij}`` (0-indexed) and the diagonal ``E_{ii}``.

    Ordering (reproduces σ/2 for N=2, λ/2 for N=3):
    ``for j in 1..N−1: for i in 0..j−1: T_S(i,j), T_A(i,j); then T_D(j)``.
    """
    gens = []
    for j in range(1, N):
        for i in range(j):
            Eij, Eji = Efun[(i, j)], Efun[(j, i)]
            gens.append((Eij + Eji) / 2)                 # T_S
            gens.append(-sp.I * (Eij - Eji) / 2)         # T_A
        # T_D(j): diag(1,…,1 [j times], −j, 0,…) · ½·sqrt(2/(j(j+1)))
        diag = sp.zeros(*Efun[(0, 0)].shape)
        for k in range(j):
            diag += Efun[(k, k)]
        diag -= j * Efun[(j, j)]
        gens.append(sp.sqrt(sp.Rational(2, j * (j + 1))) / 2 * diag)
    return gens


def generalized_gell_mann(N):
    """The ``N²−1`` hermitian fundamental generators of SU(N) (PDG norm
    ``Tr(T^aT^b)=δ^{ab}/2``).  Byte-identical to ``σ/2`` (N=2) and ``λ/2`` (N=3)."""
    Efun = {(i, j): _matrix_unit(N, i, j)
            for i in range(N) for j in range(N)}
    return _hermitian_basis(N, Efun)


# --------------------------------------------------------------------------- #
#  Structure constants (from the fundamental), cached per N                   #
# --------------------------------------------------------------------------- #

_F_CACHE = {}


def structure_constants_dict(N):
    """``f^{abc} = −2i Tr([T^a,T^b] T^c)`` from the fundamental of SU(N).

    Returns a dict ``{(a, b, c): value}`` of the non-zero entries (cached)."""
    if N in _F_CACHE:
        return _F_CACHE[N]
    T = generalized_gell_mann(N)
    n = len(T)
    f = {}
    for a in range(n):
        for b in range(a + 1, n):
            comm = T[a] * T[b] - T[b] * T[a]
            for c in range(n):
                # sparse trace: Σ_{(x,y)∈nz(T[c])} comm[y,x]·T[c][x,y]
                val = sp.S.Zero
                for (x, y), tv in T[c].todok().items():
                    val += comm[y, x] * tv
                val = sp.nsimplify(sp.expand(-2 * sp.I * val))
                if val != 0:
                    f[(a, b, c)] = val
                    f[(b, a, c)] = -val         # antisymmetry in a↔b
    _F_CACHE[N] = f
    return f


def _adjoint_generators(N):
    """Adjoint generators ``(T^a)_{bc} = −i f^{abc}`` (structure-constant
    basis — the basis the gauge-boson components are indexed in)."""
    f = structure_constants_dict(N)
    n = N * N - 1
    gens = [sp.zeros(n, n) for _ in range(n)]
    for (a, b, c), val in f.items():
        gens[a][b, c] = -sp.I * val
    return gens


# --------------------------------------------------------------------------- #
#  Gelfand–Tsetlin construction of a general irrep                            #
# --------------------------------------------------------------------------- #

def _lower_rows(upper):
    """All rows of length ``len(upper)−1`` interlacing ``upper`` (betweenness
    ``upper[i] ≥ v[i] ≥ upper[i+1]``)."""
    k = len(upper)
    if k == 1:
        yield ()
        return
    ranges = [range(upper[i + 1], upper[i] + 1) for i in range(k - 1)]
    yield from itertools.product(*ranges)


def _gt_patterns(top):
    """All Gelfand–Tsetlin patterns with fixed top row ``top`` (a pattern is a
    tuple of rows, top row first)."""
    results = []

    def rec(rows):
        last = rows[-1]
        if len(last) == 1:
            results.append(tuple(rows))
            return
        for low in _lower_rows(last):
            rec(rows + [low])

    rec([tuple(top)])
    return results


def _row(pattern, k, N):
    """The row of ``pattern`` that has ``k`` entries (``m_{·,k}``)."""
    return pattern[N - k]


def _increment(pattern, k, j, N):
    """New pattern with ``m_{j,k}`` (1-indexed ``j``) raised by 1."""
    rows = list(pattern)
    idx = N - k
    row = list(rows[idx])
    row[j - 1] += 1
    rows[idx] = tuple(row)
    return tuple(rows)


def _raise_coeff(pattern, k, j, N):
    """GT matrix element of the simple raiser ``E_{k,k+1}`` moving
    ``m_{j,k} → m_{j,k}+1`` (1-indexed ``j`` in ``1..k``)."""
    rk = _row(pattern, k, N)
    rk1 = _row(pattern, k + 1, N)
    rkm1 = _row(pattern, k - 1, N) if k - 1 >= 1 else ()
    ljk = rk[j - 1] - j                       # l_{j,k}
    num = 1
    for i in range(1, k + 2):                 # Π_{i=1}^{k+1} (l_{i,k+1} − l_{j,k})
        num *= (rk1[i - 1] - i) - ljk
    for i in range(1, k):                     # Π_{i=1}^{k−1} (l_{i,k−1} − l_{j,k} − 1)
        num *= (rkm1[i - 1] - i) - ljk - 1
    den = 1
    for i in range(1, k + 1):                 # Π_{i≠j} (l_{i,k}−l_{j,k})(…−1)
        if i == j:
            continue
        lik = rk[i - 1] - i
        den *= (lik - ljk) * (lik - ljk - 1)
    return sp.sqrt(sp.Rational(-num, den))


def _gt_generators(N, dynkin):
    """Hermitian generators of the SU(N) irrep ``dynkin`` via the GT basis."""
    top = _partition(N, dynkin)
    patterns = _gt_patterns(top)
    # descending lexicographic on the flattened (top-first) pattern
    patterns.sort(key=lambda p: sum(p, ()), reverse=True)
    index = {p: i for i, p in enumerate(patterns)}
    D = len(patterns)
    assert D == weyl_dim(N, dynkin), "GT pattern count ≠ Weyl dimension"

    # simple raisers E_{k,k+1} and lowerers E_{k+1,k}=E_{k,k+1}^†
    Eplus = {}
    for k in range(1, N):
        R = sp.zeros(D, D)
        for col, p in enumerate(patterns):
            for j in range(1, k + 1):
                p2 = _increment(p, k, j, N)
                row = index.get(p2)
                if row is not None:
                    R[row, col] += _raise_coeff(p, k, j, N)
        Eplus[(k, k + 1)] = R

    # E_{ij} matrix units of gl(N): diagonal, simple, then via commutators
    E = {}
    for k in range(1, N):
        E[(k - 1, k)] = Eplus[(k, k + 1)]              # 0-indexed keys
        E[(k, k - 1)] = Eplus[(k, k + 1)].conjugate().T
    for gap in range(2, N):
        for i in range(N - gap):
            j = i + gap
            comm = E[(i, i + 1)] * E[(i + 1, j)] - E[(i + 1, j)] * E[(i, i + 1)]
            E[(i, j)] = comm
            E[(j, i)] = comm.conjugate().T
    # diagonal E_{ii}: eigenvalue w_k − w_{k−1} on each pattern
    for a in range(N):
        k = a + 1
        diag = []
        for p in patterns:
            wk = sum(_row(p, k, N))
            wkm1 = sum(_row(p, k - 1, N)) if k - 1 >= 1 else 0
            diag.append(wk - wkm1)
        E[(a, a)] = sp.diag(*diag)

    # build-time self-check: unitary star structure + su(2) subalgebra
    for k in range(1, N):
        R = Eplus[(k, k + 1)]
        Ldag = R.conjugate().T
        comm = R * Ldag - Ldag * R
        expect = E[(k - 1, k - 1)] - E[(k, k)]
        assert sp.expand(comm - expect) == sp.zeros(D, D), \
            f"GT self-check failed for SU({N}) {dynkin} at simple root {k}"

    return _hermitian_basis(N, E)


# --------------------------------------------------------------------------- #
#  Label resolution and the public generator entry points                     #
# --------------------------------------------------------------------------- #

def _fundamental_dynkin(N):
    return (1,) + (0,) * (N - 2)


def _adjoint_dynkin(N):
    if N == 2:
        return (2,)
    return (1,) + (0,) * (N - 3) + (1,)


def _canonical(dynkin):
    """Canonical member of the conjugation pair ``{λ, reversed(λ)}`` — the
    lexicographically larger.  Returns ``(canonical, conjugated?)``."""
    rev = tuple(reversed(dynkin))
    if tuple(dynkin) >= rev:
        return tuple(dynkin), False
    return rev, True


def _search_dimension(N, d):
    """All canonical Dynkin tuples of dimension ``d`` (one per conjugation
    class)."""
    if (d + 1) ** (N - 1) > 2_000_000:
        raise ValueError(
            f"dimension {d} too large to search for SU({N}); pass an "
            f"explicit Dynkin tuple instead")
    classes = {}
    for dyn in itertools.product(range(d), repeat=N - 1):
        if weyl_dim(N, dyn) == d:
            canon, _ = _canonical(dyn)
            classes[canon] = True
    return list(classes)


def resolve_rep(N, label):
    """Resolve a representation ``label`` to ``(dynkin, conjugated)``.

    Accepts a Dynkin tuple of length ``N−1``, an integer dimension label
    (``1``/``N``/``N²−1``/other; negative → conjugate), or a string ``"dbar"``.
    """
    # --- Dynkin tuple ---
    if isinstance(label, (tuple, list)):
        if len(label) != N - 1:
            raise ValueError(
                f"SU({N}) Dynkin label must have {N - 1} entries, got {label!r}")
        dyn = tuple(int(a) for a in label)
        if any(a < 0 for a in dyn):
            raise ValueError(f"Dynkin labels must be non-negative: {label!r}")
        return _canonical(dyn)

    # --- string "<d>bar" ---
    if isinstance(label, str):
        s = label.strip()
        if s.endswith("bar"):
            dyn, conj = resolve_rep(N, int(s[:-3]))
            return dyn, not conj
        return resolve_rep(N, int(s))

    # --- integer dimension label ---
    d = int(label)
    if d < 0:
        dyn, conj = resolve_rep(N, -d)
        return dyn, not conj
    if d == 1:
        return (0,) * (N - 1), False
    if d == N:
        return _fundamental_dynkin(N), False
    if d == N * N - 1:
        return _adjoint_dynkin(N), False
    candidates = _search_dimension(N, d)
    if not candidates:
        raise ValueError(f"no SU({N}) irrep of dimension {d}")
    if len(candidates) > 1:
        pretty = ", ".join(str(c) for c in sorted(candidates, reverse=True))
        raise ValueError(
            f"SU({N}) dimension {d} is ambiguous ({pretty}); "
            f"pass an explicit Dynkin tuple")
    return candidates[0], False


_GEN_CACHE = {}


def su_n_generators(N, dynkin):
    """Canonical (un-conjugated) hermitian generators of the SU(N) irrep with
    Dynkin labels ``dynkin`` — cached; returns fresh mutable copies."""
    key = (N, tuple(dynkin))
    if key in _GEN_CACHE:
        return [sp.Matrix(m) for m in _GEN_CACHE[key]]
    if all(a == 0 for a in dynkin):
        gens = [sp.zeros(1, 1) for _ in range(N * N - 1)]
    elif tuple(dynkin) == _fundamental_dynkin(N):
        gens = generalized_gell_mann(N)
    elif tuple(dynkin) == _adjoint_dynkin(N):
        gens = _adjoint_generators(N)
    else:
        gens = _gt_generators(N, dynkin)
    _GEN_CACHE[key] = tuple(sp.ImmutableMatrix(m) for m in gens)
    return [sp.Matrix(m) for m in gens]


def generators_for_label(N, label):
    """Hermitian generators of SU(N) for any representation ``label``
    (resolving conjugation: ``T̄^a = −(T^a)^*``)."""
    dynkin, conj = resolve_rep(N, label)
    gens = su_n_generators(N, dynkin)
    if conj:
        gens = [-g.conjugate() for g in gens]
    return gens


def rep_dimension(N, label):
    """Dimension of the SU(N) representation ``label``."""
    dynkin, _ = resolve_rep(N, label)
    return weyl_dim(N, dynkin)
