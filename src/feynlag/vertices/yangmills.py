"""Triple and quartic gauge self-couplings from group data.

Instead of expanding ``−¼ F_{μν}F^{μν}`` with explicit Lorentz indices, the
self-interactions are computed group-theoretically (their Lorentz structures
are universal and fixed by the catalog):

- cubic:   ``L ⊃ −g f^{abc} (∂_μ A^a_ν) A^{bμ} A^{cν}``
  → vertex ``V(i,j,k) = −g F_ijk × [g^{μν}(p₁−p₂)^ρ + cyclic]`` with
  ``F_ijk = Σ_abc f^{abc} U[a,i] U[b,j] U[c,k]``,
- quartic: pairwise tensors ``E_{(ij)(kl)} = Σ_e F^e_ij F^e_kl`` with
  ``F^e_ij = Σ_ab f^{abe} U[a,i] U[b,j]``, entering
  ``−g²/4 Σ_e (f^{abe}A^aA^b)(f^{cde}A^cA^d)`` structures.

``U`` is the (possibly complex, unitary) matrix from adjoint components to
physical bosons: ``A^a = Σ_i U[a,i] V_i`` (e.g. W¹,W²,W³,B → W⁺,W⁻,Z,γ; the
identity if the bosons don't mix).  Overall sign/i conventions are pinned at
UFO validation against the FeynRules SM model.
"""

import itertools

import sympy as sp

__all__ = ["structure_constants", "cubic_couplings", "quartic_couplings"]


def structure_constants(group):
    """``f^{abc}`` of a gauge group, from the fundamental representation:
    ``f^{abc} = −2i Tr([T^a, T^b] T^c)``.

    Returns:
        dict ``{(a, b, c): f_abc}`` (only non-zero entries).
    """
    if group.abelian:
        return {}
    T = group.generators(_fundamental_label(group))
    n = group.n_generators
    f = {}
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            comm = T[a] * T[b] - T[b] * T[a]
            for c in range(n):
                val = sp.nsimplify(
                    sp.simplify(-2 * sp.I * sp.trace(comm * T[c])))
                if val != 0:
                    f[(a, b, c)] = val
    return f


def _fundamental_label(group):
    """Label of the fundamental representation (2 for SU2, 3 for SU3)."""
    from ..groups.gauge import SU2, SU3
    if isinstance(group, SU2):
        return 2
    if isinstance(group, SU3):
        return 3
    raise TypeError(f"no fundamental representation known for {group!r}")


def _f_tensor(group):
    """Structure constants as a dense rank-3 nested list."""
    n = group.n_generators
    f = structure_constants(group)
    return [[[f.get((a, b, c), sp.S.Zero) for c in range(n)]
             for b in range(n)] for a in range(n)], n


def cubic_couplings(group, physical=None, U=None):
    """Rotated cubic self-coupling tensor ``−g F_ijk``.

    Args:
        group: non-abelian gauge group.
        physical: physical boson symbols ``V_i`` (default: the group's own
            adjoint components, ``U = 1``).
        U: matrix with ``A^a = Σ_i U[a,i] V_i``.

    Returns:
        dict ``{(V_i, V_j, V_k): −g·F_ijk}`` (only non-zero, all orderings —
        the tensor is totally antisymmetric under simultaneous exchange).
        Lorentz structure: the universal YM ``VVV`` catalog entry.
    """
    f, n = _f_tensor(group)
    if physical is None:
        physical = list(group.bosons().components)
        U = sp.eye(n)
    U = sp.Matrix(U)
    m = len(physical)

    couplings = {}
    for i, j, k in itertools.product(range(m), repeat=3):
        F = sp.S.Zero
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    if f[a][b][c] != 0:
                        F += f[a][b][c] * U[a, i] * U[b, j] * U[c, k]
        F = sp.simplify(F)
        if F != 0:
            couplings[(physical[i], physical[j], physical[k])] = -group.g * F
    return couplings


def quartic_couplings(group, physical=None, U=None):
    """Rotated quartic tensor ``−g²/4 Σ_e F^e_ij F^e_kl`` per boson quadruple.

    Returns:
        dict ``{(V_i, V_j, V_k, V_l): coupling}`` where the coupling
        multiplies the ``(f·A·A)²`` contraction structure — i.e. the value of
        ``−g²/4 Σ_e F^e_ij F^e_kl`` for that *ordered* quadruple.  The UFO
        writer assembles the three metric-pair structures from the ordered
        tensor (Phase 5).
    """
    f, n = _f_tensor(group)
    if physical is None:
        physical = list(group.bosons().components)
        U = sp.eye(n)
    U = sp.Matrix(U)
    m = len(physical)

    # F^e_ij = sum_ab f_abe U[a,i] U[b,j]
    Fe = [[[sp.S.Zero for _ in range(m)] for _ in range(m)] for _ in range(n)]
    for e in range(n):
        for i in range(m):
            for j in range(m):
                val = sp.S.Zero
                for a in range(n):
                    for b in range(n):
                        if f[a][b][e] != 0:
                            val += f[a][b][e] * U[a, i] * U[b, j]
                Fe[e][i][j] = sp.simplify(val)

    couplings = {}
    for i, j, k, l in itertools.product(range(m), repeat=4):
        val = sp.S.Zero
        for e in range(n):
            val += Fe[e][i][j] * Fe[e][k][l]
        val = sp.simplify(-group.g ** 2 / 4 * val)
        if val != 0:
            couplings[(physical[i], physical[j],
                       physical[k], physical[l])] = val
    return couplings
