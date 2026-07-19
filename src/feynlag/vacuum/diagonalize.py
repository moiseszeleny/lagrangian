"""Rotations from the weak basis to the physical (mass) basis.

- :class:`Rotation`: holds ``new = R · old``, produces the substitution dict
  ``old → Rᵀ·new`` for rewriting the Lagrangian, and verifies that it
  actually diagonalizes a given mass matrix (the tan-2θ verification rule of
  CONVENTIONS.md is enforced by construction here: angles come from the
  off-diagonal condition, and :meth:`Rotation.check` re-verifies).
- 2×2 analytic orthogonal diagonalization (the BSM workhorse).
- Larger symbolic matrices: use a :func:`rotation_2x2`-style user ansatz and
  ``solve_mixing_angle_2x2`` per block, or numeric diagonalization at export
  time — never rely on symbolic ``eigenvects`` for >2×2 (see plan risks).

SVD (Dirac) and Takagi (Majorana) arrive with the fermion sector (Phase 4).
"""

import sympy as sp

__all__ = ["Rotation", "rotation_2x2", "solve_mixing_angle_2x2",
           "diagonalize_orthogonal_2x2", "diagonalize_svd",
           "diagonalize_svd_2x2", "diagonalize_takagi", "MajoranaRotation"]


class Rotation:
    """A change of basis ``new = R · old``.

    Args:
        old_fields: weak-basis symbols.
        new_fields: physical-basis symbols (created by the caller — physics
            names like ``h``, ``H``, ``G0``, ``A0``).
        matrix: the rotation ``R`` with ``new_i = Σ_j R[i,j] old_j``.
        kind: 'orthogonal' (default), 'unitary' or 'general' — selects how
            the inverse is computed (``Rᵀ``, ``R†``, ``R⁻¹``).
    """

    def __init__(self, old_fields, new_fields, matrix, kind="orthogonal"):
        self.old_fields = list(old_fields)
        self.new_fields = list(new_fields)
        self.matrix = sp.Matrix(matrix)
        if kind not in ("orthogonal", "unitary", "general"):
            raise ValueError(f"unknown rotation kind {kind!r}")
        self.kind = kind
        n = len(self.old_fields)
        if self.matrix.shape != (n, n) or len(self.new_fields) != n:
            raise ValueError("rotation matrix/field dimensions do not match")

    @property
    def inverse(self):
        if self.kind == "orthogonal":
            return self.matrix.T
        if self.kind == "unitary":
            return self.matrix.conjugate().T
        return self.matrix.inv()

    def substitution(self):
        """Dict rewriting old (weak) fields in terms of new (physical) ones:
        ``old_i → Σ_j (R⁻¹)[i,j] new_j``."""
        Rinv = self.inverse
        new_vec = sp.Matrix(len(self.new_fields), 1, self.new_fields)
        old_expr = Rinv * new_vec
        return {old: sp.expand(old_expr[i])
                for i, old in enumerate(self.old_fields)}

    def apply(self, M):
        """``R M R⁻¹``-conjugated matrix in the new basis (for symmetric M
        and orthogonal R this is ``R M Rᵀ``)."""
        return self.matrix * sp.Matrix(M) * self.inverse

    def check(self, M, simplifier=sp.simplify):
        """Verify that this rotation diagonalizes ``M``.

        Returns:
            ``(ok: bool, off_diagonal_residuals: list)``.
        """
        D = self.apply(M)
        residuals = []
        n = D.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = simplifier(D[i, j])
                if r != 0:
                    residuals.append(((i, j), r))
        return (not residuals, residuals)

    def masses_squared(self, M, simplifier=sp.simplify):
        """Diagonal entries of ``R M R⁻¹`` (physical masses², new-field order)."""
        D = self.apply(M)
        return [simplifier(D[i, i]) for i in range(D.shape[0])]

    def __repr__(self):
        pairs = ", ".join(f"{o}→{n}" for o, n in
                          zip(self.old_fields, self.new_fields))
        return f"Rotation({pairs}, kind={self.kind!r})"

    def _repr_latex_(self):
        new_vec = sp.Matrix(self.new_fields)
        old_vec = sp.Matrix(self.old_fields)
        eq = sp.Eq(new_vec, self.matrix * old_vec, evaluate=False)
        return f"$\\displaystyle {sp.latex(eq)}$"


def rotation_2x2(theta):
    """``R(θ) = [[cosθ, sinθ], [−sinθ, cosθ]]`` —
    ``new₁ = cθ old₁ + sθ old₂``, ``new₂ = −sθ old₁ + cθ old₂``."""
    c, s = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[c, s], [-s, c]])


def solve_mixing_angle_2x2(M, positive_denominator=True):
    """Mixing angle of a real symmetric 2×2 matrix from the off-diagonal
    condition ``(R M Rᵀ)₁₂ = 0``:  ``tan 2θ = 2 M₁₂ / (M₁₁ − M₂₂)``.

    Returns:
        ``(theta, tan2theta)`` — the angle ``θ = ½ atan(...)`` and the
        defining ``tan 2θ`` expression (keep it: CONVENTIONS.md requires
        verifying any rotation angle against its tan-2θ source).
    """
    M = sp.Matrix(M)
    if M.shape != (2, 2):
        raise ValueError("solve_mixing_angle_2x2 needs a 2×2 matrix")
    tan2theta = sp.cancel(2 * M[0, 1] / (M[0, 0] - M[1, 1]))
    theta = sp.atan(tan2theta) / 2
    return theta, tan2theta


def diagonalize_orthogonal_2x2(M, old_fields, new_fields, angle=None):
    """Diagonalize a real symmetric 2×2 mass matrix.

    Args:
        M: the matrix.
        old_fields, new_fields: 2 symbols each.
        angle: optional symbol for the mixing angle.  If given, the returned
            rotation is written in terms of it and the defining relation is
            attached as ``rotation.angle_relation`` (``Eq(tan(2θ), ...)``);
            if None, the explicit ``atan`` expression is inserted.

    Returns:
        :class:`Rotation` (with ``angle_relation`` attribute).
    """
    theta_expr, tan2theta = solve_mixing_angle_2x2(M)
    theta = angle if angle is not None else theta_expr
    rot = Rotation(old_fields, new_fields, rotation_2x2(theta),
                   kind="orthogonal")
    rot.angle_relation = sp.Eq(sp.tan(2 * theta), tan2theta)
    rot.angle_solution = theta_expr
    return rot


def _orthogonal_diagonalizer(M):
    """Orthonormal ``O`` with ``O M Oᵀ`` diagonal, for symmetric ``M``
    (rows of ``O`` are normalized eigenvectors).  Small matrices only."""
    M = sp.Matrix(M)
    P, D = M.diagonalize(normalize=True)   # M = P D P⁻¹, P orthogonal
    return P.T, D


def diagonalize_svd(M, left_fields, right_fields, new_left, new_right):
    """Singular-value decomposition for a Dirac mass matrix.

    Finds rotations with ``U_L M U_Rᵀ = diag(m_i ≥ 0)``: ``U_L``
    diagonalizes ``M Mᵀ`` and ``U_R`` diagonalizes ``Mᵀ M`` (real ``M``;
    complex Yukawas can be handled numerically at export time).  Right-handed
    rows are sign-fixed so the masses come out non-negative.

    Returns:
        ``(rot_left, rot_right)`` :class:`Rotation` objects
        (``ψ_L → U_L ψ_L``, ``ψ_R → U_R ψ_R``).
    """
    M = sp.Matrix(M)
    UL, _ = _orthogonal_diagonalizer(M * M.T)
    UR, _ = _orthogonal_diagonalizer(M.T * M)

    # align the eigenvalue ordering of the right rotation with the left one
    D = UL * M * UR.T
    n = D.shape[0]
    # permute/sign-fix columns of UR so D is diagonal and non-negative
    for i in range(n):
        if sp.simplify(D[i, i]) == 0:
            # find the column j with D[i, j] != 0 and swap
            for j in range(n):
                if i != j and sp.simplify(D[i, j]) != 0:
                    UR = UR.elementary_row_op("n<->m", row1=i, row2=j)
                    D = UL * M * UR.T
                    break
    for i in range(n):
        if D[i, i].is_negative:
            UR[i, :] = -UR[i, :]
    rot_left = Rotation(left_fields, new_left, UL, kind="orthogonal")
    rot_right = Rotation(right_fields, new_right, UR, kind="orthogonal")
    return rot_left, rot_right


def diagonalize_svd_2x2(M, left_fields, right_fields, new_left, new_right,
                        angle_left=None, angle_right=None):
    """Analytic biunitary SVD of a real 2×2 Dirac mass matrix.

    ``θ_L`` diagonalizes ``M·Mᵀ`` via :func:`diagonalize_orthogonal_2x2` —
    the clean symbolic route (``diagonalize_svd``'s
    ``Matrix.diagonalize(normalize=True)`` path returns unusable
    nested-sqrt/``Abs`` forms for symbolic entries).  ``θ_R`` is then
    **derived from θ_L and M directly** (``N = R(θ_L)·M``; since
    ``N·Nᵀ`` is diagonal by construction, ``N``'s rows are automatically
    orthogonal, so ``θ_R = atan(N[0,1]/N[0,0])`` biunitarily pairs with
    ``θ_L`` by construction) rather than solved independently from
    ``Mᵀ·M``: solving the two angles independently only fixes each up to
    its own sign/eigenvector-ordering convention, and those two
    conventions can disagree — for a generic (non-symmetric) ``M`` this
    silently produces an anti-diagonal (row-swapped) ``U_L·M·U_Rᵀ`` for
    roughly half of parameter space, not merely a sign flip. Deriving
    θ_R from θ_L guarantees ``U_L·M·U_Rᵀ`` is exactly diagonal always
    (the off-diagonal vanishing is an algebraic identity, not dependent
    on the sign pattern of ``M``'s entries) — verified in
    ``tests/test_fermion_sector.py``/``tests/test_vll.py`` both
    symbolically and at random numeric points per CONVENTIONS.md.

    Args:
        M: real 2×2 mass matrix (rows = bar/left legs, cols = right legs).
        left_fields / right_fields: 2 weak-basis symbols each.
        new_left / new_right: 2 physical-basis symbols each.
        angle_left / angle_right: optional angle symbols; if given, the
            rotations carry them with the defining ``tan 2θ`` relation
            attached (``.angle_relation`` / ``.angle_solution``).

    Returns:
        ``(rot_left, rot_right)`` :class:`Rotation` objects.
    """
    M = sp.Matrix(M)
    if M.shape != (2, 2):
        raise ValueError("diagonalize_svd_2x2 needs a 2×2 matrix")
    rot_left = diagonalize_orthogonal_2x2(M * M.T, left_fields, new_left,
                                          angle=angle_left)
    N = rotation_2x2(rot_left.angle_solution) * M
    theta_R_expr = sp.atan(N[0, 1] / N[0, 0])
    tan2R = sp.cancel(2 * N[0, 0] * N[0, 1] / (N[0, 0] ** 2 - N[0, 1] ** 2))
    aR = angle_right if angle_right is not None else theta_R_expr
    rot_right = Rotation(right_fields, new_right, rotation_2x2(aR),
                         kind="orthogonal")
    # tan(2·aR) expanded to tan(aR) BEFORE substituting aR->theta_R_expr:
    # sympy proves tan(atan(z))=z trivially but not tan(2*atan(z))=2z/(1-z²)
    # directly, so expanding first is what makes the relation verifiable.
    rot_right.angle_relation = sp.Eq(sp.expand_trig(sp.tan(2 * aR)), tan2R)
    rot_right.angle_solution = theta_R_expr
    return rot_left, rot_right


def diagonalize_takagi(M, old_fields=None, new_fields=None):
    """Takagi factorization of a (real) symmetric Majorana mass matrix.

    Returns ``(U, D)`` with ``M = U D Uᵀ`` and ``D`` diagonal non-negative:
    a real orthogonal diagonalization with factors of ``i`` absorbing
    negative eigenvalues (the standard Majorana phase convention).

    If ``old_fields``/``new_fields`` are given, also returns the
    corresponding unitary :class:`Rotation` (``new = U† old``) as third
    element.
    """
    M = sp.Matrix(M)
    if sp.simplify(M - M.T) != sp.zeros(*M.shape):
        raise ValueError("Takagi factorization needs a symmetric matrix")
    O, D = _orthogonal_diagonalizer(M)      # O M Oᵀ = D (may be negative)
    n = D.shape[0]
    phases = sp.eye(n)
    for i in range(n):
        if D[i, i].is_negative:
            phases[i, i] = sp.I
    # M = Oᵀ D O ⇒ with U = Oᵀ·phases:  U |D| Uᵀ = Oᵀ phases |D| phases Oᵀᵀ
    U = O.T * phases
    D_abs = sp.Matrix(n, n, lambda a, b: sp.Abs(D[a, b]) if a == b else 0)
    if old_fields is not None and new_fields is not None:
        rot = Rotation(old_fields, new_fields, U.conjugate().T,
                       kind="unitary")
        return U, D_abs, rot
    return U, D_abs


class MajoranaRotation:
    """Rotate weak-basis neutrinos to physical Majorana mass eigenstates.

    Given the Takagi factor ``U`` of a seesaw mass matrix (``M_ν = U D Uᵀ``, from
    :func:`diagonalize_takagi` on :func:`~feynlag.vacuum.masses.seesaw_mass_matrix`)
    in the left-handed basis ``n = (ν_L, ν_R^c)``, the ``N`` physical Majorana
    fields ``χ_k`` (``χ_k = χ_k^c``) satisfy ``n = U^* χ``.  Because ``χ`` is
    self-conjugate, ``(P_L χ)^c = P_R χ``, so the weak Weyl fields substitute as

        ν_L[g]   → Σ_k U*[g, k]      χ_L[k]      ν̄_L[g] → Σ_k U[g, k]      χ̄_L[k]
        ν_R[g]   → Σ_k U[n_L+g, k]   χ_R[k]      ν̄_R[g] → Σ_k U*[n_L+g, k] χ̄_R[k]

    (rows ``0 … n_L−1`` of ``U`` are the ``ν_L`` generations, rows ``n_L …`` the
    ``ν_R`` generations).  The convention is pinned in
    ``tests/test_seesaw.py``: substituting these into the SM charged current
    gives ``W ℓ̄ χ_k = (g/√2)·U*[g,k]`` — for the heavy states the light–heavy
    mixing ``≈ m_D M_R⁻¹``, and → 0 as ``M_R → ∞`` (decoupling).

    Args:
        U: the ``N×N`` Takagi matrix (a concrete/analytic ``Matrix`` — **not** a
            symbolic ``IndexedBase``, whose ``U[i,k]`` would collide with the
            field leg in :func:`~feynlag.vertices.bilinear.expand_bilinear`).
        nuL, nuR: the weak ``ν_L`` / ``ν_R`` component ``IndexedBase``\\ s.
        nuLbar, nuRbar: their Dirac-adjoint ``IndexedBase``\\ s.
        chiL, chiR, chiLbar, chiRbar: the physical Majorana ``IndexedBase``\\ s
            (``χ_L[k]`` = ``P_L χ_k`` etc.).
        n_L: number of left-handed neutrinos (the ``ν_R`` row offset).
    """

    def __init__(self, U, nuL, nuR, nuLbar, nuRbar,
                 chiL, chiR, chiLbar, chiRbar, n_L):
        self.U = sp.Matrix(U)
        self.N = self.U.rows
        self.n_L = n_L
        self.nuL, self.nuR = nuL, nuR
        self.nuLbar, self.nuRbar = nuLbar, nuRbar
        self.chiL, self.chiR = chiL, chiR
        self.chiLbar, self.chiRbar = chiLbar, chiRbar

    def _leg(self, base, gen):
        """Substitution value for one weak neutrino leg at generation ``gen``.

        Compares by ``==`` (name), not identity: after a rotation's ``xreplace``
        the Lagrangian's ``IndexedBase`` can be a name-equal but object-distinct
        instance (SymPy does not intern ``IndexedBase``), so an ``is`` check
        would silently miss it and leave the field un-rotated.
        """
        U, N = self.U, self.N
        if base == self.nuL:
            return sum(U[gen, k].conjugate() * self.chiL[k] for k in range(N))
        if base == self.nuLbar:
            return sum(U[gen, k] * self.chiLbar[k] for k in range(N))
        if base == self.nuR:
            return sum(U[self.n_L + gen, k] * self.chiR[k] for k in range(N))
        if base == self.nuRbar:
            return sum(U[self.n_L + gen, k].conjugate() * self.chiRbar[k]
                       for k in range(N))
        return None

    def apply(self, expr, gen_indices, n_gen):
        """Rewrite ``expr`` in the physical basis.

        ``gen_indices`` is the flavour-index symbol (or tuple of symbols) the
        weak fields carry; each is expanded over ``range(n_gen)`` (every
        generation substituted with its ``U``-block combination of the
        ``χ_k``).  Charged-lepton ``Indexed`` legs on the same indices keep
        their concrete generation.
        """
        import itertools
        if not isinstance(gen_indices, (list, tuple)):
            gen_indices = (gen_indices,)
        bases = {self.nuL, self.nuR, self.nuLbar, self.nuRbar}
        total = sp.S.Zero
        for combo in itertools.product(range(n_gen), repeat=len(gen_indices)):
            term = expr.subs(dict(zip(gen_indices, combo)), simultaneous=True)
            term = term.replace(
                lambda x: isinstance(x, sp.Indexed) and x.base in bases,
                lambda x: self._leg(x.base, int(x.indices[0])))
            total += term
        return sp.expand(total)
