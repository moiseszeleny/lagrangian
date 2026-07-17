# 7. Diagonalization and the Physical Basis

## Physics statement

A weak-basis mass matrix built in {doc}`masses` is generically not diagonal
— its off-diagonal entries *are* the mixing between weak eigenstates.
Finding the physical (mass) basis means finding a rotation that
diagonalizes it, then rewriting every Lagrangian term in terms of the new,
physical fields. `feynlag` keeps these two facts — "here is a rotation" and
"apply it to the Lagrangian" — cleanly separated in the
{class}`~feynlag.vacuum.diagonalize.Rotation` object.

## `Rotation`: the generic substitution machinery

A `Rotation(old_fields, new_fields, matrix, kind)` (`vacuum/diagonalize.py:23`)
stores $\text{new} = R\cdot\text{old}$ and exposes:

- **`substitution()`**: the dict rewriting *old* fields in terms of *new*
  ones, `old_i → Σ_j (R⁻¹)_{ij} new_j` — i.e. the inverse relation, since
  the Lagrangian is written in weak-basis fields and needs those replaced.
  `kind` selects how the inverse is computed cheaply: `R^T` for
  `'orthogonal'`, `R^†` for `'unitary'`, a generic `R.inv()` for `'general'`
  — never a blind numerical inverse when the structure of $R$ already
  guarantees a cheap closed form.
- **`apply(M)`**: $R\,M\,R^{-1}$ (equivalently $RMR^T$ for symmetric $M$,
  orthogonal $R$) — the matrix conjugated *into* the new basis.
- **`check(M)`**: re-verifies that `apply(M)` really is diagonal, returning
  every nonzero off-diagonal residual rather than trusting the
  construction blindly — this is the code-level enforcement of
  CONVENTIONS.md's rule that a rotation angle must be verified against its
  defining condition, not assumed correct because it came from a
  textbook formula.

`Model.rotate(rotation)` registers a `Rotation` in application order;
`Model.physical_lagrangian` applies every registered rotation's
`substitution()` via `xreplace`, in that same order — which is exactly what
lets two rotations be **chained** (e.g. `examples/sm_u1x.py`'s Weinberg
rotation producing an intermediate $Z^0$ symbol, immediately consumed by a
second Z–Z′ rotation registered right after it).

## 7.1 Analytic 2×2 orthogonal diagonalization

### Derivation: the $\tan2\theta$ formula

For a real symmetric $2\times2$ matrix $M$ and
$R(\theta) = \begin{pmatrix}\cos\theta & \sin\theta\\-\sin\theta & \cos\theta\end{pmatrix}$,
demanding the off-diagonal entry of $R M R^T$ vanish is the defining
condition for the mixing angle. Expanding $(RMR^T)_{12}$ directly:

$$
(RMR^T)_{12} \;=\; \cos2\theta\,M_{12} \;-\; \tfrac12\sin2\theta\,(M_{11}-M_{22}),
$$

using $M_{12}=M_{21}$ (symmetry). Setting this to zero gives

$$
\tan2\theta \;=\; \frac{2M_{12}}{M_{11}-M_{22}}.
$$

`solve_mixing_angle_2x2(M)` (`vacuum/diagonalize.py:110`) returns both
$\theta = \tfrac12\arctan(\tan2\theta)$ **and** the `tan2theta` expression
itself — CONVENTIONS.md requires keeping the defining relation around for
verification, not just the solved angle, because a solved `atan` can hide a
branch/sign choice that only the original `tan 2θ` condition makes
unambiguous. `diagonalize_orthogonal_2x2` wraps this into a full `Rotation`
and attaches `.angle_relation` (`Eq(tan(2θ), tan2theta)`) for exactly that
downstream verification.

## 7.2 SVD (Dirac mass matrices)

### Derivation: why $M M^T$ and $M^T M$

A biunitary decomposition seeks $U_L, U_R$ orthogonal with $U_L M U_R^T =
D$ diagonal. If such $U_L, U_R$ exist, then

$$
U_L(MM^T)U_L^T = U_L M U_R^T\,U_R M^T U_L^T = D D^T = D^2, \qquad
U_R(M^TM)U_R^T = U_R M^T U_L^T\,U_L M U_R^T = D^TD = D^2
$$

(using $U_R U_R^T = U_L U_L^T = I$) — i.e. $U_L$ *must* diagonalize $MM^T$
and $U_R$ *must* diagonalize $M^TM$, both to the same $D^2$. This is a
**necessary** condition, not sufficient: diagonalizing $MM^T$ fixes $U_L$
only up to an independent orthogonal transformation within each degenerate
eigenspace (sign flips, or full rotations for repeated eigenvalues), and
`Mᵀ M`'s diagonalization fixes $U_R$ with the same ambiguity —
**independently**. `diagonalize_svd` (`vacuum/diagonalize.py:158`)
therefore computes $U_L$, $U_R$ from `_orthogonal_diagonalizer` separately
and then **re-aligns** them: it forms $D=U_LMU_R^T$, and for any row where
the intended diagonal entry vanishes, swaps rows of $U_R$ to find the
correct partner column, then flips the sign of any row of $U_R$ needed to
make $D$'s diagonal non-negative.

### Derivation: the analytic 2×2 route (`diagonalize_svd_2x2`)

For symbolic $2\times2$ inputs, `Matrix.diagonalize(normalize=True)`
(SymPy's generic path, used inside `_orthogonal_diagonalizer`) returns
unusable nested-radical/`Abs` expressions once $M$'s entries are symbolic
rather than numeric — so `diagonalize_svd_2x2` avoids ever solving $U_R$
independently from $M^TM$ at all. Instead:

1. Get $\theta_L$ the ordinary 2×2 way, from $M M^T$ (§7.1).
2. Form $N = R(\theta_L)\,M$. Since $R(\theta_L)(MM^T)R(\theta_L)^T$ is
   diagonal by construction of $\theta_L$, and that product equals
   $N N^T$, **$N N^T$ is diagonal** — i.e. $N$'s two rows are orthogonal
   vectors (in the ordinary Euclidean sense).
3. A single rotation that zeroes the second entry of row 0 of $N$,
   $\theta_R = \arctan\bigl(N_{01}/N_{00}\bigr)$, **also** zeroes the
   second entry of row 1: two orthogonal 2-vectors, both rotated by the
   *same* angle, either become simultaneously axis-aligned or stay
   simultaneously off-diagonal — there is no intermediate case in two
   dimensions. So the single condition from row 0 is already sufficient to
   diagonalize the whole matrix.

This makes $U_L M U_R^T$ **exactly** diagonal by an algebraic identity
(orthogonality of $N$'s rows), not by a sign convention that happens to
work for "roughly half of parameter space" — which is precisely the
failure mode of solving $\theta_L$ and $\theta_R$ **independently** from
$MM^T$ and $M^TM$ separately: each angle is only fixed up to its own
eigenvector-ordering/sign convention, and those two conventions can
disagree for a generic (non-symmetric) $M$, silently producing an
anti-diagonal (row-swapped) result. Deriving $\theta_R$ *from* $\theta_L$
and $M$ removes the second independent choice entirely.

## 7.3 Takagi factorization (Majorana mass matrices)

A (real) symmetric matrix $M=M^T$ always admits a real orthogonal
diagonalization $M = O^TDO$ (ordinary spectral theorem), but the diagonal
entries $D_{ii}$ can be **negative** — unphysical for a mass. Takagi
factorization instead seeks $M = UD_{\rm abs}U^T$ with $U$ unitary and
$D_{\rm abs}\ge0$, absorbing the sign into a phase. `diagonalize_takagi`
(`vacuum/diagonalize.py:247`) builds this directly: with `phases` diagonal,
`phases[i,i] = i` (imaginary unit) wherever $D_{ii}<0$ and $1$ otherwise,

$$
U \equiv O^T\cdot\text{phases}, \qquad
U\,D_{\rm abs}\,U^T = O^T\,\text{phases}\,D_{\rm abs}\,\text{phases}\,O
= O^T\bigl(\text{phases}^2 D_{\rm abs}\bigr)O = O^TDO = M,
$$

using $\text{phases}^2_{ii} = i^2 = -1$ exactly where $D_{ii}$ was
negative, so $\text{phases}^2 D_{\rm abs} = D$ recovers the original
(signed) diagonal. $U$ is genuinely complex (not merely orthogonal) only
when at least one eigenvalue was negative — the standard Majorana-phase
convention for absorbing an unphysical negative mass into the field
redefinition.

## 7.4 Fermion mass-basis rotations

Fermion rotations work through the **same** `Rotation`/`Model.rotate`
machinery as bosons, with one structural difference: `old_fields`/
`new_fields` are `Indexed` — e.g.
`Rotation([eL[i], EL[i]], [e1L[i], e2L[i]], rotation_2x2(θ))` — carrying an
explicit flavor-index symbol `i` that must be the **same** symbol
everywhere the Lagrangian was written, so `xreplace`'s key matching
actually fires.

Two rules make this work correctly:

- **Register two rotations per chirality** — one for the field leg, one for
  the bar leg, both with the *same* rotation matrix (a fermion field and
  its Dirac adjoint must rotate together for the bilinear to remain a
  consistent physical object).
- **`expand_bilinear` is mandatory after rotation**, and is applied
  automatically by `extract_fermion_vertices` and `fermion_mass_matrix`
  ({doc}`vertices`). After `xreplace` substitutes a rotated field, a
  `Bilinear` slot ends up holding an `Add` — e.g.
  `cosθ·ψ₁[i] + sinθ·ψ₂[i]` — but `Bilinear` is an opaque custom
  `Function`, so ordinary `sp.expand()` does **not** distribute it over
  that sum (the same "teach SymPy about a custom operator's linearity"
  problem `PartialMu`/`D_linear` solves for derivatives, {doc}`lagrangian`,
  applied here to two argument slots instead of one). Without
  `expand_bilinear`, vertex extraction would group terms by the *unsplit*
  composite `Add`-valued key instead of by each individual mass-eigenstate
  bilinear — silently wrong, not an error. `examples/sm_vll.py` and its
  tutorial notebook include a standalone demonstration of exactly this
  failure mode.

## Design gotchas

- **Never use generic `Matrix.diagonalize()`/`eigenvects()` on symbolic
  matrices larger than 2×2.** SymPy's closed-form solver produces nested
  radicals that don't simplify — use a `rotation_2x2`-style ansatz per
  2×2 block, or fall back to numeric diagonalization at export time.
- **Verify every rotation against its `tan2θ` (or equivalent) defining
  condition**, not merely `sin²+cos²=1` — `Rotation.check()` exists
  precisely so this isn't left to manual inspection.

## Validation

- `tests/test_scalar_pipeline_thdm.py::test_alpha_rotation_diagonalizes_cp_even`,
  `::test_goldstone_beta_rotation` — the analytic 2×2 route, §7.1.
- `tests/test_fermion_sector.py::TestDiagonalization`,
  `::TestRotatedBilinearExtraction` — the biunitary SVD route and the
  `expand_bilinear` requirement, §7.2/§7.4.
- `tests/test_vll.py::test_mixing_angles`, `::test_svd_diagonalizes` — the
  full VLL 2×2 biunitary case end to end, both symbolically and at random
  numeric points.

## Minimal snippet

```python
from feynlag import diagonalize_svd_2x2, rotation_2x2

rot_L, rot_R = diagonalize_svd_2x2(
    M, [eL[i], EL[i]], [eLbar[i], ELbar[i]],
    [e1L[i], e2L[i]], [e1Lbar[i], e2Lbar[i]],
    angle_left=sp.Symbol("theta_L"), angle_right=sp.Symbol("theta_R"))
model.rotate(rot_L)
model.rotate(rot_R)
```
