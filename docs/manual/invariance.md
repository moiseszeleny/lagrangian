# 4. Checking Invariance

## Physics statement

A Lagrangian term is physical only if it respects the symmetries the model
declares: gauge invariance under every continuous group, invariance under
every discrete symmetry, hermiticity of the total Lagrangian, and mass
dimension $\le 4$ (renormalizability). `feynlag`'s strategy for all of this
is **explicit component transformation** — build the actual transformed
expression and check that it equals the original — rather than tracking
abstract representation labels and inferring invariance from index
contraction rules. This is slower symbolically but categorically safer: it
catches the invariance violation a purely bookkeeping-based checker would
miss if a term were mistyped.

`Model.check_invariance()` (`lagrangian.py:327`) is the entry point,
running every check below over every Lagrangian term and every declared
group, and collecting failures into an `InvarianceReport` rather than
raising on the first one.

## 4.1 Gauge invariance

### Algorithm

For each `(term, group)` pair, `check_gauge_invariance` (`invariance.py:185`)
via `gauge_variation`:

1. Introduce one real `Dummy` parameter $\alpha^a$ per generator of `group`.
2. Build the substitution map `_transform_map` sending every component
   `comp → comp + iα^a(T^a\cdot\text{comps})_i` (first-order in $\alpha$),
   **and** the identical linear substitution for `PartialMu(comp)`.
3. Apply the map (after Leibniz-expanding any compound `PartialMu`, so only
   atomic `PartialMu(component)` nodes remain to match the substitution
   keys), plus the parallel fermion-leg transform if the term contains a
   `Bilinear`.
4. Compute `delta = expand(transformed − term)`, then for each $\alpha^a$
   take `diff(delta, α^a)` and set every $\alpha$ to zero — this isolates
   the $O(\alpha)$ coefficient of each generator.
5. The term is gauge-invariant iff every one of these coefficients is
   identically zero.

### Derivation: why the transform commutes with $\partial_\mu$

The infinitesimal gauge transformation of a field is
$\delta\phi = i\alpha^aT^a\phi$ with $\alpha^a$ **spacetime-constant**
(the check tests global, not local, invariance of the raw kinetic term —
the compensating $-igT^aA^a_\mu$ piece inside $D_\mu$ is what promotes this
to a genuine local symmetry, and it's already baked into any term written
with `Dmu`). Because $\alpha^a$ carries no $x$-dependence,

$$
\partial_\mu(\delta\phi) \;=\; \partial_\mu\bigl(i\alpha^aT^a\phi\bigr) \;=\; i\alpha^aT^a(\partial_\mu\phi) \;=\; \delta(\partial_\mu\phi).
$$

In other words, the variation operator and $\partial_\mu$ **commute
exactly** for a global transformation. This is the entire justification for
`_transform_map` giving `PartialMu(comp)` the *identical* linear
substitution as `comp` itself (`invariance.py:64`) — there is no need to
push the substitution through the `PartialMu` wrapper by differentiating,
which is fortunate, because `PartialMu` has no `fdiff` rule (see
{doc}`lagrangian`) and any such attempt would leave an unevaluated,
non-cancelling `Subs(Derivative(...))` residue. Substituting *outside* both
operators, using their known commutation, sidesteps the trap entirely. This
exact bug — differentiating through `PartialMu` — was hit and fixed once
for `Dmu`-built kinetic terms during development; understanding why the
commuting-substitution route is not just a shortcut but the *correct* fix
is why this derivation is spelled out here rather than left implicit in the
code comment.

### Fermion legs: a different substitution mechanism

Fermion components are `IndexedBase`-typed and only ever appear as
`Indexed(base, flavor_index)` — a bare `xreplace` dict (keyed on whole
symbols) cannot handle this, because the flavor index on the *matched* node
must be preserved symbolically in the replacement. `_fermion_transform`
(`invariance.py:87`) instead uses `sympy.Basic.replace` with a **predicate +
builder function**: the predicate matches any `Indexed` node whose base is
a known fermion component, and the builder receives the matched node itself
(so it can read off `node.indices`) and returns

$$
\delta\psi_r[i] \;=\; i\alpha\sum_c T_{rc}\,\psi_c[i], \qquad
\delta\bar\psi_r[i] \;=\; -i\alpha\sum_c \bar\psi_c[i]\,T_{cr}
$$

— the bar leg transforming in the conjugate (dual) representation, sign and
index order flipped relative to the field leg. Since $T$ is hermitian (PDG
convention) and $\delta\psi = i\alpha T\psi$, the bar-leg convention
$\delta\bar\psi = -i\alpha\,\bar\psi T$ is exactly what makes a singlet
bilinear invariant: $\delta(\bar\psi\psi) = \delta\bar\psi\cdot\psi +
\bar\psi\cdot\delta\psi = -i\alpha\,\bar\psi T\psi + i\alpha\,\bar\psi T\psi
= 0$ — the standard "dual representation" prescription, generalized here to
matrix multiplication order matching the flavor-preserved index structure.
After the fermion transform is applied,
`expand_bilinear` ({doc}`vertices`) distributes the now `Add`-valued
bar/field legs of any `Bilinear` node before the $O(\alpha)$ coefficient
extraction runs.

## 4.2 Discrete invariance

### Algorithm

Discrete transformations are **finite** — no $\alpha$-linearization or
differentiation is needed, because a group element like $S_3$'s
transposition is not infinitesimally close to the identity.
`check_discrete_invariance` (`invariance.py:198`) iterates over
`group.generator_maps()` (one substitution dict per generator) and, for
each, checks `expand(transformed − term) == 0` directly (falling back to
`simplify` for the residual when the raw `expand` doesn't immediately
vanish — needed because $Z_N$ phases like $\omega^k$ often require trig/exp
simplification to visibly cancel).

### Derivation: the bar-leg matrix $X = (M^{-1})^T$

For a discrete multiplet $(\psi_1,\dots,\psi_d)$ in irrep matrix $M$
(`group.assign(irrep, ...)`), the field-leg substitution is the direct
image of the representation: $\psi_i' = \sum_k M_{ik}\psi_k$. The bar leg
needs its *own* matrix $X$, and $X \ne M$ in general — it must be
**derived**, not assumed, from the physical requirement that the diagonal
bilinear sum $\sum_i \bar\psi_i\psi_i$ (the fermion kinetic/mass term
pattern) stays invariant. Writing $\bar\psi_i' = \sum_k X_{ik}\bar\psi_k$:

$$
\sum_i \bar\psi_i'\psi_i' \;=\; \sum_i\Bigl(\sum_k X_{ik}\bar\psi_k\Bigr)\Bigl(\sum_l M_{il}\psi_l\Bigr)
\;=\; \sum_{k,l}\Bigl(\sum_i X_{ik}M_{il}\Bigr)\bar\psi_k\psi_l .
$$

Invariance requires this to equal $\sum_k \bar\psi_k\psi_k$, i.e. the
coefficient tensor $\sum_i X_{ik}M_{il}$ must equal $\delta_{kl}$ — in
matrix form $X^TM = I$, i.e.

$$
X = (M^{-1})^T .
$$

`fermion_generator_data()` (`groups/discrete.py:126`) computes this
generically as `M.inv().T` for every generator, rather than assuming a
shortcut. $X$ only coincides with $M$ itself when $M$ is real orthogonal
($M^{-1} = M^T \Rightarrow (M^{-1})^T = M$) — true for every $S_3$ irrep
generator (built from real rotations/reflections, see {doc}`declaration`),
but **false** for $Z_N$'s complex-phase irreps, where $X = \bar M$ instead
(a unitary, non-orthogonal $M = [\omega^k]$ gives $M^{-1} = M^\dagger =
\bar M^T$, so $(M^{-1})^T = \bar M$). Computing `M.inv().T` generically
rather than hard-coding "use $M$ for bosons, $M$ again for fermions" is
precisely what makes the discrete-fermion invariance check correct for
*both* group families without a special case per group type.

The bosonic substitution in `generator_maps()` needs no analogous
derivation because a *scalar* bilinear pattern doesn't arise the same way —
bosonic invariants are written explicitly (e.g. via `S3.doublet_product`)
and checked by the same finite-substitution machinery without a bar leg.

## 4.3 Hermiticity

### Algorithm

For bosonic sectors, `check_hermiticity(expr)` (`invariance.py:221`) is
literally `expand(expr - conjugate(expr)) == 0` (with a `simplify` fallback
for the residual). `Model.check_invariance` runs this once per sector on
the sector's *total* (not per-term), because cross-term cancellations
(e.g. a Yukawa term and its explicitly-written `+ h.c.` partner) are
expected and must not each individually vanish.

### Derivation: the fermion-bilinear conjugation rule

A Yukawa or gauge-current term written with a `Bilinear` needs its own
hermitian-conjugate rule, since `sympy.conjugate` has no built-in notion of
Dirac adjoints. `Bilinear._eval_conjugate` (`vertices/bilinear.py:65`)
implements

$$
\bigl(\bar\psi_1\Gamma\psi_2\bigr)^\dagger \;=\; \bar\psi_2\,\bar\Gamma\,\psi_1, \qquad \bar\Gamma \equiv \gamma^0\Gamma^\dagger\gamma^0,
$$

the standard field-theory identity for a fermion sandwich, with the bar/field
roles swapped, flavor indices carried over on their own field (the field-2
index goes with the new bar, the field-1 index with the new field), and
$\Gamma \to \bar\Gamma$. `dirac.dirac_conjugate` (`dirac.py:476`) computes
$\bar\Gamma$ for exactly the catalog of structures the library actually
builds:

- $\bar\Gamma = 1$ for $\Gamma = 1$ (`diracI`) — trivial.
- **Bare $\gamma^\mu$ is self-conjugate.** In the standard Dirac
  representation with metric $(+,-,-,-)$, $\gamma^0$ is hermitian and
  $\gamma^{\mu\dagger} = \gamma^0\gamma^\mu\gamma^0$, so
  $\bar\gamma^\mu = \gamma^0(\gamma^0\gamma^\mu\gamma^0)\gamma^0 = \gamma^\mu$
  (using $(\gamma^0)^2=1$).
- **Bare chiral projectors swap.** $P_{L,R} = (1\mp\gamma_5)/2$ are
  hermitian, but $\gamma_5$ **anticommutes** with every $\gamma^\mu$
  (Clifford algebra), hence with $\gamma^0$ in particular:
  $\gamma^0 P_L\gamma^0 = \gamma^0\frac{1-\gamma_5}{2}\gamma^0 =
  \frac{1-(-\gamma_5)}{2}\cdot(\gamma^0)^2 \cdot(\ldots)$; carrying through
  the anticommutation carefully gives $\gamma^0 P_L \gamma^0 = P_R$ and
  vice versa — a **bare** projector's conjugate is the *opposite* chirality
  projector.
- **A vector current is self-conjugate.** For $\Gamma = \gamma^\mu P_L$ (or
  $P_R$), the two effects above cancel: $\gamma^\mu P_L = P_R\gamma^\mu$
  (moving $\gamma^\mu$ past $P_L$ flips its chirality by the same
  anticommutation), so $\bar\Gamma = \gamma^0(\gamma^\mu P_L)^\dagger\gamma^0
  = \gamma^0 P_L\gamma^{\mu\dagger}\gamma^0$ works out to $\gamma^\mu P_L$
  again — the familiar "V$-$A ports to V$-$A" statement: the h.c. of
  $\bar\nu\gamma^\mu P_L\ell$ is $\bar\ell\gamma^\mu P_L\nu$, same $\Gamma$,
  bar/field swapped. This is exactly `fermion_gauge_current`'s output
  structure, which is why a gauge current needs **no** hand-written `+ h.c.`
  term to pass the hermiticity check — it is hermitian by construction.

Everything **outside** this catalog (`diracI`, `diracPL`, `diracPR`, bare
`DiracGamma(mu)`, and `DiracGamma(mu)*diracPL`/`diracPR`) raises
`NotImplementedError` rather than guessing — e.g. products of two or more
gamma matrices need the Clifford algebra $\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu}$
applied first, and reverse under conjugation, which is genuinely more work
than the library currently implements. This means: **a Yukawa term written
without its explicit `+ h.c.` partner now genuinely fails
`Model.check_invariance()`'s hermiticity check** — the library will not
silently assume a Yukawa is self-conjugate the way a gauge current is.

## 4.4 Mass dimension

`check_mass_dimension` (`invariance.py:233`) does symbolic power counting
in mass units. It assigns dimension $3/2$ to every fermion component and
$1$ to every scalar/vector component and every VEV/fluctuation symbol,
reads parameter dimensions off `Parameter.unit_dim`, and injects a fresh
positive symbol $u$ (mass unit) via `term.xreplace({s: u**d for s, d in
dims.items()})` — but first rewrites `PartialMu(arg) → u*arg` (derivative
adds one power of mass) and collapses every opaque `Bilinear(bar, gamma,
field)` node to a bare $u^3$ (two spin-$\tfrac12$ legs, $\tfrac32+\tfrac32=3$
— the internal Dirac/flavor structure carries no mass dimension of its
own). The worst monomial's degree in $u$ (numerator minus denominator
degree, via `sp.degree`) is the term's mass dimension; the check passes iff
it's $\le 4$.

## Design gotchas summary

- **Never differentiate through `PartialMu`/`Bilinear`.** Both are custom
  `Function`s with no `fdiff`; every transform in this chapter is built to
  substitute *outside* them by exploiting a commutation property, never by
  calling `sp.diff` on an expression where they wrap a variable-dependent
  argument (see {doc}`lagrangian` for the full statement of this trap).
- **`as_coeff_Mul()` is the wrong tool for peeling a coefficient off an
  `Indexed` atom.** It defaults to `rational=True` and only pulls out
  *Rational* coefficients, silently leaving Dummy/Symbol factors (like the
  `alpha` from `_fermion_transform`) bundled into the "atom" side. Code that
  splits `coeff * Indexed(...)` (e.g. `expand_bilinear`'s
  `_split_indexed_term`, {doc}`vertices`) must instead find the single
  `Indexed` factor explicitly via `as_ordered_factors()`.
- **Discrete-multiplet assignment rejects mixed Fermion/non-Fermion
  members** (`DiscreteSymmetry.assign`, {doc}`declaration`) — a design
  consequence of the field-leg (`xreplace`) vs. fermion-leg (`.replace()`
  with index-preserving builder) split described above.

## Validation

- `tests/test_invariance.py::TestGaugeInvariance`,
  `::TestFermionGaugeInvariance` — the $O(\alpha)$ coefficient extraction,
  bosonic and fermionic.
- `tests/test_invariance.py::TestDiscreteInvariance`,
  `::TestFermionDiscreteInvariance` — finite-generator substitution,
  including the $X=(M^{-1})^T$ bar-leg case for both $S_3$ and $Z_N$.
- `tests/test_invariance.py::TestHermiticityAndDimension` — bosonic
  hermiticity and power counting.
- `tests/test_dirac.py` — the Clifford algebra and `dirac_conjugate`
  catalog directly.
- `tests/test_invariance.py::TestModelReport` — `Model.check_invariance()`
  end to end, including the `InvarianceReport` failure-collection contract.

## Minimal snippet

```python
report = model.check_invariance(hermiticity=True, dimension=True)
if not report.ok:
    report.raise_on_failure()   # or inspect report.failures directly
```
