# 3. Writing the Lagrangian

## Physics statement

The declared fields are plain algebraic symbols with no spacetime
dependence — `H_1` is just a SymPy `Symbol`, not a function of $x^\mu$.
That's exactly what the vertex extractor needs ({doc}`vertices`), but it
means `∂_μ φ` cannot be a literal SymPy derivative: there is nothing to
differentiate *with respect to*. `feynlag` instead represents `∂_μ` as an
**opaque placeholder operator**, `PartialMu`, that is carried symbolically
through gauge-invariance checking and only resolved to a momentum tag
(`∂_μφ → ip(φ)φ`) at the very end, right before vertex extraction. Writing
a Lagrangian term is therefore: build it out of field symbols, `Dmu`,
`dag`, and ordinary SymPy arithmetic, and tag it with a physics *sector*.

## Sectors

`Lagrangian` (`lagrangian.py:52`) is a flat list of `LagrangianTerm(expr,
sector, name)`, where `sector` is one of `kinetic`, `gauge`, `potential`,
`yukawa`, `other`. Sectors matter downstream: `Model.potential` reads only
the `potential` sector (with a sign flip, since the Lagrangian stores
$\mathcal L \supset -V$), `check_invariance`'s hermiticity check runs
per-sector, and `physical_lagrangian(sector=...)` lets vertex extraction
target one sector at a time. Sectors are otherwise inert bookkeeping — the
invariance and extraction machinery treats the whole Lagrangian uniformly.

## `PartialMu`: the derivative placeholder

`PartialMu` (`operators.py:22`) is a `sympy.Function` subclass that
deliberately **stays unevaluated on construction** (`eval` returns `None`
except for the trivial constant case) and only resolves via an explicit
`.doit()`:

```python
class PartialMu(Function):
    def doit(self, **hints):
        field = self.args[0]
        return I * momentum(field) * field
```

This is the symbolic realization of the momentum-space rule pinned in
`CONVENTIONS.md`: for a field carrying momentum $p(\phi)$ flowing *with*
it into the vertex, $\partial_\mu\phi \to i\,p(\phi)\,\phi$. Deferring this
substitution — rather than applying it the moment a term is written — is
what lets `check_invariance` ({doc}`invariance`) operate on the *same*
symbolic `PartialMu(comp)` atoms that appear in a `Dmu`-built kinetic term,
transforming them exactly like their underlying field.

## `Dmu`: the covariant derivative

`Dmu(field, gauge_groups=None)` (`operators.py:127`) builds the covariant
derivative as a column `Matrix` over the field's components, implementing
the pinned sign convention $D_\mu = \partial_\mu - igT^aA^a_\mu$:

$$
(D_\mu\phi)_i \;=\; \partial_\mu\phi_i \;-\; i\sum_G g_G\sum_a A^{a}_\mu\,(T^a\phi)_i
$$

summed over every gauge group $G$ the field is charged under (default: all
of `field.reps`). Each term is built directly from the data assembled in
{doc}`declaration`: `group.g` (the coupling symbol), `group.bosons()`
(created on first access — `groups/base.py`'s `_gauge_bosons` memoizes one
`GaugeBoson` per group), and `field.generators(group)` (the Kronecker-built
generator matrices). The Lorentz index $\mu$ itself is never represented
explicitly; it's implicit in the pairing between a `PartialMu`/momentum tag
and the vertex classifier reconstructing Lorentz structure from the spin
content later ({doc}`vertices`).

## Algorithm: Leibniz-expanding `PartialMu` over products (`D_linear`)

A hand-written or `Dmu`-built term can contain `PartialMu` wrapping a
*product* of several fields (this happens naturally once gauge-invariance
checking substitutes `comp → comp + iαTφ` inside an existing
`PartialMu(comp)`, or whenever a term is written as
`PartialMu(phi1 * phi2)` directly). Before that can be pushed to momentum
space, it must be expanded via the ordinary product (Leibniz) rule:

$$
\partial_\mu(\phi_1\phi_2\cdots\phi_n) \;=\; \sum_{k=1}^n \phi_1\cdots(\partial_\mu\phi_k)\cdots\phi_n .
$$

`D_linear(expr, fields)` (`operators.py:46`) implements this recursively
over the argument of an (unwrapped) `PartialMu` head:

1. **`Add`**: distribute over the sum — $\partial_\mu(a+b) = \partial_\mu a + \partial_\mu b$ — by recursing on each summand.
2. **`Pow`** (`φ^n`, integer `n>0`): apply the single-variable chain rule directly, $\partial_\mu(\phi^n) = n\phi^{n-1}\partial_\mu\phi$, rather than expanding to `n` Leibniz terms.
3. **`Mul`**: split each factor into "field-valued" (in `fields`, or an integer power of one) vs. "constant" (couplings, `I`, everything else); for `k` field factors, emit `k` terms, each with exactly one factor wrapped in `PartialMu` and the rest — including the non-field coefficient — left untouched.
4. **Bare field**: `PartialMu(expr)` directly.
5. **Anything else** (a pure parameter/VEV expression, once fluctuations are set to their vacuum value): $\partial_\mu(\text{const}) = 0$.

Step 5 is a deliberate fix over the DLRSM1 original, which returned the
expression *unchanged* in this branch — a latent bug that only manifests
once VEV-shifted arguments reach `D_linear` (a constant term wrongly kept
alive as if it still carried a derivative).

`expand_derivatives(expr, fields)` applies `D_linear` to every `PartialMu`
node found by `expr.replace(PartialMu, ...)`; `to_momentum_space(expr,
fields)` chains that with the final `∂_μφ → ip(φ)φ` substitution, dropping
derivatives of anything that doesn't depend on `fields` (constants, at this
final stage, contribute nothing).

## Design gotcha: no `fdiff`, by design

`PartialMu` and `Bilinear` ({doc}`vertices`) are custom `Function`s with
**no `fdiff` rule**. This is not an oversight — the two-track design means
neither operator has a meaningful ordinary derivative in the sense SymPy
expects (`PartialMu` isn't a function of a continuous variable at all in
this framework; `Bilinear` is a fermion sandwich, not a differentiable
scalar function). The consequence, stated precisely because it has caused
two real bugs (both fixed, see {doc}`invariance`): **any code that
differentiates an expression containing `PartialMu` or `Bilinear` must
never let the differentiation variable end up inside their arguments** —
SymPy's chain rule falls back silently to an unevaluated, non-cancelling
`Subs(Derivative(...))` placeholder rather than raising. The fix pattern
used throughout the library is to exploit that the relevant transformation
*commutes* with the operator and substitute **outside** it (`_transform_map`
in `invariance.py`), or to explicitly redistribute the operator over `Add`
first (`D_linear`, `_expand_bilinear`) before any differentiation happens.

## Validation

- `tests/test_operators.py::test_leibniz_two_fields`,
  `::test_leibniz_with_coefficient`, `::test_linearity_over_sums`,
  `::test_single_field_with_power` — the `D_linear` cases above.
- `tests/test_operators.py::test_kinetic_term_momentum_space` — a full
  `Dmu`-built kinetic term through `to_momentum_space`.

## Minimal snippet

```python
from feynlag import Dmu, dag, Lagrangian

DH = Dmu(H)                       # column Matrix, one PartialMu-tagged
                                   # entry per component (see declaration.md)
L = Lagrangian()
L.add((dag(DH) * DH)[0], sector="kinetic")   # (D_mu H)^dagger (D^mu H)
```
