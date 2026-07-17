# 5. Spontaneous Symmetry Breaking

## Physics statement

A scalar potential invariant under the gauge/discrete symmetries can still
have a minimum away from the origin. Expanding the fields around that
minimum — the vacuum expectation value (VEV) — rather than around zero is
what turns gauge bosons and fermions massive and produces the physical
Higgs/Goldstone spectrum. `feynlag` splits this into two concerns: the
{class}`~feynlag.vacuum.ewsb.Vacuum` object (the VEV shift itself) and the
tadpole system (the *condition* that the shift point is actually a
stationary point of the potential).

## The VEV shift

`Scalar.expand_vev({component: vev})` ({doc}`declaration`) registers, per
scalar component, the CONVENTIONS.md-pinned expansion

$$
\phi^0 \;\to\; \frac{v + h + i\,a}{\sqrt2} \qquad \text{(complex component)}, \qquad
\phi \;\to\; v + \phi \qquad \text{(real component, no $1/\sqrt2$)},
$$

introducing two fresh real symbols `phi_r`, `phi_i` per complex component
(the CP-even and CP-odd fluctuations). `Vacuum` (`vacuum/ewsb.py:18`)
aggregates every scalar's `shift_map` into one substitution dict and
provides two operations built on it:

- `Vacuum.shift(expr)` — expand around the vacuum, **keeping** fluctuations
  (`xreplace` the shift map, then `sp.expand`). This is what turns the weak
  Lagrangian into the physical-basis Lagrangian's raw material.
- `Vacuum.at_vacuum(expr)` — evaluate strictly *at* the vacuum point: shift,
  then set every fluctuation to zero, **and** every scalar component that
  never received a VEV (e.g. a charged component, which cannot get one by
  charge conservation) to zero as well — both the component and its
  conjugate, so the potential really is evaluated at the true minimum
  candidate, not just the neutral-VEV slice of it.

## Tadpoles: algorithm

The tadpole conditions are $\{\text{vev}: \partial V/\partial\text{vev}
\vert_{\rm vacuum} = 0\}$ — the requirement that the vacuum point is a
stationary point. `extract_tadpoles` (`vacuum/tadpoles.py:13`) computes
this directly: `V0 = vacuum.at_vacuum(potential)`, then
`sp.diff(V0, vev)` for every registered VEV symbol (raising if a VEV isn't
a plain `Symbol` — a composite VEV expression can't be differentiated
against meaningfully).

### Derivation: why $\partial V/\partial\text{vev}$ *is* the tadpole coefficient

The tadpole condition is usually stated as "the coefficient of the
linear-in-fluctuation term vanishes" — i.e. the coefficient of $h$ (not
$v$) in $V(v+h+\ldots)$. `extract_tadpoles` instead differentiates $V$
evaluated at the vacuum **with respect to $v$ itself**. These give the same
answer by the chain rule: since $v$ and $h$ enter the shift **additively**
($\phi \to (v+h+ia)/\sqrt2$), $\partial/\partial v$ acting on any function of
$(v+h)$ and $\partial/\partial h$ acting on the same function are literally
the same differential operator up to which symbol survives the subsequent
$h\to0$ evaluation:

$$
\left.\frac{\partial V}{\partial v}\right|_{h=0} \;=\; \left.\frac{\partial V}{\partial h}\right|_{h=0}
$$

because both compute $V'(v+h)|_{h=0}$ for the same one-variable function
$V'$. Differentiating with respect to the VEV *symbol* directly — rather
than first extracting a linear-order Taylor coefficient in the fluctuation
— is therefore not an approximation or a different quantity; it is the
identical tadpole condition, computed the cheaper way (one `sp.diff` call
instead of a full Taylor expansion and coefficient extraction).

## Solving

`solve_tadpoles(potential, vacuum, for_params)` (`vacuum/tadpoles.py:31`)
turns the tadpole dict into `sp.Eq(..., 0)` equations and calls `sp.solve`
for the requested parameters. It raises rather than guessing whenever
`sp.solve` returns zero solutions (over-constrained/inconsistent system) or
more than one branch (under-determined choice of sign/root — the caller
must pick a branch manually and use `InternalParameter.define` directly).
On success, every `InternalParameter` among `for_params` gets its `expr`
defined with the solution — completing the declaration made in
{doc}`declaration` (an `InternalParameter` starts life with `expr=None` and
is only usable once some pipeline stage, most often this one, fills it in).

`Model.solve_tadpoles` (`lagrangian.py:175`) wraps this, remembers the
solution in `self._tadpole_solutions` (applied automatically by
`mass_matrix` and `physical_lagrangian`, {doc}`pipeline`), and invalidates
the `Model` cache.

## Worked case: the over-constrained 3HDM+S₃ system

`examples/thdm_s3.py` is deliberately built so the tadpole system has
*more* independent conditions than free VEV ratios — the S₃-symmetric
potential, when its tadpole equations are solved simultaneously, does not
admit an arbitrary vacuum: it **forces** a specific alignment,
$v_1 = \sqrt3\,v_2$ in the literature basis (components swap in feynlag's
real-orthogonal S₃ irrep basis — see {doc}`declaration`). This is not a
numerical accident of the benchmark point chosen; it is a structural
consequence of the S₃ Clebsch–Gordan structure of the potential terms
(`S3.doublet_product`) forcing the minimum to sit on a symmetry-preserving
submanifold of VEV space. `solve_tadpoles` reproduces this alignment
directly from the symbolic system, without it being hand-imposed.

## Validation

- `tests/test_scalar_pipeline_sm.py::test_tadpole_solution` — the SM Higgs
  case, $\mu^2 = \lambda v^2$.
- `tests/test_scalar_pipeline_thdm.py::test_tadpole_solutions_match_literature`
  — 2HDM tadpoles cross-checked against the standard Gunion–Haber form.
- `tests/test_thdm_s3.py::test_tadpole_alignment_sqrt3` — the forced
  $\sqrt3$ vacuum alignment described above.

## Minimal snippet

```python
model.check_invariance(raise_on_failure=True)
solution = model.solve_tadpoles([mu2])   # {mu2.symbol: lam.symbol * v.symbol**2}
```
