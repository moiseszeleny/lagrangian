# 1. The Pipeline

## Physics statement

Deriving Feynman rules for a BSM Lagrangian is a fixed sequence of physics
operations, regardless of the model: write down the symmetric Lagrangian,
verify it really is symmetric, break the symmetry spontaneously, find the
mass eigenstates, and read off the interaction vertices in the mass basis.
`feynlag` takes that sequence literally and gives each step its own module.

## The seven stages

```{mermaid}
flowchart LR
    A["1. Declare<br/>fields, params, groups"] --> B["2. Write<br/>Lagrangian terms"]
    B --> C["3. Check<br/>invariance, hermiticity, dimension"]
    C --> D["4. Break<br/>VEV shift + tadpoles"]
    D --> E["5. Diagonalize<br/>mass matrices -> rotations"]
    E --> F["6. Extract<br/>vertices"]
    F --> G["7. Export<br/>LaTeX / UFO"]
```

| # | Stage | Module(s) | Manual chapter |
|---|-------|-----------|-----------------|
| 1 | Declare | `parameters.py`, `fields.py`, `groups/gauge.py`, `groups/discrete.py` | {doc}`declaration` |
| 2 | Write | `lagrangian.py`, `operators.py` | {doc}`lagrangian` |
| 3 | Check | `invariance.py` | {doc}`invariance` |
| 4 | Break | `vacuum/ewsb.py`, `vacuum/tadpoles.py` | {doc}`ssb` |
| 4b| Masses | `vacuum/masses.py` | {doc}`masses` |
| 5 | Diagonalize | `vacuum/diagonalize.py` | {doc}`diagonalization` |
| 6 | Extract | `vertices/extract.py`, `vertices/bilinear.py`, `vertices/yangmills.py`, `vertices/vertex.py` | {doc}`vertices` |
| 7 | Export | `export/latex.py`, `export/ufo/` | {doc}`export` |

`dirac.py` (Clifford algebra) and `verify/checks.py` (the dual
symbolic+numeric verification toolkit, {doc}`verification`) are used across
every stage rather than owning one.

Sitting *before* stage 2 is an optional exploration-branch helper,
{doc}`suggest` — instead of hand-writing the Lagrangian, it enumerates every
gauge/discrete-invariant term the declared field content admits, using the
stage-3 invariance machinery as an oracle.

## Orchestration: `Model` is a lazy pipeline

Every stage above is exposed as a method or cached property on
{class}`~feynlag.lagrangian.Model`, and **nothing is computed at
construction or at import time**. `Model.__init__` just stores its
arguments; `Model.vacuum`, `Model.tadpoles()`, `Model.mass_matrix(...)`,
`Model.physical_lagrangian()` and `Model.feynman_rules(...)` each compute on
first call and memoize in `Model._cache`, a plain dict cleared by
`Model._invalidate()`. The two state-mutating calls —
`Model.solve_tadpoles` and `Model.rotate` — both call `_invalidate()` before
returning, so any later `physical_lagrangian()` call recomputes downstream
of the new tadpole solution or rotation rather than serving a stale
memoized result.

This is a deliberate reaction to a **known flaw in the DLRSM1 reference
implementation** `feynlag` was built from: DLRSM1 computed pipeline results
as a side effect of module import, which made partial or reordered pipelines
silently return inconsistent state. Laziness plus explicit invalidation
means the `Model` object can always be inspected mid-pipeline (e.g. call
`check_invariance()` before ever touching EWSB) with no risk of a stale
cache masking a later change.

```python
m = Model("SM", ...)
m.check_invariance()          # stage 3, no caching needed
m.solve_tadpoles([mu2])       # stage 4 -- invalidates any cached
                               # physical_lagrangian
m.mass_matrix([...])          # stage 4b, computed on demand
m.rotate(weinberg_rotation)   # stage 5 -- invalidates again
m.feynman_rules([...])        # stage 6, built from the now-current
                               # physical_lagrangian
```

## Two representations, two tracks

Bosonic fields (scalars, gauge bosons) are plain commuting `sympy.Symbol`s.
Fermion fields are `sympy.IndexedBase`-typed (one flavor-indexed component
per gauge component) and every fermion bilinear `psibar Gamma psi` is
wrapped in the opaque `Bilinear(bar, gamma, field)` atom, which never
enters the commuting-symbol machinery. This split runs through every stage
from declaration (`fields.py`) to extraction (`vertices/extract.py` vs.
`vertices/bilinear.py`) and is covered in detail in {doc}`declaration` and
{doc}`vertices`.

## Validation

There is no single test for "the pipeline" -- `test_scalar_pipeline_sm.py`
(`test_lazy_pipeline_no_state_leak`) specifically pins the laziness/cache
invalidation contract described above.
