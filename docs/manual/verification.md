# 10. Verification Philosophy

## Physics statement

A symbolic pipeline this long — invariance checking, tadpole solving, mass
diagonalization, vertex extraction, all chained together — has many places
where an algebra bug could silently produce a plausible-looking but wrong
result. `feynlag`'s house rule (CONVENTIONS.md) is that **every physical
result gets both a symbolic check and an independent random-point numeric
check**. This isn't redundancy for its own sake: the two checks fail in
different ways, so together they catch far more than either alone.

## Why `simplify()` alone isn't enough: the canonical-form problem

Two SymPy expressions that are mathematically identical can differ
*syntactically* in a way `sp.simplify()` fails to collapse to zero —
trigonometric identities, nested radicals, and nonlinear parameter
combinations are common culprits, and in general deciding whether two
elementary expressions are identically equal is not a fully solvable
problem (Richardson's theorem: equality of expressions built from the
elementary functions and radicals is undecidable in general). `simplify()`
is a good heuristic, not a decision procedure — it can, and in practice
occasionally does, fail to prove two genuinely equal expressions equal,
which would make a purely symbolic test suite either flaky (if it retries
with heavier simplification) or blind to certain bugs (if a `!= 0` check
is used to assert *inequality* and simplify happens to fail to reduce a
truly-zero expression, the test wrongly "passes" for the wrong reason).

`numeric_equal(expr_a, expr_b, symbols, n_points=20, tol=1e-10, ...)`
(`verify/checks.py:82`) sidesteps the whole problem: it `lambdify`s both
expressions (with `modules="mpmath"` for arbitrary-precision evaluation,
avoiding ordinary float cancellation/overflow issues at the sampled points)
and compares their **numeric values at `n_points` random points** in the
free symbols, to a tight relative tolerance. Two expressions that agree to
$10^{-10}$ relative precision at 20 independently random points are, for
all practical purposes, the same analytic function — this is not a formal
proof (a contrived pair of functions could theoretically agree densely
without being identical), but it is an *independent* source of confidence
from symbolic simplification, catching the overwhelming majority of real
algebra mistakes a `simplify()`-based test would miss, and vice versa (a
purely numeric check can't catch a systematic sign error that happens to
cancel at every sampled point, which is why the library requires *both*,
never one in place of the other).

## The rest of the verification toolkit

`verify/checks.py` collects the other reusable physics invariants used
throughout the pipeline:

- **`is_hermitian(M)` / `is_symmetric(M)`** — symbolic matrix checks
  (`M == M.conjugate().T` / `M == M.T` after `simplify`), used e.g. to
  confirm a mass matrix has the structure its diagonalization routine
  ({doc}`diagonalization`) assumes.
- **`check_dimension(expr, symbol_dims, target_dim)`** — a standalone,
  model-independent version of the mass-power-counting machinery in
  {doc}`invariance`'s `check_mass_dimension`, for verifying dimensional
  consistency of an arbitrary derived expression outside the `Model`
  context (e.g. a hand-derived formula being cross-checked against the
  library's output).
- **`seesaw_light_mass(M_D, M_R)`** — the leading-order type-I seesaw
  formula $m_\nu \approx -M_DM_R^{-1}M_D^T$, a common cross-check target
  for models with heavy right-handed neutrinos.
- **`decoupling_limit(expr, heavy_scale, direction="oo")`** — symbolic
  limit as a heavy mass scale goes to infinity (or zero), used to confirm
  a BSM extension reduces to the SM prediction when new physics is decoupled
  (e.g. `test_vll.py::test_sm_decoupling`, {doc}`diagonalization`).
- **`round_trip_reconstruct(lagrangian, fields)`** — extracts interaction
  coefficients ({doc}`vertices`) and rebuilds the Lagrangian from them,
  returning the residual `expand(original − reconstructed)`. This is a
  **self-consistency** check on the extractor itself, independent of any
  particular model's physics: it verifies that grouping-by-monomial and
  summing coefficients genuinely reconstructs what went in, with no terms
  silently dropped or double-counted. `ok` tolerates a nonzero residual
  only if that residual is entirely free of the field symbols (a pure
  parameter/constant term — e.g. a vacuum-energy constant — isn't an
  "interaction" and legitimately isn't reconstructed by an interaction
  extractor).

## Design gotcha: `numeric_equal`'s sampling domain

`sample_range` defaults to `(0.1, 10.0)` — positive reals bounded away from
zero. This is a reasonable default for couplings, masses, and VEVs (which
carry positivity assumptions throughout the library, CONVENTIONS.md), but
is the **wrong** default for an expression with a singularity or branch cut
inside that range, or one that genuinely needs negative/complex sample
points to exercise properly — callers comparing such expressions must pass
an adjusted `sample_range` explicitly rather than trusting the default
blindly.

## Pinned physical results and their tests

A non-exhaustive map from headline physical results to the test that pins
them — useful as a starting point when checking whether a change to the
library affects a known-correct physics prediction:

| Result | Test |
|---|---|
| $\mu^2 = \lambda v^2$ | `test_scalar_pipeline_sm.py::test_tadpole_solution` |
| $m_h^2 = 2\lambda v^2$ | `test_scalar_pipeline_sm.py::test_higgs_and_goldstone_masses` |
| $h^3=-3im_h^2/v$, $h^4=-3im_h^2/v^2$ | `test_scalar_pipeline_sm.py::test_higgs_self_couplings_pinned` |
| $m_W = gv/2$, Weinberg rotation | `test_gauge_sector_sm.py::test_gauge_mass_matrix`, `::test_weinberg_rotation` |
| $hWW = igm_Wg^{\mu\nu}$ | `test_gauge_sector_sm.py::test_hWW_coupling` |
| $\gamma W^+W^- = e$, $ZW^+W^- = g\cos\theta_W$ | `test_gauge_sector_sm.py::test_cubic_gauge_couplings` |
| $h\ell\ell = -im_\ell/v$ | `test_fermion_sector.py::TestYukawa` |
| $W\ell\nu = ig/\sqrt2\,\gamma^\mu P_L$, $Z$ couplings $\propto T^3-Q\sin^2\theta_W$ | `test_fermion_sector.py::TestGaugeCurrents` |
| 2HDM tadpoles/masses vs. Gunion–Haber | `test_scalar_pipeline_thdm.py` (whole file) |
| 3HDM+S₃ forced $\sqrt3$ alignment | `test_thdm_s3.py::test_tadpole_alignment_sqrt3` |
| $qqg$: `T(3,1,2)`, $ggg = -g_s$, $gggg$ | `test_qcd.py` (whole file), `test_ufo_qcd.py` |
| VVVV assembly (SU(2) and SU(3)) | `test_yangmills.py::TestVVVVAssemblyGroundTruth` |
| VLL: right-handed-only Z FCNC, h-coupling sum rule, SM decoupling | `test_vll.py` (whole file) |
| Z–Z′ mixing, tan 2θ′, Goldstone counting, B−L limit | `test_u1x.py` (whole file) |
| UFO import + parameter resolution + `hWW` numeric | `test_ufo_export.py` |

Every chapter of this manual cites the specific tests relevant to its own
derivations in its own **Validation** section; this table is a
cross-cutting index by physical result rather than by pipeline stage.

## Minimal snippet

```python
from feynlag import numeric_equal

ok, max_diff = numeric_equal(hWW_extracted, i * g * m_W_expr, [g, v])
assert ok, f"symbolic and numeric routes disagree: {max_diff}"
```

## Round-tripping the exported UFO

Internal verification proves the *symbolic* result is right; it does not prove
the *written UFO directory* is right — a writer bug can emit a coupling string
that references an undefined symbol or divides by zero at run time. Traditional
generators leave this to the author (the well-known irregular-validation
problem). `verify_ufo_numeric` closes the gap by re-importing the written UFO,
resolving the whole parameter chain, and evaluating every coupling string:

```python
from feynlag import verify_ufo_numeric

write_ufo(path, "MyModel", params, particles, bosonic_vertices=verts)
report = verify_ufo_numeric(path)
assert report.ok, report.failures        # every coupling is a finite number
```

A non-finite input (or a malformed generated expression) is reported in
`report.failures` rather than propagating as a silent `NaN`. Pinned in
`test_ufo_export.py::test_ufo_roundtrip_evaluates_cleanly` /
`::test_ufo_roundtrip_flags_nonfinite`.
