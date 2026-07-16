# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`feynlag` — a pure-SymPy library that derives tree-level Feynman rules from user-written BSM (Beyond Standard Model) Lagrangians. Input style is FeynRules-like: declare particle fields with gauge/discrete representations, write Lagrangian terms explicitly with library building blocks (`Dmu`, `dag`, `Bilinear`), and the library checks invariance, breaks the symmetry, diagonalizes, and extracts vertices.

## Commands

```bash
pip install -e .[dev]              # sympy + pytest + numpy + nbstripout, editable install
pytest                             # full suite (~130s)
pytest tests/test_invariance.py -q # single file
pytest tests/test_invariance.py::TestFermionGaugeInvariance::test_yukawa_gauge_invariant -q  # single test
python examples/sm_scalar_gauge.py # run a worked example end-to-end
jupyter nbconvert --to notebook --execute --inplace examples/SM_Feynman_Rules_Tutorial.ipynb  # re-execute the tutorial notebook after editing it
```

There is no linter/formatter configured. No build step beyond the editable install.

## Architecture

### Pipeline stage → module map

The library is organized around one linear physics pipeline; each stage lives in its own module and the top-level `Model` (in `lagrangian.py`) orchestrates all of them **lazily** (`functools`-style caching via `Model._cache`, invalidated on `solve_tadpoles`/`rotate` — nothing is computed at construction or at import time, deliberately, to avoid a known flaw in the DLRSM1 reference implementation this library was built from):

1. **Declare** — `parameters.py` (`ExternalParameter` fixed by experiment, `InternalParameter` derived later, `ParameterSet` topologically sorts internal dependencies), `fields.py` (`Scalar`/`Fermion`/`GaugeBoson`, auto-generated component symbols + conjugate registry), `groups/` (`gauge.py`: U1/SU2/SU3 with explicit generator matrices; `discrete.py`: ZN/S3 with Clebsch–Gordan products).
2. **Write** — `Lagrangian`/`LagrangianTerm` (`lagrangian.py`) collect sector-tagged terms (`kinetic`, `potential`, `yukawa`, `gauge`, `other`); `operators.py` provides `Dmu` (covariant derivative) and `PartialMu` (the `∂_μ` placeholder, resolved to momentum space later).
3. **Check** — `invariance.py`: gauge invariance by infinitesimal component transformation (`δφ = iα^aT^aφ`), discrete invariance by finite generator substitution, hermiticity, mass-dimension power counting. `Model.check_invariance()` is the entry point.
4. **Break the symmetry** — `vacuum/ewsb.py` (`Vacuum`: VEV shift maps `φ⁰→(v+h+ia)/√2`), `vacuum/tadpoles.py` (extract + solve, registers solutions on `InternalParameter`s), `vacuum/masses.py` (mass matrices: real, complex/charged via Dummy-conjugate trick, gauge-boson).
5. **Diagonalize** — `vacuum/diagonalize.py`: `Rotation` (weak↔physical substitution + verification), 2×2 analytic (`tan 2θ`), SVD (Dirac), Takagi (Majorana).
6. **Extract vertices** — `vertices/extract.py` (Poly-based commuting-symbol extractor, bosons only), `vertices/bilinear.py` (`Bilinear` — the fermion-sandwich track, separate from the extractor since fermions aren't commuting symbols), `vertices/yangmills.py` (cubic/quartic pure-gauge self-couplings from structure constants, not from the extractor), `vertices/vertex.py` (`Vertex` objects classified into the closed catalog SSS/SSSS/VSS/VVS/VVSS/VVV/VVVV/FFS/FFV).
7. **Export** — `export/latex.py`, `export/ufo/` (writer + UFO `object_library`/`lorentz_map`/`pycode` for MadGraph-importable model directories; `vvvv.py` assembles the 4-boson self-coupling into the 3 UFO Lorentz structures, and colored (SU(3)) vertices get real `T`/`f` color-tensor strings — see writer.py's `add_vvv_vertex`/`add_vvvv_vertex`/`add_fermion_vertex`).

`dirac.py` (Clifford algebra, chiral projectors `diracPL`/`diracPR`) and `verify/checks.py` (dual symbolic+numeric verification utilities, `numeric_equal`) are used across stages.

### The two-track extraction design (important, easy to get wrong)

Bosonic fields (scalars, gauge bosons) are represented as plain commuting `sympy.Symbol`s and go through the `Poly`-based extractor in `vertices/extract.py`. Fermion fields are `IndexedBase`-typed (one flavor-indexed component per generator), and a fermion sandwich `ψ̄Γψ` is wrapped in the opaque `Bilinear(bar_indexed, gamma, field_indexed)` atom — this **never** enters the commuting-symbol extractor. `vertices/bilinear.py`'s `extract_fermion_vertices` groups terms by `(bar, gamma, field)` and peels boson legs off the coefficient with the bosonic extractor.

Because `Bilinear` and `PartialMu` are custom SymPy `Function`s with no `fdiff` rule, code that needs to differentiate an expression containing one of them **must never let the differentiation variable end up inside their arguments** — SymPy's chain rule falls back to an unevaluated, non-cancelling `Subs(Derivative(...))` placeholder otherwise. This has caused two real bugs (both fixed): gauge-invariance checking of `Dmu`-built kinetic terms, and of fermion `Bilinear` terms. The fix pattern in both cases: exploit that the transform commutes with the operator (∂_μ or the flavor index) and substitute *outside*, or explicitly redistribute the operator over `Add` args first (see `_transform_map`/`_fermion_transform`/`_expand_bilinear` in `invariance.py`, and `D_linear` in `operators.py`) — never call `sp.diff` on an expression where `PartialMu`/`Bilinear` wraps a variable-dependent argument.

Related gotcha: `Expr.as_coeff_Mul()` defaults to `rational=True` and only pulls out *Rational* coefficients, silently leaving Dummy/Symbol factors bundled into the "atom" side. Any code splitting `coeff * atom` terms (as the bilinear-distribution logic does) must find the atom explicitly via `as_ordered_factors()`, not rely on `as_coeff_Mul()`.

**Fermion mass-basis rotations** work through the ordinary `Rotation`/`Model.rotate` machinery with `Indexed` old-fields (e.g. `Rotation([eL[i], EL[i]], [e1L[i], e2L[i]], rotation_2x2(θ))`) — register **two rotations per chirality** (field-side AND bar-side, same matrix), and write every Lagrangian term with the same flavor-index symbol so the `xreplace` keys match. The resulting `Add`-valued legs inside `Bilinear` slots are distributed by `expand_bilinear` (`vertices/bilinear.py`, moved from invariance.py), which `extract_fermion_vertices` and `fermion_mass_matrix` apply automatically — without it, extraction silently grouped by unsplit composite keys. For symbolic 2×2 Dirac (biunitary/SVD) diagonalization use `diagonalize_svd_2x2` (θ_L from `M·Mᵀ`, θ_R from `Mᵀ·M`, analytic tan-2θ route) — `diagonalize_svd`'s generic `.diagonalize()` path returns unusable nested radicals for symbolic entries. `DiracFermion` is untested and its `chirality=None` gauge current breaks `dirac_conjugate` — model vector-like fermions as two `WeylFermion`s with identical reps (see `examples/sm_vll.py`).

Fermion-bilinear **hermiticity** is checked via `Bilinear._eval_conjugate` (`vertices/bilinear.py`), which implements `(ψ̄₁Γψ₂)† = ψ̄₂Γ̄ψ₁` with `Γ̄ = γ⁰Γ†γ⁰` computed by `dirac.dirac_conjugate` (covers `diracI`, `diracPL`/`diracPR`, bare `DiracGamma(mu)`, and `DiracGamma(mu)*diracPL`/`diracPR` — raises `NotImplementedError` for anything else, e.g. products of ≥2 gamma matrices, rather than guessing) and a `bar_partner` registry (`fields.py`) that every `Fermion` populates at construction to find "the bar of this field"/"the field of this bar". A Yukawa term written without its `+ h.c.` partner now genuinely fails `Model.check_invariance()`'s hermiticity check; a gauge current (`fermion_gauge_current`) is hermitian by construction with no hand-written h.c. needed.

**Discrete-symmetry invariance for fermion multiplets** (`ZN`/`S3` acting on fermion generations — e.g. two lepton doublets forming an S₃ doublet, matching `thdm_s3.py`'s scalar S₃ pattern but for leptons) is also checked, via `DiscreteSymmetry.fermion_generator_data()` (`groups/discrete.py`) + `_fermion_transform_discrete` (`invariance.py`). Unlike the gauge case, discrete transformations are *finite* (no `α`-linearization/differentiation needed) — a multiplet member substitutes directly as `ψ'_i = Σ_k M[i,k]ψ_k`. The bar leg uses `X = (M⁻¹)ᵀ`, **not** `M` itself (derived from requiring `Σ_i ψ̄'_iψ'_i` invariant, verified by direct computation, not assumed) — `X` only coincides with `M` when `M` is real orthogonal (true for `S3`'s irreps, not `ZN`'s complex-phase ones, where `X = M̄`). `assign()` rejects multiplets mixing `Fermion` and non-`Fermion` members (they'd need different substitution mechanisms — flat dict vs. index-preserving `.replace()`).

### v2-deferred (not implemented, not attempted)

R_ξ gauge fixing and ghosts, four-fermion operators, UFO NLO extensions.

### SU(3) / QCD vertex dynamics

The group theory (`SU3` Gell-Mann generators/structure constants, `groups/gauge.py`) and the vertex-extraction machinery (`Dmu`, `fermion_gauge_current`, `cubic_couplings`/`quartic_couplings`) were already fully generic over gauge group — no SU(3)-specific code was needed there. What closed the gap: `examples/sm_scalar_gauge.py` now includes a quark sector (SU(2) doublet `QL` + `SU3c`-triplet singlets `uR`/`dR`, flavor-generic `Yu`/`Yd` Yukawas mirroring the lepton pattern — 3 generations, undiagonalized, no CKM: CKM mixing is physically orthogonal to SU(3) vertex dynamics and a fully generic 3-generation complex-Yukawa SVD doesn't resolve in closed symbolic form) and a gluon field; `export/ufo/vvvv.py`'s `assemble_vvvv` builds the 3-structure VVVV1/2/3 decomposition of the 4-boson self-coupling that `yangmills.py`'s `quartic_couplings()` docstring had deferred as unbuilt "Phase 5" (derived by direct functional differentiation of `-¼F^aF^a`, cross-checked against both SU(2) and SU(3) in `tests/test_yangmills.py`); `export/ufo/writer.py` emits real color-tensor strings (`T(3,1,2)` for qqg, `f(1,2,3)` for ggg, three `f*f` structures for gggg) instead of the hardcoded singlet `'1'` — gluons are exported as **one** UFO particle repeated (color summed via the tensor string), never as 8 separate weak-basis components.

### Conventions (all pinned by tests — see `CONVENTIONS.md` for the full list)

Metric `(+,−,−,−)`; `P_L=(1−γ₅)/2`; `D_μ=∂_μ−igT^aA^a_μ`; VEV expansion with explicit `1/√2`; Feynman rule = `i × coefficient × ∏(field multiplicity)!`; momentum convention `∂_μφ→ip(φ)φ`; simplification hierarchy `expand→collect→factor→simplify` (simplify last); positive-assumption dummies for anything under `sqrt`; every physical result gets dual verification (symbolic diff **and** `verify.numeric_equal` random-point check).

## Tests and examples

Tests are organized per-module/per-physics-topic (`test_invariance.py`, `test_fermion_sector.py`, `test_scalar_pipeline_sm.py`, `test_gauge_sector_sm.py`, `test_ufo_export.py`, `test_qcd.py`, `test_yangmills.py`, `test_ufo_qcd.py`, `test_vll.py`, etc.) and pin actual physics values (e.g. `hWW = igm_W`, `m_h²=2λv²`, `ggg=-g_s`), not just code paths — when adding a feature, add a test that pins the expected physical result, following the existing pattern in the relevant `test_*.py` file. `test_qcd.py` pins the SU(3) sector (gauge invariance of a color-triplet current, qqg, ggg, gggg); `test_yangmills.py` is the group-generic ground-truth derivation of the VVVV quartic self-coupling assembly (SU(2) and SU(3)); `test_ufo_qcd.py` checks the emitted UFO color-tensor strings/values; `test_vll.py` pins the vector-like-lepton mixing physics (RH-only Z-FCNC, h-coupling sum rule, SM decoupling); `test_u1x.py` pins the SM×U(1)_X / Z′ physics (symbolic-charge invariance, 3×3 neutral gauge matrix, tan 2θ′, Goldstone counting, Z′ff couplings, B−L limit).

`examples/` contains full worked models used as both documentation and manual smoke tests: `sm_scalar_gauge.py` (complete SM: Higgs + electroweak gauge + lepton sector + quark/QCD sector), `sm_vll.py` (SM + vector-like lepton doublet: bare Dirac mass, biunitary 2×2 diagonalization, fermion mass-basis rotations, modified Z/W/h couplings), `sm_u1x.py` (SM × U(1)_X with X = aY + b(B−L) kept **symbolic**: X-charged Higgs → tree-level Z–Z′ mixing via two *chained* 2×2 rotations — Weinberg first, then Z–Z′ consuming the intermediate `Z0` symbol, legal because `physical_lagrangian` applies rotations sequentially in registration order; singlet scalar + Higgs portal; 3 νR with Dirac Yukawa via H̃; no UFO export — UFO's particle table calls `float()` on charges), `thdm.py` (2HDM), `thdm_s3.py` (3HDM with S₃ flavor symmetry — the tadpole system there is deliberately over-constrained, forcing a vacuum alignment), and `SM_Feynman_Rules_Tutorial.ipynb` / `SM_VLL_Tutorial.ipynb` / `SM_U1X_Tutorial.ipynb` (all *executed* notebooks — regenerate their outputs with `jupyter nbconvert --execute` after any edit that could change results, don't hand-edit output cells; the VLL notebook walks `sm_vll.py` stage by stage, including a standalone demo of why `expand_bilinear` is needed for fermion mass-basis rotations).

The notebooks are tracked through a `nbstripout` git filter (`.gitattributes`, `*.ipynb filter=nbstripout diff=ipynb`), configured with `--keep-output` so real outputs (plots, printed Feynman rules) stay in git history and in diffs — only `execution_count` and volatile per-cell metadata get normalized away on `git add`/`git diff`, so re-executing a notebook without changing its actual results produces no diff noise. A fresh clone needs to activate the filter once (`.gitattributes` alone only declares the pattern, it doesn't install the filter driver):

```bash
nbstripout --install --attributes .gitattributes
git config filter.nbstripout.clean "$(git config filter.nbstripout.clean) --keep-output"
git config diff.ipynb.textconv "$(git config diff.ipynb.textconv) --keep-output"
```

(`nbstripout --install` itself does not persist CLI flags like `--keep-output` into the stored filter command — they have to be appended to `filter.nbstripout.clean`/`diff.ipynb.textconv` by hand, as above.)
