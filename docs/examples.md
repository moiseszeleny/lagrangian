# Examples Gallery

`examples/` contains full worked models, used as both documentation and
manual smoke tests. Each file is runnable end to end
(`python examples/<name>.py`) and doubles as the "how do I actually write
this in feynlag" reference for the corresponding physics.

## `sm_scalar_gauge.py` — the complete Standard Model

Higgs sector + electroweak gauge (SU(2)_L × U(1)_Y) + one lepton generation
+ a full 3-generation quark/QCD sector (SU(2) doublet `QL`, SU(3)_c-triplet
singlets `uR`/`dR`, flavor-generic Yukawas — no CKM diagonalization, since
CKM mixing is orthogonal to SU(3) vertex dynamics). Exercises every stage of
the pipeline: gauge invariance, EWSB, tadpoles, the Higgs/Goldstone/gauge
mass matrices, the Weinberg rotation, fermion gauge currents and Yukawa
mass matrices via the bilinear track, and both LaTeX and UFO export
(including real SU(3) color-tensor strings for qqg/ggg/gggg).

## `sm_vll.py` — vector-like lepton doublet

Adds a vector-like SU(2) doublet Ψ = (N, E) with a bare Dirac mass to the
lepton sector. Demonstrates biunitary 2×2 diagonalization
(`diagonalize_svd_2x2`) of the resulting charged-lepton mass matrix,
fermion mass-basis rotations through `Indexed` fields, and the physics
signature of doublet VLL mixing: a right-handed-only Z FCNC, an h-coupling
sum rule, and the correct SM-decoupling limit as M → ∞. See
{doc}`tutorials/index` for the step-by-step notebook.

## `sm_u1x.py` — SM × U(1)_X with a Z′

Extends the SM by a second abelian factor with symbolic charge assignment
`X = a·Y + b·(B−L)`. An X-charged Higgs singlet VEV drives tree-level
Z–Z′ mixing through two *chained* 2×2 rotations (Weinberg, then Z–Z′). Also
adds 3 right-handed neutrinos with a Dirac Yukawa via H̃. Demonstrates
keeping physics symbolic in an extra charge parameter, Goldstone counting
across two broken U(1)s, and multi-stage rotation chaining
(`Model.physical_lagrangian` applies registered rotations in registration
order). No UFO export — UFO's particle table calls `float()` on charges,
incompatible with the symbolic `a`/`b`.

## `thdm.py` — softly-broken Z₂ 2HDM

The scalar sector of a two-Higgs-doublet model with a softly-broken Z₂
symmetry. Demonstrates the analytic 2×2 orthogonal diagonalization
(`diagonalize_orthogonal_2x2`) and the α mixing angle, cross-checked
against the standard Gunion–Haber/Branco expressions.

## `thdm_s3.py` — 3HDM with an S₃ flavor symmetry

Three Higgs doublets, (H1, H2) forming an S₃ doublet and HS a singlet. The
potential is built entirely from the library's own S₃ Clebsch–Gordan
products (`S3.doublet_product`) rather than hand-written invariants — a
stress test of the discrete-symmetry machinery. The tadpole system is
deliberately over-constrained and forces a specific vacuum alignment
(v₁ = √3 v₂ in the literature basis).

Each model is described algorithmically, stage by stage, in the
{doc}`manual/pipeline`; the manual cites which pinned test in `tests/`
fixes each physical result these examples produce.
