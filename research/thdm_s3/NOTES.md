# 3HDM-S₃ — research log

The S₃-symmetric three-Higgs-doublet model: `(H1, H2)` an S₃ doublet, `HS` the
singlet. The symbolic build lives in `examples/thdm_s3.py` (potential only,
weak-basis CP-even mass matrix) and `examples/THDM_S3_Tutorial.ipynb` (the
pedagogical walk-through). This directory is where the open questions go.

## Layout

| file | what it is |
|---|---|
| `model.py` | the reusable build — algebra only. `build_model()`, the three mass matrices, the geometric rotation, the lambdified spectrum. Lifts the tutorial's notebook-local §6–§8 into importable form. |
| `constraints.py` | numpy-vectorized boundedness-from-below + tree-unitarity conditions, quoted from [DasDey14]. |
| `01_scalar_parameter_space.ipynb` | scalar parameter space under theory constraints. **Done.** |
| `results/viable_points.json` | benchmark points from that scan. |

Notebooks import `model.py` from their own directory; from the repo root use
`sys.path.insert(0, "research/thdm_s3")`.

**Kernel note.** These notebooks are executed with the plain `python3` kernel,
not the repo's `lagrangian` kernel, because the `lagrangian` conda env has no
`numpy`/`matplotlib` (the scan and the figures need both). To use the standard
kernel instead, install the dev extras into it:
`/home/moises/miniconda3/envs/lagrangian/bin/pip install -e ".[dev]"`.

## Findings

### 1. The tutorial benchmark sits at the wrong electroweak scale

`examples/THDM_S3_Tutorial.ipynb` picks $(v_1,v_2,v_S)=(200,115,80)$ GeV
precisely so that $\sqrt{\sum v_i^2}\approx246$ GeV ([GomezBock21] Eq. 8,
stated in its md cell 5). It then imposes the S₃ alignment as the substitution
`align = {v1: v2/sqrt(3)}`, which **replaces** $v_1=200$ by $115/\sqrt3=66.4$
and never re-checks the scale. The aligned vacuum is $v=155.03$ GeV.

Consequence: at fixed λ and θ every mass² is homogeneous of degree 2 in the
VEVs, so the §9 spectrum is uniformly low by $(246/155.03)^2 = 2.518$ in mass²,
i.e. **every mass in GeV is low by a factor 1.587**. The *qualitative*
conclusion there ("5 of 7 states tachyonic") is unaffected.

Fix: impose both conditions at once. They leave one free parameter, the same
angle θ the geometric rotation uses ([GomezBock21] Eq. 24): $v_{12}=v\sin θ$,
$v_S=v\cos θ$, and the alignment then fixes $v_1=v_{12}/2$,
$v_2=\sqrt3\,v_{12}/2$ exactly. Implemented as `model.aligned_vevs`, pinned by
`tests/test_thdm_s3.py::test_aligned_vacuum_can_meet_the_electroweak_scale`.

**Not yet propagated to the tutorial.** Correcting `examples/THDM_S3_Tutorial.ipynb`
§9 is a separate, small follow-up (re-execute + commit).

### 2. feynlag's masses reproduce the published closed forms — and fix the dictionary

The λ basis is *not* trivially the literature's. The λ₄ invariant transcribed
literally from [DasDey14] into feynlag's component labels **fails**
`check_discrete_invariance` against S₃ — the doublet irrep bases differ.

Rather than argue conventions, the dictionary was derived from physics:
requiring feynlag's independently-computed pseudoscalar and charged masses to
equal [GomezBock21]'s closed forms Eqs. (30)–(33). All four match **exactly**,
and only for

$$a=2\lambda_8,\; b=\lambda_5,\; c=2\lambda_1,\; d=2\lambda_2,\;
e=-\lambda_4,\; f=\lambda_6,\; g=2\lambda_3,\; h=2\lambda_7 .$$

Composing with the [GomezBock21]↔[DasDey14] correspondence gives
$\lambda_k^{\rm DD}=\lambda_k^{\rm FL}$ for all $k\neq4$ and
$\lambda_4^{\rm DD}=-\lambda_4^{\rm FL}$. Since λ₄ enters the theory conditions
only as $|\lambda_4|$ or $\lambda_4^2$, they transfer unchanged.

This doubles as an end-to-end validation of feynlag's 3HDM-S₃ chain — potential,
tadpoles, all three mass matrices and the rotation — against published closed
forms. Pinned by `tests/test_thdm_s3.py::test_masses_match_gomezbock_closed_forms`.

### 3. The tutorial's benchmark λ's are not an admissible theory

At $\lambda_k=0.05k$ the potential is **not bounded from below** ([DasDey14]
Eq. 4 fails). "5 of 7 tachyonic" was the symptom; the point was never a valid
theory. Tree unitarity is comfortably satisfied there (largest eigenvalue 3.63
against the 16π ≈ 50.3 bound).

### 4. Boundedness, not unitarity, is the binding constraint

Over $\lambda\in U(-2,2)^8$: boundedness keeps 5.7% of points, unitarity then
removes **none** of them. Tree unitarity only bites at $|\lambda|\gtrsim$ several.
Within boundedness, Eq. (4g) and Eq. (4d)/(4f) do most of the rejecting.

Of 2 000 000 sampled points: 114 293 bounded+unitary → 7 927 tachyon-free →
7 196 with $m_{H^\pm}>80$ GeV → 334 with a CP-even state at $125\pm3$ GeV.
Viable spectra cluster sub-TeV, consistent with [DasDey14]'s own conclusion
that "many new scalars must be lurking below 1 TeV".

## Open questions / next

- **The 125 GeV cut is a mass condition only.** Testing [GomezBock21]'s
  alignment scenarios A/B (Eqs. 54–55; one CP-even state coupling maximally to
  $W/Z$, the other decoupled) needs the **gauge–scalar couplings** of their
  Eq. (49). `examples/thdm_s3.py` declares *only* the potential — no kinetic
  terms, no gauge bosons in the Lagrangian — so those couplings do not exist to
  be computed. Blocked on adding the electroweak kinetic sector.
- **`02_s3_fermion_sector.ipynb`** — S₃ leptons/quarks. The machinery exists and
  is unused by any 3HDM example: `S3.fermion_generator_data()`
  (`groups/discrete.py`) + `_fermion_transform_discrete` (`invariance.py`), with
  the bar leg transforming as $X=(M^{-1})^{\mathsf T}$ (= $M$ here, S₃'s irreps
  being real orthogonal). **Blocker:** the S₃ irrep assignment for the three
  generations is a physics choice that must come from the literature —
  [BabuWuXu23] is the closest published S₃-3HDM Yukawa treatment. Literature
  pass first, code second.
- **`03_scalar_decays.ipynb`** — widths/BRs for $h_0,H_1,H_2,A_1,A_2,H^\pm$.
  **Blocker:** the physical basis is never registered on the `Model`. The
  geometric rotation is a loose `sp.Matrix`; to get physical-basis vertices it
  must become a registered `Rotation` (3×3 weak→physical per sector) so
  `model.rotate` + `feynman_rules` apply — the pattern `examples/thdm.py`
  uses for the 2HDM's 2×2 case. Also depends on the fermion sector, since
  `pheno.DecayCalculator` needs fermions to decay into (FFS), and on the gauge
  kinetic sector for $h\to VV$ (VVS).
- The scan samples λ uniformly, which is inefficient given the shape of the
  viable region. A targeted sampler would resolve its boundary far better.

## References

- **[GomezBock21]** M. Gómez-Bock, M. Mondragón, A. Pérez-Martínez, *"Scalar and
  gauge sectors in the 3-Higgs Doublet Model under the S₃-symmetry"*,
  Eur. Phys. J. C **81**, 942 (2021),
  [arXiv:2102.02800](https://arxiv.org/abs/2102.02800),
  [doi:10.1140/epjc/s10052-021-09731-3](https://doi.org/10.1140/epjc/s10052-021-09731-3).
  Eq. (2)/(6) the $(a,\ldots,h)$ potential; Eq. (8) the $v=246$ GeV constraint;
  Eq. (13) the √3 alignment; Eq. (24) the θ parametrization; Eq. (29) the
  geometric rotation; Eqs. (30)–(33) the pseudoscalar/charged closed-form
  masses; Eqs. (54)–(55) the alignment scenarios A/B. Also cited in
  `docs/manual/ssb.md`.
- **[DasDey14]** D. Das, U. K. Dey, *"Analysis of an extended scalar sector with
  S₃ symmetry"*, Phys. Rev. D **89**, 095025 (2014),
  [arXiv:1404.2491](https://arxiv.org/abs/1404.2491),
  [doi:10.1103/PhysRevD.89.095025](https://doi.org/10.1103/PhysRevD.89.095025).
  Eq. (3c) the λ₁…λ₈ potential; Eq. (4a)–(4g) boundedness-from-below;
  Eq. (36) + (37a)–(37l) the tree-unitarity eigenvalues. This is the actual
  source of the constraints — [GomezBock21] §2.2 delegates to it.
- **[BabuWuXu23]** K. S. Babu, Y. Wu, S. Xu, *"Fermion Masses, Neutrino Mixing
  and Higgs-Mediated Flavor Violation in 3HDM with $S_3$ Permutation
  Symmetry"*, [arXiv:2312.15828](https://arxiv.org/abs/2312.15828) (2023).
  The fermion-sector reference for notebook 02 — not yet read in detail.
- **[Yildirim26]** E. Yildirim, *"Double SM-like Higgs Production at future
  $e^+e^-$ colliders in the 3-Higgs Doublet Model under the $S_3$ symmetry"*,
  [arXiv:2604.24421](https://arxiv.org/abs/2604.24421) (2026). Applies
  perturbative unitarity, vacuum stability and LHC/Tevatron data to this same
  model; a cross-check target for the constraint implementation.
