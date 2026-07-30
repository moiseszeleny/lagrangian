# 3HDM-S₃ — research log

The S₃-symmetric three-Higgs-doublet model: `(H1, H2)` an S₃ doublet, `HS` the
singlet. The symbolic build lives in `examples/thdm_s3.py` (potential only,
weak-basis CP-even mass matrix) and `examples/THDM_S3_Tutorial.ipynb` (the
pedagogical walk-through). This directory is where the open questions go.

## Layout

| file | what it is |
|---|---|
| `model.py` | the reusable build — algebra only. `build_model()` (with `soft=True` for soft S₃ breaking), the three mass matrices, the geometric rotation, the lambdified spectrum, the quartic potential + its numerical boundedness scan. Lifts the tutorial's notebook-local §6–§8 into importable form. |
| `constraints.py` | numpy-vectorized boundedness-from-below + tree-unitarity conditions, quoted from [DasDey14] — **plus** the corrected neutral-direction condition derived in finding 4 below. |
| `fermions.py` | the S₃ fermion sectors — basis map to [LFVHD], the five Yukawa structures, mass-matrix extraction, the two-stage diagonalization, the $G_k$/$Q_i$ LFV couplings. |
| `LFVHD_3HDMS3.tex` | the draft this work checks against; **patched** (findings 6–8). |
| `01_scalar_parameter_space.ipynb` | scalar parameter space under theory constraints. **Done.** |
| `02_s3_fermion_sector.ipynb` | lepton + quark Yukawa sectors, the draft's derivations, the CKM obstruction, soft breaking. **Done.** |
| `results/viable_points.json` | benchmark points from the scalar scan. |
| `results/quark_soft_fit.json` | soft-breaking quark benchmark (masses + Cabibbo angle). |

Notebooks import `model.py` from their own directory; from the repo root use
`sys.path.insert(0, "research/thdm_s3")`.

**Kernel note.** These notebooks are executed with the plain `python3` kernel,
not the repo's `lagrangian` kernel, because the `lagrangian` conda env has no
`numpy`/`matplotlib` (the scan and the figures need both). To use the standard
kernel instead, install the dev extras into it:
`/home/moises/miniconda3/envs/lagrangian/bin/pip install -e ".[dev]"`.

Re-execute with the **miniconda** nbconvert, not the one first on `PATH`
(`~/.local/bin/jupyter-nbconvert` runs under a different Python and dies on a
missing `packaging`/`dateutil`):

```bash
/home/moises/miniconda3/bin/jupyter-nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 \
  research/thdm_s3/01_scalar_parameter_space.ipynb
```

**Memory note.** The box has ~7 GB and often <2 GB free; a 2 M-point scan that
materializes full-length temporaries gets OOM-killed (silently — nbconvert dies
with no message and leaves the notebook untouched). `constraints.py` therefore
applies every heavy array function in blocks (`_chunked`, `CHUNK = 250_000`),
which holds peak RSS to ~285 MB for all masks over 2 M points.

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

The dictionary is established in two steps, neither of them assumed.

**Seven entries by basis-independence.** Under a general $SO(2)$ rotation of the
S₃ doublet, the singlet contraction $(s_1,s_2)$ rotates by α while $d_2$ rotates
by 2α. Testing all eight quartic structures symbolically: **seven are invariant**
and only $s\cdot d_2$ (the λ₄ structure) can change. Invariant structures are the
same object in either basis, so their coefficients transfer by term matching:
$a=2\lambda_8$, $b=\lambda_5$, $c=2\lambda_1$, $d=2\lambda_2$, $f=\lambda_6$,
$g=2\lambda_3$, $h=2\lambda_7$.

**The last entry — λ₄'s sign — from physics.** The only S₃-preserving
transformation that changes $s\cdot d_2$ is $\alpha=\pi$, i.e. $H_{1,2}\to-H_{1,2}$,
which flips it and leaves the other seven alone. Which sign applies is settled by
requiring feynlag's independently-computed pseudoscalar and charged masses to equal
[GomezBock21]'s closed forms Eqs. (30)–(33). All four match **exactly**, and only
for $e=-\lambda_4$.

Composing with the [GomezBock21]↔[DasDey14] correspondence gives
$\lambda_k^{\rm DD}=\lambda_k^{\rm FL}$ for all $k\neq4$ and
$\lambda_4^{\rm DD}=-\lambda_4^{\rm FL}$. Since the flip is a *field
redefinition*, no physical condition may depend on it — and indeed λ₄ enters the
theory conditions only as $|\lambda_4|$ or $\lambda_4^2$, so they transfer
unchanged. That is a consequence, not a coincidence.

This doubles as an end-to-end validation of feynlag's 3HDM-S₃ chain — potential,
tadpoles, all three mass matrices and the rotation — against published closed
forms. Pinned by `tests/test_thdm_s3.py::test_masses_match_gomezbock_closed_forms`.

### 3. The tutorial's benchmark λ's are not an admissible theory

At $\lambda_k=0.05k$ the potential is **not bounded from below** ([DasDey14]
Eq. 4 fails). "5 of 7 tachyonic" was the symptom; the point was never a valid
theory. Tree unitarity is comfortably satisfied there (largest eigenvalue 3.63
against the 16π ≈ 50.3 bound).

### 4. The published boundedness conditions are not sufficient

**The most substantive finding.** Boundedness from below is directly testable —
$V_4$ is homogeneous of degree 4, so the potential is bounded iff $V_4\ge0$ on the
unit sphere in field space. Testing [DasDey14] Eq. (4) against feynlag's own $V_4$
(12 real dof, extracted by `model.quartic_potential`) produces **explicit
counterexamples**: λ points Eq. (4) accepts that have a real *neutral* direction
along which $V_4<0$. Over the 2 M-point scan the corrected condition removes
**9.35%** of everything Eq. (4) admits.

The cause is exact. On the real neutral slice $(H_1^0,H_2^0,H_S^0)=(r_1,r_2,r_S)$,

$$V_4=(\lambda_1+\lambda_3)x^2+(\lambda_5+\lambda_6+2\lambda_7)xy+\lambda_8y^2
+2\lambda_4 r_S r_1(r_1^2-3r_2^2),\quad x=r_1^2+r_2^2,\ y=r_S^2,$$

and the last term is the S₃ cubic covariant $2\lambda_4\sqrt y\,x^{3/2}\cos3\psi$,
worst case $-2|\lambda_4|x^{3/2}\sqrt y$. With $x=1$, $t=\sqrt y$ the **exact**
condition on this slice is

$$f(t)=\lambda_8t^4+(\lambda_5+\lambda_6+2\lambda_7)t^2-2|\lambda_4|t+(\lambda_1+\lambda_3)\ \ge 0
\quad\text{for all } t\ge0,$$

and **[DasDey14] Eq. (4g) is precisely $f(1)>0$** — one point of that curve.
Necessary, not sufficient. Implemented as `constraints.neutral_real_bfb_min`
(closed-form minimisation via batched companion matrices) and
`strict_bfb_mask`; validated against brute-force direction sampling on the same
slice (0 disagreements). Reproduced in `01_scalar_parameter_space.ipynb` §5.3–§5.5.

Still only *necessary* overall — it covers the real neutral slice, not
complex-neutral or charged directions. So the notebook also runs
`model.numeric_bfb_min` over the full 12-dimensional space as an empirical
backstop on the quoted benchmark points; it removes a further 3 of 322 (§8),
confirming the residual gap is real but small once the other cuts are applied.

**Unresolved:** whether the [DasDey14] Erratum (Phys. Rev. D 91, 039905, 2015 —
not on arXiv, paywalled) already corrects this. The insufficiency of the arXiv-v2
conditions is established regardless.

### 5. Boundedness, not unitarity, is the binding constraint

Over $\lambda\in U(-2,2)^8$: boundedness keeps ~5.7% of points (and the corrected
condition removes ~10% of those), unitarity then removes **none**. Tree unitarity
only bites at $|\lambda|\gtrsim$ several. Within boundedness, Eq. (4g) and
Eq. (4d)/(4f) do most of the rejecting.

Viable spectra cluster sub-TeV, consistent with [DasDey14]'s own conclusion
that "many new scalars must be lurking below 1 TeV". Current cut-flow numbers
live in `results/viable_points.json` (`cut_flow`), regenerated by the notebook.

### 6. The draft's S₃ basis differs from feynlag's — but the couplings map 1:1

`LFVHD_3HDMS3.tex` uses $a=R(-2\pi/3)$, $b=$ reflection about 30°; feynlag uses
$\rho(a)=R(2\pi/3)$, $\rho(b)=\mathrm{diag}(1,-1)$. They are conjugate by a unique
O(2) **reflection at $\pi/6$** (`fermions.tex_basis_map`, verified not assumed);
$\det=-1$, which is why the draft's order-3 generator is feynlag's inverse. The
visible symptom is the alignment: $v_1=\sqrt3v_2$ in the draft, $v_2=\sqrt3v_1$
in feynlag.

Four of the five Yukawa structures are *dot products* of two S₃ doublets and so
are invariant under any orthogonal basis change; only the triple-doublet
structure ($Y_2$) is basis-sensitive — the fermionic analogue of $\lambda_4$
(finding 2). Under the reflection the draft's $Y_2$ maps onto feynlag's with
coefficient $+1$, so **the coupling dictionary is the identity** (unlike the
scalar case, where $\lambda_4$ flipped sign).

### 7. The draft's charged-lepton Yukawa sector is invariant and complete

All five terms pass gauge and S₃ invariance in feynlag. `suggest_yukawa`,
enumerating independently, returns exactly five structures — matching
$Y_1^\ell\ldots Y_5^\ell$. Character theory agrees: the trivial rep appears once
in $\mathbf2^{\otimes3}$, and over the eight $(\bar L,H,e_R)$ irrep assignments
exactly five invariants exist.

**Gap:** the draft declares $\nu_{1R},\nu_{2R}$ (a $\mathbf2$) and $\nu_{SR}$ (a
$\mathbf1$) and never uses them. Those fields permit further S₃- and
gauge-invariant operators (Dirac structures on $\tilde H$, plus bare Majorana
masses, the $\nu_R$ being gauge singlets). Enumerated in
`02_s3_fermion_sector.ipynb` §8; not built.

### 8. Three defects in the draft — found, corrected, and patched into the `.tex`

1. **The $\mu\leftrightarrow Y$ dictionary is a uniform factor $\sqrt2$ too
   large.** With the draft's own $\langle H_i^0\rangle=v_i/\sqrt2$ and its
   explicit prefactors, the Lagrangian gives $\mu_1^\ell=Y_1^\ell v_3/2$,
   $\mu_2^\ell=Y_2^\ell v_2/2$, $\mu_3^\ell=Y_3^\ell v_3/\sqrt2$,
   $\mu_4^\ell=Y_4^\ell v_2/2$, $\mu_5^\ell=Y_5^\ell v_2/2$. The printed values
   are those of the convention $\langle H_i^0\rangle=v_i$, which contradicts
   $v=246$ GeV in $m_W$. **Harmless downstream** — the $\mu$'s are eliminated
   for physical masses, so $M_\ell$, $G_k$ and $Q_i$ are unaffected; it matters
   only if one wants the Yukawa couplings themselves.
2. **Eq. (muil_equations) line 3 was circular.** The correct general relation is
   $(m_\tau-m_\mu)^2=(2\mu_3-m_\mu-m_\tau)^2+4(\mu_4+\mu_5)^2$, not
   $+16\mu_4\mu_5$; the printed form already presupposes the $\mu_4=\mu_5$ it is
   then used to derive. The conclusion survives by a non-circular route
   (patched, along with Eq. muil_sols line 3 and the $\mu_4\mu_5\ge0$ remark,
   which is now automatic).
3. **A coefficient slip:** $\mathrm{tr}(N)=4\mu_4^2+m_\mu^2+m_\tau^2$, not
   $2\mu_4^2+\ldots$. The conclusion $\mu_4=0$ is unaffected.

Everything else checks out: $R_S=R_AR_H$ identically, the $O_{12}$
block-diagonalization with $m_e=\mu_1-2\mu_2$, $\tan2\theta_\ell$ and the
$p_{1,2}$ closed forms, and $Q_1(A)=\mathrm{diag}(m)/v$ — the last an *exact
identity* (the first column of $R_A$ is the vacuum direction), not a fit.

### 9. Exact S₃ forces $V_{us}=0$; soft breaking is what turns it on

The quark sector (not treated in the draft) has 10 Yukawa couplings — 5 down via
$H$, 5 up via $\tilde H$, which is the same S₃ doublet because the S₃ matrices
are real. $M_u$ and $M_d$ therefore have **identical structure**.

The 1–2 block-diagonalizing rotation satisfies $\tan2\psi=-r$ with $r=v_1/v_2$:
it depends on the **vacuum alone**, never on the Yukawas, so the same $O_{12}$
acts in every sector. The first generation decouples **iff**
$r^2-\sqrt{r^2+1}-1=0$, whose only positive root is $r=\sqrt3$ — exactly the
vacuum that preserves the residual $\mathbb Z_2$. Hence with exact S₃

$$V_{\rm CKM}=O_{23}(\theta_u)^{\mathsf T}O_{23}(\theta_d)\ \Longrightarrow\
V_{us}=V_{ub}=V_{cd}=V_{td}=0,$$

against $|V_{us}|=0.2243$. This is [DasDeyPal16]'s "unbroken $\mathbb Z_2$ ⟹
approximate CKM block structure", derived here independently.

**Soft breaking fixes it, and costs nothing elsewhere.** Being dimension-2 by
definition it touches no quartic, so *every* boundedness/unitarity result of
finding 4 and notebook 01 carries over unchanged. What it does is release the
vacuum from $r=\sqrt3$: `build_model(soft=True)` adds the four CP-conserving
S₃-breaking quadratics (the $\mathbf2$ pair and the $H_S^\dagger H_{1,2}$ pair;
the hermitian quadratic space is 6-dimensional in the real symmetric case, of
which 2 are invariant), and the tadpole system stops being over-constrained.
A bounded fit then reproduces all six quark masses and the Cabibbo angle at a
misalignment of only $r-\sqrt3\approx-0.06$ (`results/quark_soft_fit.json`).

**Caveat, and it is a big one.** That fit targets six masses and $|V_{us}|$ only.
The untargeted elements come out badly wrong — $|V_{cb}|\approx0.98$ against a
measured $0.0408$, i.e. near-maximal 2–3 mixing. So finding 9 establishes the
*mechanism* by which soft breaking generates the Cabibbo angle, **not** that this
model reproduces the observed CKM matrix. A genuine global fit (all four CKM
parameters and six masses at once) may well fail, since the two-stage structure
ties the 2–3 rotation to the same few parameters that fix the masses. Open.

## Open questions / next

- **Get the [DasDey14] erratum.** Phys. Rev. D **91**, 039905 (2015), not posted
  to arXiv, APS paywalled. It may already fix the Eq. (4g) insufficiency found
  above, and it may also touch the unitarity eigenvalues Eq. (37) — which the
  notebook checks for internal consistency (typeset↔code identity, discriminants
  manifestly ≥ 0) but cannot check against a corrected source. **Cross-check
  target:** [BentoRomaoSilva22], which recomputes unitarity bounds for all
  symmetry-constrained 3HDMs and post-dates the erratum.
- **Complete the boundedness conditions.** The derived condition is exact only on
  the real neutral slice; complex-neutral and charged directions are covered
  numerically, not analytically. [BotoRomaoSilva22] does this properly for
  U(1)×U(1), U(1)×Z₂ and Z₂×Z₂ — not S₃ — and is the methodological model.
- **The 125 GeV cut is a mass condition only.** Testing [GomezBock21]'s
  alignment scenarios A/B (Eqs. 54–55; one CP-even state coupling maximally to
  $W/Z$, the other decoupled) needs the **gauge–scalar couplings** of their
  Eq. (49). `examples/thdm_s3.py` declares *only* the potential — no kinetic
  terms, no gauge bosons in the Lagrangian — so those couplings do not exist to
  be computed. Blocked on adding the electroweak kinetic sector.
- **A global quark fit** — see the caveat in finding 9. Whether the soft-broken
  model can reproduce all four CKM parameters *and* the six quark masses at once
  is open, and the $|V_{cb}|$ result is mild evidence against it without further
  structure. CP violation would additionally need complex Yukawas; everything in
  notebook 02 is real.
- **Soft breaking and the scalar spectrum.** The four soft quadratics shift the
  scalar mass matrices. Finding 4 / notebook 01 are untouched (quartics only),
  but the *spectrum* is not — re-running the parameter scan in the soft-broken
  vacuum is the natural follow-up, and would connect notebooks 01 and 02.
- **The neutrino sector** (finding 7) is enumerated but not built: Dirac Yukawas
  on $\tilde H$ plus S₃-allowed Majorana $\nu_R$ masses. feynlag has the pieces
  (`seesaw_mass_matrix`, `MajoranaRotation`, Takagi) — see `examples/sm_seesaw.py`.
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
  [arXiv:1404.2491](https://arxiv.org/abs/1404.2491) (v2, May 2014),
  [doi:10.1103/PhysRevD.89.095025](https://doi.org/10.1103/PhysRevD.89.095025).
  Eq. (3c) the λ₁…λ₈ potential; Eq. (4a)–(4g) boundedness-from-below;
  Eq. (36) + (37a)–(37l) the tree-unitarity eigenvalues. This is the actual
  source of the constraints — [GomezBock21] §2.2 delegates to it.
  ⚠️ **Erratum: Phys. Rev. D 91, 039905 (2015)** — *not* posted to arXiv (the
  arXiv record stops at v2) and paywalled at APS. Everything transcribed into
  `constraints.py` therefore comes from the **pre-erratum** arXiv v2; see
  finding 4 and the open questions.
- **[BentoRomaoSilva22]** M. P. Bento, J. C. Romão, J. P. Silva, *"Unitarity
  bounds for all symmetry-constrained 3HDMs"*, JHEP **08** (2022) 273,
  [arXiv:2204.13130](https://arxiv.org/abs/2204.13130),
  [doi:10.1007/JHEP08(2022)273](https://doi.org/10.1007/JHEP08(2022)273).
  Post-dates the erratum; the cross-check target for the unitarity eigenvalues.
- **[BotoRomaoSilva22]** R. Boto, J. C. Romão, J. P. Silva, *"Bounded from below
  conditions on a class of symmetry constrained 3HDM"*, Phys. Rev. D **106**,
  115010 (2022), [arXiv:2208.01068](https://arxiv.org/abs/2208.01068),
  [doi:10.1103/PhysRevD.106.115010](https://doi.org/10.1103/PhysRevD.106.115010).
  Covers U(1)×U(1), U(1)×Z₂, Z₂×Z₂ — not S₃ — but is the methodological model for
  doing the boundedness analysis properly (BFB-n and BFB-c directions separately).
- **[LFVHD]** M. Zeleny-Mora, M. Mondragón, T. A. Valencia-Pérez, *"Exploring LFV
  Higgs decays in the Three Higgs Doublet Model"*, draft — `LFVHD_3HDMS3.tex` in
  this directory. Supplies the S₃ lepton assignment, the Yukawa Lagrangian, the
  charged-lepton mass matrix and its diagonalization, and the $Q_i$ LFV
  couplings. Checked (and patched) in findings 6–8.
- **[DasDeyPal16]** D. Das, U. K. Dey, P. B. Pal, *"S₃ symmetry and the quark
  mixing matrix"*, Phys. Lett. B **753**, 315 (2016),
  [arXiv:1507.06509](https://arxiv.org/abs/1507.06509),
  [doi:10.1016/j.physletb.2015.12.038](https://doi.org/10.1016/j.physletb.2015.12.038).
  The quark-sector companion: an unbroken Z₂ leaves the CKM matrix approximately
  block diagonal, and soft S₃ breaking in the scalar sector generates the small
  elements. Independently re-derived in finding 9. (This is very likely the
  `Das:2015sca` key cited by the draft, though its `.bib` is not in the repo.)
- **[BabuWuXu23]** K. S. Babu, Y. Wu, S. Xu, *"Fermion Masses, Neutrino Mixing
  and Higgs-Mediated Flavor Violation in 3HDM with $S_3$ Permutation
  Symmetry"*, [arXiv:2312.15828](https://arxiv.org/abs/2312.15828) (2023).
  An alternative published S₃-3HDM Yukawa treatment; not used — the draft
  supplied the assignment instead.
- **[PDG]** S. Navas *et al.* (Particle Data Group), *Review of Particle
  Physics*, Phys. Rev. D **110**, 030001 (2024),
  [doi:10.1103/PhysRevD.110.030001](https://doi.org/10.1103/PhysRevD.110.030001).
  Quark masses and CKM elements used in the notebook-02 fit.
- **[Yildirim26]** E. Yildirim, *"Double SM-like Higgs Production at future
  $e^+e^-$ colliders in the 3-Higgs Doublet Model under the $S_3$ symmetry"*,
  [arXiv:2604.24421](https://arxiv.org/abs/2604.24421) (2026). Applies
  perturbative unitarity, vacuum stability and LHC/Tevatron data to this same
  model; a cross-check target for the constraint implementation.
