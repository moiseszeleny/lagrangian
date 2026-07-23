# Tutorials

Seven fully executed Jupyter notebooks, walking a worked model stage by
stage with real (stored) output — plots, mass matrices, Feynman rules. They
are tracked through the `nbstripout --keep-output` git filter (see the
repo's `CLAUDE.md`), so what you see below is exactly what re-running the
notebook produces.

```{toctree}
:maxdepth: 1

Particle_Decays_Tutorial
SUN_Groups_Tutorial
SM_Feynman_Rules_Tutorial
SM_VLL_Tutorial
SM_U1X_Tutorial
ModelBuilding_Tutorial
SM_Seesaw_Tutorial
```

## Particle Decays Tutorial

How to get from a Lagrangian to a measured lifetime, for a student who has
met Feynman rules and Dirac spinors but never carried a decay calculation
through to a number. Derives everything by hand first — two-body phase space
and the Källén function, the spin sums that turn $|\mathcal{M}|^2$ into a
Dirac trace, the trace theorems, polarisation sums — and only then reveals
`DecayCalculator` as the automation of exactly those steps. Lands on
$\Gamma(h\to f\bar f) = N_c m_h m_f^2\beta^3/8\pi v^2$ (with the $\beta^3$
explained as the P-wave signature of a CP-even scalar), $\Gamma(Z\to\nu\bar\nu)$
and $\Gamma(W\to\ell\nu)$ within ~1% of the PDG, and the familiar Higgs
branching-ratio-versus-mass plot. Closes with the two mistakes that fail
*silently* — forgetting that a Dirac fermion is two Weyl fields (an error that
vanishes in the massless limit), and closed channels turning $\sqrt\lambda$
imaginary — and then builds a $Z'$ from scratch to show the same machinery on
a new model.

## SU(N) Groups Tutorial

A gentle, self-contained introduction to gauge representations for a reader
new to particle physics (linear algebra + basic QM only). Builds up from
"what is a generator?" through the Lie algebra and representations to
`feynlag`'s dynamic any-SU(N)/any-irrep machinery: Dynkin labels and the Weyl
dimension formula, a peek at the highest-weight/ladder (Gelfand–Tsetlin)
construction (with the tell-tale `√2` ladder entries of the **6** of SU(3)),
conjugate representations `T̄ = −T*`, and anomaly coefficients — culminating in
the SU(5) `5̄ + 10` anomaly cancellation and a gauge-invariance check of a
scalar in the fundamental of SU(4).

## SM Feynman Rules Tutorial

Builds the full Standard Model (Higgs, electroweak gauge, leptons, QCD)
from scratch and extracts its Feynman rules, mirroring
`examples/sm_scalar_gauge.py` one pipeline stage at a time.

## SM VLL Tutorial

Adds a vector-like lepton doublet to the SM and walks the biunitary
diagonalization of the resulting mass matrix, including a standalone demo
of why `expand_bilinear` is required for fermion mass-basis rotations to
extract correctly.

## SM U(1)_X Tutorial

Extends the SM by a second, symbolically-charged abelian gauge factor and
walks the chained Weinberg → Z–Z′ rotation that results from tree-level
kinetic/mass mixing.

## Model Building Tutorial

Goes one step earlier than the others: instead of analyzing a
hand-written Lagrangian, it shows the *model-building* tools. For a dark
`U(1)_D` sector with symbolic charges it uses `feynlag.anomalies` to derive
the anomaly-free charge assignment (forcing the dark fermion to be
vector-like), `feynlag.suggest` to enumerate the invariant operator basis
(and catch a mistuned charge that admits no mass term), and
`build_lagrangian` to assemble a validated model before running the full
pipeline to the dark-photon mass and `Z_D χχ` coupling.

## SM Seesaw Tutorial

The Standard Model extended by right-handed neutrinos with a large Majorana
mass — the **type-I seesaw**. Uses the Majorana machinery (`diracC`,
`MajoranaBilinear`, `majorana_mass_matrix`) to build the `[[0, m_D], [m_Dᵀ,
M_R]]` mass matrix, `diagonalize_takagi` for the light (sub-eV) + heavy (~M_R)
spectrum, and the charge-conjugation-aware `MajoranaRotation` to extract the
physical heavy-neutrino couplings — showing `W ℓ̄ N = (g/√2)·V` with the
light–heavy mixing `V ≈ m_D/M_R`, and its decoupling as `M_R → ∞`.
