# Tutorials

Eight fully executed Jupyter notebooks, walking a worked model stage by
stage with real (stored) output — plots, mass matrices, Feynman rules. They
are tracked through the `nbstripout --keep-output` git filter (see the
repo's `CLAUDE.md`), so what you see below is exactly what re-running the
notebook produces.

```{toctree}
:maxdepth: 1

Particle_Decays_Tutorial
Scattering_Tutorial
SUN_Groups_Tutorial
SM_Feynman_Rules_Tutorial
SM_VLL_Tutorial
SM_U1X_Tutorial
ModelBuilding_Tutorial
SM_Seesaw_Tutorial
THDM_S3_Tutorial
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
branching-ratio-versus-mass plot. Then completes the picture: the
`DiracParticle` fermion sector, and the **off-shell $1\to3$** $h\to WW^*/ZZ^*$
(a $W^*$ line-shape figure showing the virtual $W$ never reaches its mass shell,
and the canonical Higgs BR chart — $b\bar b$ 61%, $WW^*$ 25%). Closes with the
two mistakes that fail *silently* — forgetting that a Dirac fermion is two Weyl
fields (an error that vanishes in the massless limit), and closed channels
turning $\sqrt\lambda$ imaginary — and then builds a $Z'$ from scratch to show
the same machinery on a new model.

## Scattering Tutorial

The first native **cross section**, not a decay width: $e^+e^-\to\mu^+\mu^-$
through a photon (Tier 1), matched against the Peskin & Schroeder closed form,
then $e^+e^-\to\mu^+\mu^-$ through the Z alone (Tier 2), where a real electron
Z coupling is pulled straight out of a built SM Lagrangian rather than typed
in as a textbook formula. Derives by hand why the $\gamma_5$ (ε-tensor) term
that vanishes in *every* decay this library computes — two independent
momenta is never enough — survives for $2\to2$, where two chiral currents
meeting at one propagator give three, and walks the two derived (not quoted)
identities in `feynlag.pheno.epsilon` that compute it: the trace coefficient
$\kappa=-4i$ and the Gram-determinant sign $s_{\det}=-1$. Reproduces the LEP
forward–backward asymmetry $A_{FB}=\tfrac34A_eA_f$ from first principles, with
a $d\sigma/d\cos\theta$ figure showing the chiral tilt against a symmetric
vector-coupling baseline, and closes with a total-cross-section sanity check
showing the ε term is an angular effect only — it integrates away, so Tier 1's
QED benchmark is untouched.

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

## 3HDM with S₃ Tutorial

The library's group-theory stress test: three Higgs doublets, with
`(H1, H2)` forming an `S3` doublet and `HS` an `S3` singlet, and the
potential built entirely from `S3.doublet_product`'s own
$2\otimes2=1\oplus1'\oplus2$ Clebsch–Gordan decomposition. Introduces the
one genuinely new invariance concept in the whole tutorial set — a
**finite** discrete-symmetry check (`check_discrete_invariance`, the exact
group substitution) rather than the infinitesimal linearization every
gauge check elsewhere relies on — and shows a model where the vacuum
isn't free to tune: with three VEVs but only two independent mass
parameters, the third tadpole condition **forces** the alignment
$v_1^2=v_2^2/3$, derived directly from the symbolic tadpole system rather
than assumed — matching, up to the basis swap, the tadpole solution of the
literature S₃-3HDM model (Gómez-Bock, Mondragón & Pérez-Martínez, EPJC 81,
942 (2021)) this example follows. Builds the pseudoscalar and charged mass
matrices alongside the CP-even one, then implements and verifies that
paper's geometric rotation ansatz: it exactly, symbolically diagonalizes
the Goldstone-protected pseudoscalar/charged sectors for any couplings,
while the CP-even sector needs one further dynamical mixing angle (reusing
the same 2×2 tool `thdm.py` uses for the plain 2HDM) — closing with a
numerical stability scan (mirroring the paper's own) that turns a mostly
tachyonic benchmark point into seven genuine physical scalar masses.
