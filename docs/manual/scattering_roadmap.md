# 17. Extending to 2→2 Scattering

## The gap, stated honestly

Everything in `feynlag.pheno` before this chapter computes a **decay**: one
particle in, two (or three, off-shell) out. It cannot compute a **cross
section** at all — there is no notion of two incoming particles, no
Mandelstam invariants, no flux factor, and no way to combine two vertices
joined by a propagator into an amplitude rather than an already-squared
number. The covariant-trace design was chosen partly with this in mind (see
`CLAUDE.md`'s note that it gives "a cleaner path to a later 2→2"), and the
repository already carries two MadGraph-validated cross-section benchmarks —
$\sigma(e^+e^-\to\mu^+\mu^-) = 2.7878\pm0.0027$ pb and
$\sigma(e^+e^-\to W^+W^-) = 19.498\pm0.058$ pb at $\sqrt s = 200$ GeV
(`docs/benchmark.md`) — so the acceptance oracles for a native 2→2 engine
already exist and cost nothing new to obtain.

Two facts drive the whole design that follows.

**The ε (γ₅) term does not vanish for 2→2.** The 1→2 engine's
{func}`~feynlag.pheno.lorentz.reduce_projectors` proves the γ₅ trace term is
zero whenever a chiral fermion current meets at most 2 independent momenta
and 2 free Lorentz indices — true for every decay, because a two-body final
state supplies only $P=p_1+p_2$, one combination, and the polarization sums
it meets are symmetric. A 2→2 process has **3** independent momenta
($k_4=k_1+k_2-k_3$ is not independent), and with two chiral fermion currents
$|\mathcal M|^2$ genuinely contains an $\varepsilon\cdot\varepsilon$ term — the
forward–backward asymmetry, a real, physically necessary piece of the answer,
not a computational nuisance. The guard's threshold was `n_momenta > 3`,
which **passes** at exactly the 2→2 value and would have silently returned a
wrong $d\sigma/d\cos\theta$; Tier 1 tightens it to `n_momenta > 2` (§17.5).

**The 1→2 squared-amplitude functions are terminal and cannot be
extended.** {func}`~feynlag.pheno.amplitudes.ffs_squared`/`ffv_squared`
return an already-squared scalar, with the 1→2 mass-interference term and the
parent's spin average (`ffv_squared`'s hardcoded `/3`) baked in as literals
for that one topology. There is no partial expression to attach a propagator
or a second diagram to. Tier 1 therefore builds the amplitude explicitly —
open fermion chains, propagators, a genuine sum over diagrams — reusing the
covariant engine (`pheno/lorentz.py`) unchanged underneath.

## 17.1 Tier 1 — kinematics, one diagram, no interference ✅ (delivered)

**Channels**: $e^+e^-\to\mu^+\mu^-$ through a single photon (pure QED);
$f_1\bar f_1\to f_2\bar f_2$ through one s-channel scalar. **Effort:
small–medium.**

What Tier 1 deliberately does *not* attempt: more than one diagram (γ/Z
interference — Tier 3). This is a hard guard, not an omission: the engine
raises `NotImplementedError` rather than returning a number that would be
wrong by an $O(1)$ fraction. (Chiral couplings on both vertices of the same
diagram — the ε·ε term above — were Tier 1's other hard guard; §17.2 below
computes it instead.)

### What was built

- {class}`~feynlag.pheno.kinematics.TwoToTwoKinematics` — Mandelstam
  invariants `s` (primary, `positive=True`) and `t` (primary, `real=True`,
  negative throughout the physical region), with `u` a **derived** property
  so momentum conservation holds by construction rather than assertion; the
  ten-entry on-shell `dot` table (including a dedicated momentum head for the
  dependent $k_4$, so conservation is a testable invariant of the table, not
  expression-level algebra colliding with `TensAdd` expansion inside
  {func}`~feynlag.pheno.lorentz.dirac_trace`); the flux factor
  $1/(2\sqrt{\lambda(s,m_1^2,m_2^2)})$; `t_bounds`/`cos_theta`/`t_of_cos`; and
  the $d\sigma/dt$, $d\sigma/d\cos\theta$ conversion factors. `s` and `t` are
  kept as the primary symbols rather than `(s,\cos\theta)` because the `dot`
  table is then linear with no radicals — a $\cos\theta$ parametrization would
  drag $\sqrt{\lambda_i\lambda_f}$ into every table entry and hence into every
  Dirac trace, exactly the reason
  {class}`~feynlag.pheno.kinematics.ThreeBodyKinematics` confines its own
  radicals to `s12_bounds`.
- {mod}`~feynlag.pheno.diagrams` — the amplitude-level object layer:
  {class}`~feynlag.pheno.diagrams.Leg`, {class}`~feynlag.pheno.diagrams.ChainVertex`
  (with `.bar()`, matching {func}`feynlag.dirac.dirac_conjugate`, and
  `.epsilon_coefficient()`), {class}`~feynlag.pheno.diagrams.SpinorChain`
  (the open, γ₅-free trace of one fermion line — diagonal and
  chirality-mixing pieces derived the same way as
  `ffs_squared`/`ffv_squared`, but left open on external Lorentz indices
  instead of immediately reduced), {class}`~feynlag.pheno.diagrams.BosonPropagator`
  and {class}`~feynlag.pheno.diagrams.Diagram`/`Amplitude`. `Amplitude.squared`
  combines two chains through one propagator and reduces to on-shell dot
  products with a single call to {func}`~feynlag.pheno.lorentz.contract_to_dots`
  — never per-chain — matching the library's existing rule that that function
  runs once, on a fully-contracted expression.
- {mod}`~feynlag.pheno.propagator` gained amplitude-level (not
  squared-modulus) propagators — `propagator_denominator`,
  `scalar_propagator`, `vector_propagator` — alongside the existing
  {func}`~feynlag.pheno.propagator.breit_wigner`/`vector_propagator_numerator`,
  which Tier 1 reuses verbatim rather than duplicating.
- {class}`~feynlag.pheno.particles.ExternalState` — spin/colour degrees of
  freedom for an external scattering state, and the home of the averaging
  factor. This is the direct fix for the anti-pattern in
  `ffv_squared`'s hardcoded `/3`: an initial-state spin average masquerading
  as part of the Lorentz algebra, which made that function unusable the
  moment the vector is *internal* rather than the parent.
  {func}`~feynlag.pheno.scattering.average_factor` computes
  $1/\prod_i \mathrm{dof}_i$ from declared data, and
  {func}`~feynlag.pheno.scattering.cross_section`/`differential_cross_section`
  apply it exactly once, never inside a squared amplitude.
- {mod}`~feynlag.pheno.scattering` — `average_factor`,
  `differential_cross_section`, `cross_section` (symbolic `t`-integral), and
  the two worked Tier-1 assemblers, `ffv_s_channel_squared`/
  `ffs_s_channel_squared`, in the same coupling-in/number-out style as
  {func}`~feynlag.pheno.offshell.scalar_offshell_vv_width`.

### `reduce_projectors`: tightened now, retired for chains in Tier 2

`n_momenta > 3` became `n_momenta > 2` in
{func}`~feynlag.pheno.lorentz.reduce_projectors` — the only two callers,
`ffs_squared`/`ffv_squared`, always pass `n_momenta=2`
(a chain's own two legs), so the change is invisible to the 1→2 suite; a
regression test now pins that `n_momenta=3` raises
(`tests/test_pheno.py::test_gamma5_guard_raises_outside_two_body`). This
function still correctly proves each *individual* chain's own ε term is zero
— that reasoning is unaffected by the overall process being 2→2, since a
chain's trace only ever involves its own two legs. What it cannot see is the
**cross-chain** ε·ε term that appears when *two* chiral chains meet through a
propagator; {class}`~feynlag.pheno.diagrams.Amplitude` computes that term
directly now (§17.2, delivered) instead of refusing it, and
`SpinorChain.trace` no longer calls `reduce_projectors` at all — it stays
frozen as the 1→2-only helper `ffs_squared`/`ffv_squared` use
(`tests/test_scattering.py::test_reduce_projectors_is_no_longer_on_the_chain_path`).

### Verified

Against the textbook QED closed form (Peskin & Schroeder §5.1 [PS95])
$\sigma(e^+e^-\to\mu^+\mu^-) = \frac{4\pi\alpha^2}{3s}\cdot\frac{\beta(3-\beta^2)}{2}$,
$\beta=\sqrt{1-4m_\mu^2/s}$ — symbolically exact, plus the massless limit
$4\pi\alpha^2/3s$ — and against an independent explicit-4×4-Dirac-matrix
oracle built in the CM frame (sharing no code with the covariant engine, in
the same spirit as `tests/test_pheno.py`'s `_oracle_*`). At the exact
`docs/benchmark.md` parameter point ($\alpha^{-1}=132.50698$, $\sqrt s=200$
GeV), the QED-only cross section comes out **2.322 pb** — the photon-only
fraction of MadGraph's full 2.7878 pb, with the $\approx20\%$ gap being
exactly the $\gamma$/$Z$ interference Tier 3 supplies, so this number cannot
be accidentally "passed" here. `tests/test_scattering.py` also pins that a
chiral coupling on both vertices genuinely differs from the naive "half the
vector result" guess that *does* hold for a 1→2 decay
(`test_ffv_chiral_is_half_the_vector_result` in `test_pheno.py`) — by up to
$O(1)$, at several angles — which is what justified Tier 1's refusal to
compute that case rather than silently guessing, before §17.2 computed it
for real.

## 17.2 Tier 2 — the ε (γ₅) algebra ✅ (delivered)

**Channels**: $e^+e^-\to\mu^+\mu^-$ through the $Z$ alone; any process with a
single diagram carrying two chiral fermion currents. **Effort: large.**

This landed before Tier 3, not folded into it. It was needed the moment one
diagram has two chiral currents — no interference required — so it could be
validated in isolation, against one closed form
($d\sigma/d\cos\theta \propto (1+\cos^2\theta) + A_{FB}\cos\theta$, with the
sign and magnitude of the forward–backward asymmetry pinned against
[LEPEWWG06]), with exactly one new failure mode. Folded into Tier 3 alongside
multi-diagram interference, a wrong $A_{FB}$ could equally have been the ε
reduction, a relative diagram sign, or fermion-flow bookkeeping — three
candidate bugs instead of one.

### What was built

- {mod}`~feynlag.pheno.epsilon` (new) — the only module with ε knowledge.
  SymPy's own `sympy.physics.hep.gamma_matrices` ships neither a γ₅ object
  nor an ε-aware `gamma_trace` (checked directly against the installed
  version, not assumed), so both identities below are hand-rolled rather
  than delegated to a SymPy primitive:
  - $\mathrm{Tr}[\gamma^a\gamma^b\gamma^c\gamma^d\gamma_5] =
    \kappa\,\varepsilon^{abcd}$ — {func}`~feynlag.pheno.epsilon.gamma5_trace_coefficient`
    reads $\kappa$ off the literal $\gamma^0\gamma^1\gamma^2\gamma^3\gamma_5$
    matrix from `feynlag.dirac._dirac_rep`, against a stated
    $\varepsilon^{0123}=+1$ normalization
    ({func}`~feynlag.pheno.epsilon.levi_civita_array`) — $\kappa=-4i$,
    **derived, not quoted**, the same discipline
    `dirac.majorana_symmetry_sign` uses for its own sign.
  - $\varepsilon^{a\ldots}\varepsilon^{b\ldots} = s_{\det}\!\cdot\!\det[g^{a_ib_j}]$
    — {func}`~feynlag.pheno.epsilon.epsilon_product_sign` derives
    $s_{\det}=-1$ the same way, against the $(+,-,-,-)$ metric.
    {func}`~feynlag.pheno.epsilon.epsilon_pair_tensor` expands this as 24
    signed products of four metrics with all free Lorentz indices left
    **open**, each slot resolved by content (two momenta contract through a
    fresh dummy per pairing; a momentum meeting a free index just evaluates
    the head there; two free indices meet the ordinary `LorentzIndex.metric`)
    — no `Eps` tensor object needed, and the all-momentum case reduces on
    its own to the roadmap's $-\det[p_i\cdot p'_j]$ Gram determinant.
  - {func}`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes` — the
    single-ε cross terms (one chain's ε piece times the other chain's
    ordinary trace) reduce to a scalar times $\varepsilon$(four momenta drawn
    from the diagram's own external legs); since
    $\varepsilon(a,b,c,d)^2=-\det[\cdot]$ (identity 2 again), a vanishing
    Gram determinant over every 4-subset of the diagram's momenta **proves**
    every such term is zero — a computed prove-or-refuse guard reading the
    diagram's actual `kin.dot` table, not a momentum-count proxy that would
    need re-tuning for a future topology.
- {meth}`~feynlag.pheno.diagrams.SpinorChain.epsilon_structure` — a chain's
  ε piece as `(prefactor, slots)` or `None` (a scalar chain, or a
  pure-vector `g_L=g_R` chain, has none — exactly the cases the pre-Tier-2
  engine already handled correctly, so those results are unchanged
  bit-for-bit). `SpinorChain.trace` no longer calls `reduce_projectors` —
  see the retirement note above.
- {meth}`~feynlag.pheno.diagrams.Amplitude.squared` assembles the ε
  contribution alongside the existing non-ε piece — both are separately
  fully-contracted scalars before being summed, so the house rule that
  {func}`~feynlag.pheno.lorentz.contract_to_dots` runs **exactly once**, on a
  fully-contracted expression, survives literally. Contracting the two
  propagator numerators against the ε tensor stepwise (one numerator at a
  time, with `.expand()` between them) is ~3.5× faster than contracting both
  at once for a massive mediator.
- {func}`~feynlag.pheno.scattering.forward_backward_asymmetry` — the Tier-2
  acceptance observable, $A_{FB}=(\sigma_F-\sigma_B)/(\sigma_F+\sigma_B)$
  over the $\cos\theta\gtrless0$ hemispheres. Needs no
  {class}`~feynlag.pheno.particles.ExternalState`\ s, unlike
  `differential_cross_section`/`cross_section`: the spin average, any
  identical-particle symmetry factor, and the $d\sigma/d\cos\theta$
  conversion factor are all $\cos\theta$-independent and cancel exactly in
  the ratio.

### Verified

The headline test, `test_chiral_squared_amplitude_matches_explicit_matrix_oracle`:
with **fully symbolic**, independent chiral couplings $g_L,g_R,h_L,h_R$ on
both vertices (not the pure-vector case, where the ε term is trivially zero)
through a massless mediator, the covariant engine matches the independent
explicit-4×4-matrix oracle exactly (`tests/test_scattering.py`'s
`_oracle_qed_general`) — this one identity simultaneously fixes $\kappa$, the
$\mp$ sign in the $P_{L,R}$ split, and the $\varepsilon_{0123}$ convention,
and its full generality is what shows the single-ε cross terms genuinely
cancel rather than happening to vanish at one lucky point.
`test_z_massive_mediator_matches_explicit_matrix_oracle` extends the oracle
to a massive mediator (the only test needing that extension) and checks
`vector_propagator_numerator`'s sign convention composes correctly with the ε
piece; `test_massive_mediator_epsilon_piece_drops_the_qq_term` separately
confirms the propagator's $q_\mu q_\nu/M^2$ term drops out of the ε
contribution automatically (a repeated momentum in the same ε ⟹ two equal
Gram-determinant rows ⟹ 0), for four symbolic external masses.

Against the literature: the massless-final-state angular distribution
matches the LEP Born-level form,
$\Sigma|\mathcal M|^2=(g_L^2+g_R^2)(h_L^2+h_R^2)(1+\cos^2\theta)+2(g_L^2-g_R^2)(h_L^2-h_R^2)\cos\theta$
(Eq. (1.55) of [LEPEWWG06] at zero beam polarization), and
`forward_backward_asymmetry` reproduces
$A_{FB}^{0,f}=\tfrac34A_eA_f$ (Eq. (1.66)) symbolically, plus a numeric pin
at the LEP effective weak mixing angle
$\sin^2\theta_{\mathrm{eff}}^{\mathrm{lept}}=0.23153$ giving the standard
leptonic $A_\ell\approx0.147$, $A_{FB}^{0,\ell}\approx0.016$.
`test_z_only_total_cross_section_is_epsilon_independent` confirms the ε term
is odd in $\cos\theta$ and integrates away over the full range — a chiral
mediator's *total* cross section equals a pure-vector coupling's with the
same $g_L^2+g_R^2$, so Tier 1's 2.322 pb QED benchmark cannot shift from
Tier 2 landing; $A_{FB}$ is an angular observable only.

## 17.3 Tier 3 — interference and the first cross-section benchmark

**Channels**: the full $e^+e^-\to f\bar f$ ($\gamma$/$Z$ interference); s-,
t-, u-channel topology enumeration; a `ScatteringCalculator` mirroring
{class}`~feynlag.pheno.calculator.DecayCalculator`. **Effort: medium–large.**

{class}`~feynlag.pheno.diagrams.Amplitude` drops its `len(diagrams) == 1`
guard and performs the full $\sum_{d,d'} c_d\bar c_{d'}$ double sum, which
needs relative fermion-flow signs and identical-particle exchange signs — the
2→2 analogue of `DecayCalculator.channels`, currently a one-line "vertex
contains parent" search, becomes a genuine s/t/u topology enumeration over
pairs of {class}`~feynlag.pheno.vertices.DecayVertex`.

**Oracle**: $\sigma(e^+e^-\to\mu^+\mu^-) = 2.7878\pm0.0027$ pb at the exact
`docs/benchmark.md` parameter point — 20% above Tier 1's QED-only 2.322 pb,
so this genuinely tests the interference term and not just a repeat of the
QED piece.

## 17.4 Tier 4 — derivative couplings and the gauge-cancellation acid test

**Channels**: $e^+e^-\to W^+W^-$; any process needing a `VVV`/`VSS` vertex.
**Effort: large.**

The blocker: `VSS`/`VVV` couplings carry momentum tags `p(φ)` from
{func}`~feynlag.operators.to_momentum_space`, which
{func}`~feynlag.pheno.amplitudes.amplitude_squared` explicitly refuses to
square (`amplitudes.py`'s `SUPPORTED_VERTEX_TYPES` excludes both). Tier 4
must resolve each tag to the kinematic `TensorHead` of the diagram leg `φ` is
assigned to, incoming/outgoing sign included. The relative sign between the
$W\bar\nu e$ current and the triple-gauge coupling is exactly what separates
a correct $\approx19.5$ pb from the $\approx98$ pb a flipped sign gives
(`docs/benchmark.md`'s account of the MadGraph round-trip) — Tier 4 must
*derive* that sign from the diagram construction, not import the export
script's empirical fix.

**Oracle**: $\sigma(e^+e^-\to W^+W^-) = 19.498\pm0.058$ pb at $\sqrt s=200$
GeV, plus a direct pin that $|\mathcal M|^2$ does not grow like $s^2$ at high
energy — the gauge-cancellation signature itself, checked directly rather
than only through one integrated number.

## 17.5 Tier 5 — coloured / hadronic 2→2 (parton level)

**Channels**: $q\bar q\to q'\bar q'$, $qg\to qg$, $gg\to gg$. **Effort:
medium–large.**

Needs a colour-flow layer producing the colour factor
$C_{dd'}=\sum_{\text{colour}} T_d\bar T_{d'}$ separated from the Lorentz part,
colour averaging over the initial state (already slotted via
{class}`~feynlag.pheno.particles.ExternalState.color`, unused until now), and
lifting `collect_decay_vertices`/`fermion_decay_vertices`'s twin 3-leg hard
filters so the `VVVV` ($gggg$) contact vertex — already assembled by
`export/ufo/vvvv.py`'s `assemble_vvvv` — becomes reachable.

**Oracle**: the standard parton-level $|\mathcal M|^2$ table (Ellis, Stirling
& Webber [ESW96], e.g. $q\bar q\to q'\bar q' = \tfrac49\cdot\tfrac{t^2+u^2}{s^2}$,
$gg\to gg = \tfrac92(3-\tfrac{tu}{s^2}-\tfrac{su}{t^2}-\tfrac{st}{u^2})$),
with colour factors $C_F=4/3$, $C_A=3$, $T_R=1/2$. **Explicitly excludes
PDFs and hadronic luminosity** — this tier stops at the parton-level $2\to2$
cross section, matching the "no PDFs" scope already implicit in the rest of
the library (no proton structure anywhere).

## 17.6 Summary

| tier | channels | new machinery | effort | oracle |
|---|---|---|---|---|
| 1 ✅ | $e^+e^-\to\mu^+\mu^-$ (γ only), $f\bar f\to f\bar f$ via a scalar | `TwoToTwoKinematics`; `SpinorChain`/`Diagram`/`Amplitude` (**done**); amplitude-level propagators; `ExternalState` averaging | small–medium | Peskin QED closed form + explicit-matrix oracle |
| 2 ✅ | $e^+e^-\to\mu^+\mu^-$ (Z only); any single chiral-current diagram | ε (γ₅) algebra (`pheno.epsilon`): $\mathrm{Tr}[\gamma\gamma\gamma\gamma\gamma_5]$, ε·ε → Gram determinant, `forward_backward_asymmetry` | large | $d\sigma/d\cos\theta$ shape + $A_{FB}=\tfrac34A_eA_f$ vs [LEPEWWG06]; γ₅-carrying matrix oracle (massless + massive mediator) |
| 3 | full $e^+e^-\to f\bar f$ (γ/Z interference) | multi-diagram double sum; s/t/u topology search; relative signs | medium–large | **MadGraph 2.7878 pb** |
| 4 | $e^+e^-\to W^+W^-$; derivative couplings | `VVV`/`VSS` momentum-tag resolution; external vector polarization sums | large | **MadGraph 19.498 pb**; no $s^2$ growth |
| 5 | $q\bar q\to q\bar q$, $qg\to qg$, $gg\to gg$ | colour-flow algebra; colour averaging; 4-point contact vertices | medium–large | ESW parton-level table (no PDFs) |

The recommended order is the table order: Tier 1 gave the first native cross
section for the cost of an object layer that reuses the existing covariant
engine unchanged; Tier 2 was the genuinely new algebra 2→2 requires that 1→2
never did; Tier 3 is where the two MadGraph benchmarks stop being aspirational
and start being reproduced; Tier 4 is the hardest single piece (derivative
couplings); Tier 5 generalizes to QCD but adds no new *kind* of physics beyond
colour bookkeeping.

See {doc}`decays_roadmap` for the parallel decay-side roadmap and its
`n_momenta`/ε-drop precedent, and `tests/test_scattering.py` for what Tier 1
pins.

## 17.7 References

- **[PS95]** M. E. Peskin and D. V. Schroeder, *An Introduction to Quantum
  Field Theory*, Addison-Wesley (1995), ISBN 0-201-50397-2 — Chapter 5,
  "Elementary Processes of Quantum Electrodynamics," §5.1, the
  $e^+e^-\to\mu^+\mu^-$ closed form Tier 1's cross section reproduces.
- **[ESW96]** R. K. Ellis, W. J. Stirling and B. R. Webber, *QCD and Collider
  Physics*, Cambridge Monographs on Particle Physics, Nuclear Physics and
  Cosmology 8, Cambridge University Press (1996), ISBN 978-0-521-54589-1 —
  the parton-level $2\to2$ $|\mathcal M|^2$ table Tier 5 targets.
- **[LEPEWWG06]** ALEPH, DELPHI, L3, OPAL, SLD Collaborations, LEP
  Electroweak Working Group, SLD Electroweak and Heavy Flavour Groups,
  "Precision Electroweak Measurements on the Z Resonance," Phys. Rept. 427
  (2006) 257–454, [arXiv:hep-ex/0509008](https://arxiv.org/abs/hep-ex/0509008),
  [doi:10.1016/j.physrep.2005.12.006](https://doi.org/10.1016/j.physrep.2005.12.006)
  — Eq. (1.55) the Born-level $e^+e^-\to f\bar f$ angular distribution,
  Eq. (1.56) the coupling asymmetry $A_f$, Eq. (1.66) the forward–backward
  asymmetry $A_{FB}^{0,f}=\tfrac34A_eA_f$ Tier 2's `forward_backward_asymmetry`
  reproduces.
- **[PDG]** Particle Data Group, *Review of Particle Physics* — "Kinematics"
  review (Mandelstam invariants, the two-body $t$-range, $d\sigma/dt$); see
  the current edition at [pdg.lbl.gov](https://pdg.lbl.gov/), cited without a
  year pin since PDG republishes annually and section numbers shift (same
  convention as {doc}`decays_roadmap` §16.2's use of the same review).
