# 16. Extending the Decay Calculator

## The gap, stated honestly

The decays tutorial closes with the Higgs branching-ratio plot — and at the
physical $m_h = 125$ GeV that plot is *qualitatively* wrong: with one lepton
generation and on-shell $1\to2$ kinematics only, it shows
$\mathrm{BR}(h\to\tau\tau) = 100\%$, because the on-shell $WW/ZZ$ channels are
closed ($2m_W = 160.8 > 125.25$). Nature disagrees:

| channel | BR at $m_h=125$ GeV | tier |
|---|---|---|
| $b\bar b$ | 58.2% | **1** — tree-level $f\bar f$ |
| $WW^*$ | 21.4% | **2** — off-shell $VV^*$ |
| $gg$ | 8.2% | **3** — loop-induced |
| $\tau^+\tau^-$ | 6.3% | **1** |
| $c\bar c$ | 2.9% | **1** |
| $ZZ^*$ | 2.6% | **2** |
| $\gamma\gamma$ | 0.23% | **3** |
| $Z\gamma$ | 0.15% | **3** |
| $\mu^+\mu^-$ | 0.022% | **1** |

This chapter is the roadmap for closing that gap. The three tiers differ by
orders of magnitude in effort, and they differ in *kind*: Tier 1 is model
content plus one API improvement (the $|\mathcal M|^2$ engine already
computes it); Tier 2 is a genuine architectural extension (from "square one
vertex" to "assemble a tree diagram"); Tier 3 is the only one that steps
outside the tree-level, derive-from-the-Lagrangian ethos — deliberately, and
with precedent.

Coverage arithmetic: Tier 1 alone reaches $\approx 67\%$ of the width and —
more importantly — replaces the wrong plot with a right one whose missing
pieces are quantified. Adding Tier 2 reaches $\approx 91\%$; adding Tier 3
completes the canonical picture.

## 16.1 Tier 1 — all tree-level $f\bar f$ channels

**Channels**: $b\bar b$, $c\bar c$, $s\bar s$, $\tau^+\tau^-$, $\mu^+\mu^-$.
**Effort: small.** The physics engine needs *nothing new*:
{func}`~feynlag.pheno.amplitudes.ffs_squared` already produces
$\Gamma(h\to f\bar f) = N_c\,m_h m_f^2\beta^3/8\pi v^2$, and the colour
factor $N_c$ is a solved problem — a colour-triplet channel with a single
declared colour component and `color_factors={q: 3}` on one leg yields
exactly $3\,y_q^2\,m_h\beta^3/16\pi$, verified against the leptonic case.

What is actually missing:

### Model content

Declare all three generations of leptons and quarks with realistic Yukawas.
`examples/sm_scalar_gauge.py` already contains the full pattern — `SU3c`
triplets `QL`/`uR`/`dR`, flavour-indexed `Yu`/`Yd`, the explicit Python
colour sum — so this is a modelling exercise, not new machinery.

### The colour over-count trap (verified, and the motivation for the API fix)

There is exactly one wrong way to combine the existing pieces, and it is the
*natural* way. Write the Yukawa as an explicit 3-colour sum (as
`sm_scalar_gauge.py` does — correctly, for *vertex extraction*), then collapse
the three colour components into one particle with `particle_map`, then apply
`color_factors`. Each step looks right; together they give

$$\Gamma_{\text{wrong}} = 81\,y_b^2\,(\dots) = 27\times\Gamma_{\text{right}}.$$

The mechanism: `particle_map` merges the three colour couplings
**coherently** — the vertex coupling becomes $3y_b$, contributing $9y_b^2$ to
$|\mathcal M|^2$ — and `color_factors` on both legs multiplies a further
$3\times3$. But colour is an **incoherent** sum: one final state per colour,
$\Gamma = N_c\,\Gamma_{\text{one colour}}$. The correct recipe today is *one*
colour component in the decay-facing Lagrangian plus `color_factor=3` on
*one* leg.

### The recommended addition: a post-EWSB `DiracParticle`

A natural question first — *why not declare a `DiracFermion` instead of two
`WeylFermion`s and avoid all pairing bookkeeping?* Because in the **gauge
basis it is impossible**: the SM is chiral, $b_L$ sits in an SU(2) doublet
with $Y=+1/6$ while $b_R$ is a singlet with $Y=-1/3$, and a 4-component field
carries a single rep assignment — no choice makes the unbroken Lagrangian
invariant. This is why `DiracFermion` **raises on construction**
(`fields.py`): for input fields it is not a missing feature but a physics
impossibility. (Only *vector-like* fermions, with equal L/R reps as in
`examples/sm_vll.py`, admit one; enabling `DiracFermion` as sugar that
expands to two Weyls internally for that case would be harmless, but does
not help the chiral SM fermions this tier needs.)

**After EWSB, however, the electron *is* one Dirac particle** — and the pheno
API currently encodes that fact as three loose dicts: `particle_map` (which
Weyl legs pair up), `masses` (keyed separately), and `color_factors` (keyed
separately again). Every silent failure mode the tutorial warns about traces
to this split:

- omit or misspell a `particle_map` entry → the L and R currents stay as two
  half-channels, each missing the $g_Lg_R$ mass interference (invisible in
  the massless limit — the worst kind of bug);
- a mass-key mismatch → `channels()` on
  {class}`~feynlag.pheno.calculator.DecayCalculator` *silently skips* the leg
  and the channel vanishes with no error;
- colour multiplicity applied in more than one place → the $81\times$
  over-count above.

The fix is one small dataclass — **now delivered**
({class}`~feynlag.pheno.particles.DiracParticle`):

```python
DiracParticle(
    "b",
    left=QL.components[1], right=bR.components[0],   # the two Weyl legs
    mass=mb, color=3,                                # bar legs auto-derived
)
```

with `DecayCalculator(model, particles=[...], ...)` consuming a list of these
instead of the dict triple. The bar legs come from the existing
`bar_partner` registry (`fields.py`), the mass travels with the particle, and
$N_c$ is applied once, per channel, incoherently — all three traps are now
*unrepresentable*, and any fermion channel claimed by no declared particle is
surfaced in `calc.unmatched_channels` (with a warning) rather than silently
dropped. The physics engine was untouched; this was calculator plumbing. See
{doc}`decays` §15.4 for the API and `examples/sm_higgs_decays.py` for the full
$b\bar b$-dominated fermionic branching-ratio table.

### Running masses

A physical $\Gamma(h\to b\bar b)$ uses the running mass
$\overline{m}_b(m_h)\approx 2.8$ GeV in the Yukawa, not the pole mass
$\approx 4.8$ GeV — a factor $\sim 3$ in the rate. This is a *numeric input*
choice (feed the running value), not new machinery; the roadmap notes it so
the Tier-1 numbers land near the canonical table rather than mysteriously
high.

## 16.2 Tier 2 — off-shell $WW^*$ and $ZZ^*$  ✅ (V–V\* delivered)

**Channels**: $h\to WW^*\to W f\bar f'$ (21.4%), $h\to ZZ^*\to Z f\bar f$
(2.6%). **Effort: large** — this is the architectural jump. **Delivered** in
`feynlag.pheno.offshell` (see the end of this section); the FFFF and
$t\to bW^*$ topologies remain.

The on-shell `VVS` width is exactly zero at $m_h=125$ GeV because the
two-body channel is closed; the physical decay proceeds through *one*
on-shell $V$ and one **off-shell** $V^*$ that materialises as a fermion
pair. That is a $1\to3$ process with an internal propagator — three things
the current design deliberately does not have:

1. **Propagators.** An internal line
   $\frac{-i\left(g_{\mu\nu} - q_\mu q_\nu/m_V^2\right)}{q^2 - m_V^2 + i m_V\Gamma_V}$
   with a Breit–Wigner width (which is *itself* a decay-calculator output —
   pleasingly self-referential: $\Gamma_V$ feeds back from Tier 1).
2. **$1\to3$ phase space.** The two-body factor
   $\sqrt\lambda/16\pi M^3$ is a closed form; three-body phase space is a
   2-dimensional (Dalitz) integral over $(q^2, \cos\theta)$ or
   $(m_{12}^2, m_{23}^2)$ that has no closed form once the propagator sits
   inside it — a numerical integration layer (the first in the library).
3. **Diagram assembly.** {func}`~feynlag.pheno.amplitudes.amplitude_squared`
   squares a *single* {class}`~feynlag.pheno.vertices.DecayVertex`. An
   off-shell amplitude is vertex × propagator × vertex, and squaring it puts
   the propagator momentum $q$ *inside* the trace algebra — the covariant
   engine in `pheno/lorentz.py` handles the traces (that part generalises
   cleanly), but a new assembly layer must build the chain and route the
   dot products $p_i \cdot q$.

The natural staging inside Tier 2: `hVV*` first with the narrow-$V$
approximation as a cross-check
($\Gamma_{h\to V f\bar f} \to \Gamma_{h\to VV}\,\mathrm{BR}$ above threshold),
then the full $q^2$ integration. The same machinery immediately gives
three-body decays generally (e.g. $\mu\to e\nu\bar\nu$ through the FFFF
track, top decays $t\to bW^*$ below threshold in BSM spectra), so it earns
its cost beyond the Higgs.

### What was built

`feynlag.pheno` gained four pieces, each general but exercised here on V–V\*:

- {mod}`~feynlag.pheno.propagator` — the massive-vector Breit–Wigner
  propagator (numerator $g_{\alpha\beta}-q_\alpha q_\beta/m^2$ + the
  $|q^2-m^2+im\Gamma|^{-2}$ factor).
- {class}`~feynlag.pheno.kinematics.ThreeBodyKinematics` — the Dalitz
  invariants $s_{12},s_{23}$, the $s_{12}$ bounds, and the on-shell `dot` map
  that reduces the covariant $|M|^2$ (the internal $q=p_2+p_3$ is just a tensor
  sum the existing engine contracts — no new trace machinery).
- {mod}`~feynlag.pheno.integrate` — the numerical layer: SciPy when the
  `[numeric]` extra is installed, else a numpy Gauss–Legendre fallback (SciPy
  lives in this one module).
- {mod}`~feynlag.pheno.offshell` — `scalar_vv_squared` (the covariant $|M|^2$),
  `offshell_scalar_vv_width` (analytic inner $s_{12}$ integral + numeric $q^2$
  integral), and `scalar_offshell_vv_width` (summed over the $V^*$ fermion
  channels, with the factor 2 for distinct $W^\pm$ vs. no factor for identical
  $ZZ$).

**Verified** against the Keung–Marciano closed form
$\Gamma(h\to VV^*)=\frac{3g_V^4 m_h}{512\pi^3}\delta_V R(x)$, $x=m_V^2/m_h^2$:
$\Gamma(h\to WW^*)=0.80$ MeV and $\Gamma(h\to ZZ^*)=0.089$ MeV, and by
narrow-width factorisation (a heavy scalar's $1\to3$ width → the Tier-1
$\Gamma(S\to VV)\times\mathrm{BR}$). The parity-violating $\gamma_5$/ε term
drops against the *symmetric* propagator + polarisation structure (the 1→3
analogue of the $\gamma_5$ argument in {doc}`decays` §15) — no ε-tensor algebra
needed for this topology. Worked model: `examples/sm_higgs_decays.py` (the BR
table now includes WW\*/ZZ\*). **Not yet built**: the FFFF and $t\to bW^*$
topologies (same infrastructure, a new assembler each).

## 16.3 Tier 3 — loop-induced $gg$, $\gamma\gamma$, $Z\gamma$

**Channels**: $gg$ (8.2%), $\gamma\gamma$ (0.23%), $Z\gamma$ (0.15%).
**Effort: moderate** — *if* one accepts the recommended route.

These vanish identically at tree level: the Higgs is electrically and colour
neutral, so $h\to gg/\gamma\gamma$ proceed only through a top-quark triangle
and (for $\gamma\gamma$) a $W$ loop. A tree-level library cannot reach them
from the Lagrangian, period.

**Recommended route: effective vertices.** The one-loop form factors are
standard closed forms — with $\tau = m_h^2/4m^2$ per loop particle,

$$\Gamma(h\to gg) = \frac{\alpha_s^2 m_h^3}{72\pi^3 v^2}
\left|\tfrac{3}{4}\textstyle\sum_q A_{1/2}(\tau_q)\right|^2, \qquad
\Gamma(h\to\gamma\gamma) = \frac{\alpha^2 m_h^3}{256\pi^3 v^2}
\left|A_1(\tau_W) + \textstyle\sum_f N_c Q_f^2 A_{1/2}(\tau_f)\right|^2,$$

where $A_{1/2}$ and $A_1$ are the fermion- and $W$-loop functions of the
Higgs Hunter's Guide. Implemented as `effective_hgg(...)` /
`effective_haa(...)` helpers that *return a coupling*, the width is then an
ordinary $1\to2$ the current engine already squares (`VVS` with massless
vectors — the $-g_{ab}$ polarisation sum is already in
{func}`~feynlag.pheno.amplitudes.polarization_sum`).

Say it plainly: this **imports known one-loop results instead of deriving
them** — a deliberate, documented exception to the derive-from-the-Lagrangian
ethos. There is precedent: the CKM sector ({doc}`flavor`) already takes the
FeynRules-style shortcut of *inserting* the standard-parametrisation $V$
rather than diagonalising symbolic Yukawas, because the honest general
computation is intractable and the physical answer is standard. The same
judgment applies here.

The alternative — a genuine one-loop engine (Passarino–Veltman tensor
reduction, scalar integrals $B_0/C_0$, UV renormalisation and a scheme
choice) — is a project of the same scale as the whole current library, and
already sits on the v2-deferred list next to $R_\xi$ gauges and ghosts. It
should stay there until someone wants NLO for its own sake.

## 16.4 Summary

| tier | channels | new machinery | BR gained | effort | ethos |
|---|---|---|---|---|---|
| 1 ✅ | $b\bar b,c\bar c,s\bar s,\mu\mu$ | `DiracParticle` abstraction (**done**); model content; running-mass inputs | → 67% | small | tree-level, in ethos |
| 2 ✅ | $WW^*, ZZ^*$ | propagators + $1\to3$ phase space + diagram assembly (**done**); FFFF/$t\to bW^*$ topologies remain | → 91% | large | tree-level, in ethos |
| 3 | $gg,\gamma\gamma,Z\gamma$ | effective one-loop vertices ($A_{1/2}, A_1$) | → 100% | moderate | documented exception (CKM precedent) |

**Tier 2 status**: the $V$–$V^*$ topology ($h\to WW^*$, $h\to ZZ^*$) is delivered
— `feynlag.pheno.offshell` reproduces the Keung–Marciano widths
($\Gamma(h\to WW^*)\approx0.80$ MeV, $\Gamma(h\to ZZ^*)\approx0.089$ MeV) and the
Higgs BR table is now $b\bar b$-dominated with $WW^*$ second (~25%), the canonical
shape. The general-$1\to3$ *topologies* the roadmap also names — the FFFF track
($\mu\to e\nu\nu$) and the fermionic $t\to bW^*$ — reuse the same propagator +
Dalitz infrastructure but are not yet assembled; they are the remaining Tier-2
generality.

The recommended order is the table order: Tier 1 fixes the qualitatively
wrong plot for the cost of a dataclass; Tier 2 is where the architecture
grows; Tier 3 is a bounded, well-understood graft.

See {doc}`decays` for the implemented engine, the tutorial
(`Particle_Decays_Tutorial.ipynb`) for the traps §16.1 makes
unrepresentable, and the v2-deferred list in the repository `CLAUDE.md` for
what stays out of scope.
