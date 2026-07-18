# Validated against MadGraph

The strongest correctness statement a Lagrangian→Feynman-rule tool can make is
that its output computes the *right numbers* in a downstream generator.
feynlag's Standard-Model UFO is round-tripped through **MadGraph5_aMC@NLO**: the
exported model is imported, real cross sections are computed, and they are
compared to the standard `sm` model shipped with MadGraph at the *same*
parameter point.

Reproduce it with:

```bash
python scripts/madgraph_roundtrip.py
```

which downloads MadGraph if needed, exports the SM UFO
(`scripts/export_sm_ufo.py`), and runs both models on both processes.

## Parameter point

The export uses MadGraph's stock electroweak input scheme
$(\alpha_{\rm EW}^{-1}, G_F, M_Z) = (132.50698,\ 1.16639\times10^{-5},\
91.1876)$, from which the same tree-level relations the stock `sm` uses give

$$M_W = 80.4185\ \text{GeV},\quad M_Z = 91.1876\ \text{GeV},\quad v = 246.22\ \text{GeV},$$

so the two models have numerically identical couplings by construction.
Lepton-collider beams: $\sqrt s = 200$ GeV, no PDF (`lpp=0`).

## Cross sections

| Process | feynlag UFO | stock `sm` | agreement |
|---|---|---|---|
| $e^+e^-\to\mu^+\mu^-$ | $2.7878 \pm 0.0027$ pb | $2.7878 \pm 0.0027$ pb | 0.00% (0.0σ) |
| $e^+e^-\to W^+W^-$ | $19.498 \pm 0.058$ pb | $19.497 \pm 0.058$ pb | 0.01% (0.0σ) |

Both agree with the stock model to within Monte-Carlo error.

## What each process validates

- **$e^+e^-\to\mu^+\mu^-$** exercises the photon and $Z$ FFV couplings and the
  $Z$ propagator (γ and $Z$ $s$-channel interference).
- **$e^+e^-\to W^+W^-$** is the **gauge-cancellation acid test**. The $\nu$
  $t$-channel diagram grows with energy and is tamed only by a precise
  cancellation against the $\gamma/Z$ $s$-channel diagrams — which requires the
  relative signs across the $W\bar\nu e$ FFV vertex and the $WW\gamma/WWZ$
  triple-gauge vertices to be exactly right. A wrong relative sign gives
  ≈98 pb instead of ≈19.5 pb; getting 19.5 pb confirms the cancellation holds.

## Bugs the round-trip caught

Building this benchmark surfaced two real UFO-export bugs (both fixed, now
regression-tested) and one convention subtlety — exactly the kind of thing that
is invisible to unit tests but breaks a real generator run:

1. **Relative imports** in the generated UFO (`from .object_library …`) broke
   MadGraph's param-card generation, which runs the modules as standalone
   scripts. Real UFO models use absolute imports; feynlag now does too
   (`test_ufo_export.py::test_ufo_uses_absolute_imports`).
2. **The FFV export dropped the Feynman-rule $i$** that the UFO coupling
   convention carries (the bosonic and VVV couplings already had it). This is
   invisible in $e^+e^-\to\mu^+\mu^-$ (an overall phase cancels in $|\mathcal
   M|^2$) but breaks the FFV↔VVV interference in $e^+e^-\to W^+W^-$. Fixed in
   `add_fermion_vertex`.
3. **The triple-gauge coupling** built from the complex $W^\pm$ rotation comes
   out of `cubic_couplings` with the opposite overall sign to MadGraph's VVV1
   convention; the export flips it (the real-basis QCD $ggg=-g_s$ is unaffected
   and already matches). See `scripts/export_sm_ufo.py`.

This is the payoff of feynlag's verification-first design: the round-trip is a
harness that turns "the model looks right" into "the model computes the right
cross section."
