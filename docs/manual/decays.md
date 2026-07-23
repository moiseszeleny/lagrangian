# 15. Decay Widths

## Physics statement

A vertex is not yet an observable. The bridge from a Feynman rule to
something measurable is the **partial width**: for a particle of mass $M$
decaying to two daughters of mass $m_1, m_2$,

$$
\Gamma = \frac{1}{2M}\int \overline{|\mathcal{M}|^2}\,\mathrm{d}\Phi_2
 = \overline{|\mathcal{M}|^2}\;
   \frac{\sqrt{\lambda(M^2,m_1^2,m_2^2)}}{16\pi M^3},
$$

with the Källén function
$\lambda(x,y,z) = x^2+y^2+z^2-2xy-2yz-2zx$ and
$\overline{|\mathcal{M}|^2}$ summed over final-state spins and *averaged*
over the parent's. Everything model-dependent sits in that one factor, and
computing it is what `feynlag.pheno` does.

## 15.1 Covariant, not explicit

There are two ways to square an amplitude with fermions in it.

**Explicit matrices.** Represent $\gamma^\mu$ as literal $4\times4$
matrices, pick a frame, and take `Matrix.trace()`. Exact, handles $\gamma_5$
trivially, needs no index algebra.

**Covariant traces.** Apply the trace theorems symbolically, keeping Lorentz
indices abstract, and reduce to invariants $p_i\cdot p_j$ at the end.

This module takes the **covariant** route, because $\overline{|\mathcal{M}|^2}$
then comes out frame-independent and in the form a physicist writes down —
and because it extends naturally to $2\to2$ in Mandelstam variables. The
explicit-matrix evaluator is kept as the *test oracle* (§15.6), not on the
compute path.

The traces themselves are SymPy's:
{mod}`sympy.physics.hep.gamma_matrices` implements the Clifford algebra and
Kahane simplification, so the library stays pure-SymPy with no new
dependency. Three things it does not cover are supplied by
`pheno/lorentz.py`.

### Gap 1 — mass terms

`gamma_trace` handles pure gamma products. A spin sum $\not p + m$ is a
`TensAdd` whose scalar argument has no `.sorted_components()`, and the call
raises `AttributeError`. `dirac_trace` expands and splits: terms carrying
gammas go to SymPy, purely scalar terms use $\mathrm{Tr}[c\,\mathbb{1}_4]=4c$.

### Gap 2 — $\gamma_5$ and chiral projectors

SymPy has no `G5`, so $P_{L,R}$ cannot be represented at all. The reduction
in `reduce_projectors` is exact algebra up to one dropped term:

$$
P_{L,R}\,\gamma^\mu = \gamma^\mu P_{R,L},\qquad P^2 = P,\qquad P_LP_R = 0,
$$

which pushes every projector to one end of the chain, leaving

$$
\mathrm{Tr}[X\,P_{L,R}] = \tfrac12\mathrm{Tr}[X] \mp \tfrac12\mathrm{Tr}[X\gamma_5].
$$

The second term is a totally antisymmetric $\epsilon$ tensor. **It vanishes
identically for a $1\to2$ decay**, and the argument is worth stating because
the code depends on it: $\epsilon^{\mu\nu\rho\sigma}$ needs four independent
four-vectors to be non-zero. A two-body final state supplies only two
independent momenta ($P = p_1+p_2$ is not a third), and any leftover free
index is contracted against a polarization sum
$-g_{ab} + P_aP_b/M^2$ that is **symmetric** in $(a,b)$, while $\epsilon$ is
antisymmetric. So every surviving contraction is zero.

That argument fails for $1\to3$. `reduce_projectors` therefore takes the
number of independent momenta and free indices as arguments and **raises
`NotImplementedError`** rather than dropping the term when the precondition
does not hold — a later extension fails loudly instead of returning a
plausible wrong number.

### Gap 3 — reduction to scalars

`TensExpr.replace_with_arrays` raises `ValueError: p1(L_0) not found` on
contracted dummy indices (SymPy 1.14), so `contract_to_dots` walks the
structure directly: for each `TensMul` it aligns `components` with
`get_indices()`, matches each dummy to its partner, and maps the pair to an
on-shell dot product. This is the one function touching semi-private SymPy
tensor API; it is isolated deliberately and pinned by tests, so a future
SymPy change breaks exactly one place with a clear message.

## 15.2 Spin and polarization sums

$$
\sum_s u\bar u = \not p + m, \qquad
\sum_s v\bar v = \not p - m, \qquad
\sum_\text{pol} \epsilon_a\epsilon^*_b = -g_{ab} + \frac{p_ap_b}{m^2}.
$$

For $m=0$ the longitudinal piece is absent and the polarization sum reduces
to $-g_{ab}$; this is legitimate here because every vertex feynlag extracts
is contracted with a conserved current, so the gauge-dependent remainder
cancels in $|\mathcal{M}|^2$.

## 15.3 Worked case: $V \to f\bar f$

With $\Gamma^\mu = \gamma^\mu(g_LP_L + g_RP_R)$ and
$\bar\Gamma^\mu = \gamma^\mu(\bar g_LP_L + \bar g_RP_R)$ — a vector current is
self-conjugate, the projector does *not* swap
({func}`~feynlag.dirac.dirac_conjugate`) — pushing projectors through gives

- chirality-diagonal: $\tfrac12\mathrm{Tr}[\not p_1\gamma^a\not p_2\gamma^b]$,
  the mass terms killed by $P_LP_R=0$;
- chirality-mixing: $-m_1m_2\mathrm{Tr}[\gamma^a\gamma^bP_R] = -2m_1m_2g^{ab}$,
  where the momentum terms die instead.

Contracting with the polarization sum and averaging over the parent's three
polarizations:

$$
\Gamma(V\to f\bar f) = \frac{M\beta}{24\pi}
\left[(g_L^2+g_R^2)\left(1-\frac{m^2}{M^2}\right)
      + 6\,g_Lg_R\frac{m^2}{M^2}\right].
$$

Massless limit with $g_L=g_R=g$: $\Gamma = g^2M/12\pi$. Pure $V-A$
($g_R=0$): exactly half that, which is the closed-form fingerprint of the
$\gamma_5$ handling.

## 15.4 Two Weyl legs, one particle

`DiracFermion` raises on construction in feynlag (see
{doc}`declaration`) — a Dirac fermion is modelled as two `WeylFermion`s.
So `e_L` and `e_R` are *different legs*, and
{func}`~feynlag.vertices.bilinear.extract_fermion_vertices` reports the
left- and right-handed currents under different keys, even though
$Z\to e^+e^-$ is **one** channel fed by both.

`DecayCalculator` therefore takes a `particle_map`:

```python
particle_map = {eL[i]: e, eR[i]: e, eLbar[i]: ebar, eRbar[i]: ebar}
```

Omitting it does not error — it silently produces two half-channels, each
missing the other chirality and the $g_Lg_R$ mass interference. This is the
single most likely way to get a wrong number out of the module.

## 15.5 Thresholds: symbolic vs. numeric

With symbolic masses the comparison $M \ge m_1+m_2$ is undecidable, so
`is_allowed` returns `None` and the channel is *kept* — correct, since the
user may later substitute masses that open it.

The trap is what happens next. Substituting numbers into a channel that
turns out to be closed gives $\lambda < 0$, so $\sqrt\lambda$ is
**imaginary** — and a complex width silently poisons the total width and
every branching ratio. $h\to W^+W^-$ at $m_h = 125\,\text{GeV}$ is exactly
this case.

Use `numeric_partial_widths` / `numeric_branching_ratios`, which re-test the
threshold *after* substitution and set closed channels to exactly zero, and
not `numeric(partial_widths(...))`, which does not.

## 15.6 Validation

Every width is checked two ways (`tests/test_pheno.py`):

1. against an **independent explicit-$4\times4$-matrix oracle** that shares no
   code with the covariant engine, builds $\not p$ and $\Gamma$ as literal
   Dirac-basis matrices, and carries $\gamma_5$ as an actual matrix — so it
   validates the $\gamma_5$ drop rather than re-assuming it;
2. against the textbook closed form.

Pinned results:

| quantity | value |
|---|---|
| $\Gamma(V\to f\bar f)$, $m\to0$ | $g^2M/12\pi$ |
| $\Gamma(W\to\ell\nu)$ | $g^2m_W/48\pi$ |
| $\Gamma(Z\to\nu\bar\nu)$ | $m_Z(g^2+g'^2)/96\pi = m_Z^3/24\pi v^2$ |
| $\Gamma(Z\to\ell^+\ell^-)$ | $m_Z(5g'^4-2g^2g'^2+g^4)/96\pi(g^2+g'^2)$ |
| $\Gamma(h\to f\bar f)$ | $N_c\,m_h m_f^2\beta^3/8\pi v^2$ |
| $\Gamma(h\to VV)$ | $g^2m_h^3/64\pi m_V^2\cdot\sqrt{1-4x}(1-4x+12x^2)$ |

The $\beta^3$ in $h\to f\bar f$ is the CP-even scalar signature (a CP-odd
scalar gives $\beta^1$), and $h\to ZZ$ carries the $1/2!$ identical-particle
factor while $h\to W^+W^-$ does not.

## 15.7 Scope

Implemented: **SSS, FFS, FFV, VVS** — the three-leg catalog entries reachable
in a $1\to2$ decay without derivative couplings.

Not implemented: **VSS** and **VVV**, which carry momentum tags $p(\phi)$ from
{func}`~feynlag.operators.to_momentum_space`; squaring them needs
derivative-coupling support. They raise rather than being silently skipped.
Also out of scope for this stage: $1\to3$ decays, loop-induced channels
($h\to\gamma\gamma$), and interference between distinct diagrams feeding the
same final state.

For a tiered analysis of what it would take to add all of these — the full
Higgs branching-ratio picture, from the tree-level quark channels through
off-shell $WW^*$ to the loop-induced $gg/\gamma\gamma$ — see
{doc}`decays_roadmap`.

## Minimal example

```python
from feynlag.pheno import DecayCalculator

calc = DecayCalculator(
    model, masses={Z: mZ, h: mh, e: me, ebar: me, nu: 0, nubar: 0},
    boson_fields=[h, Z, Wp, Wm],
    fermion_sectors=("gauge", "yukawa"),
    conjugate_map=cmap,
    particle_map={eL[i]: e, eR[i]: e, eLbar[i]: ebar, eRbar[i]: ebar},
    parameters=model.parameters,
)

calc.partial_widths(Z)                       # symbolic, per channel
calc.numeric_branching_ratios(Z, extra={mZ: 91.1876, me: 0.000511})
```

See `examples/sm_decays.py` for the full worked model.
