# 11. Suggesting Invariant Terms

```{note}
This is an **exploration-branch** feature (`explore/model-building-tools`),
sitting *before* stage 2 ({doc}`lagrangian`) in the pipeline: instead of
hand-writing every Lagrangian term and relying on {doc}`invariance` to catch
mistakes, the tool *enumerates* the terms a given field content and symmetry
group actually admit.
```

## Physics statement

Given the declared fields and symmetries, only finitely many operators up to
mass dimension 4 are gauge- and discrete-invariant — that finite set *is* the
most general renormalizable Lagrangian the model can have. Writing it out by
hand is error-prone (a missed operator is a missed coupling; a wrong one
fails invariance). The term-suggestion tool (`feynlag.suggest`) enumerates
that set directly, turning model-building into "declare fields → review the
suggested operator basis → attach couplings". The engine is feynlag's own,
already-tested invariance machinery ({doc}`invariance`) used as an
**oracle**: candidates are constructed from invariant building blocks, then
every emitted term is re-verified against the full `check_*` battery.

## Algorithm

`suggest_potential` / `suggest_yukawa` share one pipeline:

1. **Building blocks.** Assemble the smallest non-abelian-gauge-invariant
   pieces, each tagged with its U(1) charge(s) and mass dimension. For
   scalars: singlet components and their conjugates (dim 1); doublet inner
   products $A^\dagger B$ (dim 2, charge $q_B-q_A$); SU(2) $\varepsilon$-
   contractions $A^T(i\sigma_2)B$ (dim 2, charge $q_A+q_B$). For Yukawas: the
   fermion bilinear $\bar F_L\Gamma F_R$ sandwiched with a scalar leg (direct
   doublet contraction, the $\tilde H$ pattern, or none for a bare mass), with
   any SU(3) color indices summed diagonally (the `qcolor` pattern of
   `examples/sm_scalar_gauge.py`).
2. **Monomials.** Take multisets of blocks with total dimension in the
   window (2–4 for the potential), keeping only those that are **exactly
   U(1)-neutral** (exact rational/symbolic charge arithmetic — symbolic
   charges like U(1)_X's `a·Y+b·(B−L)` are handled).
3. **Discrete projection.** Reynolds-project each monomial onto the
   discrete-invariant subspace (below); drop the ones that project to zero.
4. **Hermitian completion.** If a candidate isn't already hermitian, emit
   `m + m†` and flag `hc_added`.
5. **Basis reduction.** Expand every candidate into component monomials and
   keep only those that increase the exact rank of the coefficient matrix —
   a minimal linearly independent basis, matching literature operator counts.
6. **Oracle.** Assert every survivor passes `check_gauge_invariance`,
   `check_discrete_invariance`, `check_mass_dimension` and `check_hermiticity`
   ({doc}`invariance`) — the house rule that no result is trusted on
   construction alone.

## Derivation: the Reynolds (group-averaging) projector

For a **non-abelian** discrete symmetry like $S_3$, individual monomials are
*not* invariant — e.g. $H_1^\dagger H_1$ maps into a combination of
$H_1^\dagger H_1$ and $H_2^\dagger H_2$ under the group. Only specific linear
combinations are invariant, so enumerate-and-filter (which works for
rephasing $Z_N$ symmetries, where monomials are charge eigenstates) fails.

The fix is the classical invariant-theory tool. For a finite group $G$ acting
linearly on the space of monomials, define the **Reynolds operator**

$$
P(m) \;=\; \frac{1}{|G|}\sum_{g\in G} g\cdot m .
$$

$P$ is a projector onto the invariant subspace: it is idempotent
($P^2 = P$, since averaging an already-averaged — hence invariant —
expression leaves it unchanged) and its image is exactly the $G$-invariants
(for invariant $m$, every $g\cdot m = m$ so $P(m)=m$; conversely any $P(m)$
is invariant because left-multiplying the sum by a fixed $g'$ just permutes
the group elements). So $P(m)$ is either $0$ (the monomial has no invariant
component) or a genuine invariant combination — precisely the
$(H_1^\dagger H_1 + H_2^\dagger H_2)/2$-type combinations that no single
monomial realizes. `reynolds_project` (`suggest.py`) builds this by
enumerating the group as **words in its generators** (`_group_elements`,
telling elements apart by their action in the faithful direct sum of irreps)
and applying each element **letter by letter** through the existing
per-generator substitution maps — sequential substitution *is* composition,
so no new group-action code is needed beyond what {doc}`invariance` already
has. For a $Z_N$-charged monomial, $P$ collapses to the familiar
"total charge $\equiv 0 \pmod N$" filter automatically (the average is $m$
if neutral, $0$ if charged).

## Design gotchas

- **SymPy caches structurally-equal expressions to one object**, so two
  value-equal candidates (e.g. the $\lambda_5$ operator $(H_1^\dagger H_2)^2$
  and $(H_2^\dagger H_1)^2$, identical after h.c. completion) share an `id`
  and cannot be told apart by identity. The basis reducer therefore returns
  and filters by *positional index*, never `id()`.
- **`conjugate(φ)` is not an independent polynomial variable** to SymPy, so
  the rank reducer substitutes each `conjugate(...)` node — and each opaque
  `Bilinear` atom — by a stable shared `Dummy` before extracting coefficient
  vectors (the same trick `charged_mass_matrix` uses, {doc}`masses`).
- **Reynolds over multiple discrete groups** composes the per-group
  projectors; this lands on the joint-invariant subspace only when the group
  actions commute (true for independently-declared symmetries). The oracle
  re-verifies regardless, so a pathological case fails loudly rather than
  silently emitting a non-invariant term.
- **`DiracFermion` is unsupported** (as everywhere in feynlag — model a
  vector-like fermion as two `WeylFermion`s, {doc}`declaration`); `suggest_yukawa`
  pairs opposite-chirality Weyls.

## Validation

`tests/test_suggest.py` pins the enumeration against the hand-built example
models via a **span-equality** helper (both operator sets expand to
component-monomial coefficient matrices with equal exact row rank, plus the
dual random-point numeric philosophy of {doc}`verification`):

- SM potential spans $\{H^\dagger H,\ (H^\dagger H)^2\}$.
- 2HDM with an exact $Z_2$ → the 7-operator basis; the soft $m_{12}^2$ term
  is correctly absent (and present when no $Z_2$ is declared).
- 3HDM+$S_3$ → the 10-operator basis, spanning `examples/thdm_s3.py`'s
  CG-built potential — the Reynolds ground-truth test.
- U(1)_X singlet → the portal basis, no bare (X-charged) $S$/$S^2$/$S^3$.
- SM Yukawa (e via $H$, d via $H$, u via $\tilde H$, with the SU(3) color
  sum); VLL bare Dirac mass + mixing Yukawa (`examples/sm_vll.py`).
- a parametrized oracle battery: every suggestion passes the full
  `check_*` set.

## Minimal snippet

```python
from feynlag import suggest_potential, suggest_yukawa, build_lagrangian

pot = suggest_potential([H1, H2, HS], [SU2L, U1Y], discrete_groups=[s3])
yuk = suggest_yukawa([QL, uR, dR], [H], [SU2L, U1Y, SU3c])
for t in pot:
    print(t)                       # e.g. SuggestedTerm((H1†H1) [potential, dim 2])

L, couplings = build_lagrangian(pot + yuk)   # fresh c1, c2, ... attached
```
