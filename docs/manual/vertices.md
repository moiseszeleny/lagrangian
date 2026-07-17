# 8. Extracting Vertices

## Physics statement

A Feynman vertex is read off the coefficient of a field monomial in the
Lagrangian — literally $n$ functional derivatives with respect to the
fields at that vertex, evaluated with every field set to zero. Once the
Lagrangian is in the physical basis ({doc}`diagonalization`), extracting
every vertex means: expand the Lagrangian, group terms by which fields (and
how many of each) appear, convert any derivative coupling to a
momentum-space tag, and multiply by the combinatorial factor that turns a
monomial coefficient into an actual Feynman rule.

## The two-track split, revisited

Bosonic fields are plain commuting `sympy.Symbol`s, so grouping "terms with
the same field content" is literally polynomial-coefficient extraction —
`vertices/extract.py` does this with SymPy's `Poly`. Fermion fields inside
a `Bilinear` are never commuting symbols in the same sense (a fermion
sandwich is a c-number *once fully contracted*, but its two legs are
`Indexed` objects wrapped in an opaque `Function`), so `vertices/bilinear.py`
runs a separate pass: group by which bilinear appears, then hand the
*coefficient* of each bilinear (a pure boson expression) to the same
`Poly`-based extractor for its boson legs. Both tracks converge on the same
Feynman-rule formula (below) and the same closed vertex catalog.

## 8.1 The bosonic extractor

`extract_interaction_coefficients(L, fields)` (`vertices/extract.py:68`)
tries a fast path first: `L.as_poly(*fields, domain='EX')`, iterating
`poly.terms()` to read off each monomial's exponent vector and coefficient
directly — `domain='EX'` (SymPy's "extended" domain) lets the polynomial
coefficients themselves be arbitrary symbolic expressions (couplings, VEVs,
$i$, momentum tags), not just rationals. If `as_poly` fails for any reason
(certain expressions don't coerce cleanly), `_extract_fallback` walks
`L.as_ordered_terms()` term by term, manually counting each field's
exponent by inspecting `Mul`/`Pow` structure — slower, but robust to
anything the fast path chokes on. Both paths produce the identical nested
dict `{n_fields: {sorted_field_tuple: coefficient}}`.

### Derivation: the Feynman-rule formula

For a Lagrangian monomial $\mathcal L \supset c\,\phi_1^{k_1}\phi_2^{k_2}
\cdots\phi_n^{k_n}$, the Feynman rule is $i$ times the fully-differentiated,
zero-field-evaluated coefficient:

$$
i\left.\frac{\partial^{k_1+\cdots+k_n}\mathcal L}{\partial\phi_1^{k_1}\cdots\partial\phi_n^{k_n}}\right|_{\phi=0}.
$$

Differentiating a single power $\phi^k$ exactly $k$ times with respect to
$\phi$ brings down $k(k-1)\cdots1 = k!$ and leaves a constant (no more
$\phi$-dependence, so setting $\phi=0$ afterward changes nothing); any
*cross* term (differentiating a monomial with respect to a field it doesn't
contain enough powers of) either still has surviving field dependence or
differentiates to zero, and vanishes at $\phi=0$ either way. So the whole
expression collapses to $c\cdot\prod_f k_f!$, i.e.

$$
\text{vertex} \;=\; i\times c\times\prod_f (\text{multiplicity of }f)! .
$$

`vertex_multiplicity(field_tuple)` (`extract.py:118`) computes $\prod_f
k_f!$ directly from a `collections.Counter` of the field tuple, and
`feynman_rule(coefficient, field_tuple)` (`extract.py:131`) combines it
with the $i\times c$ prefactor. The CONVENTIONS.md-pinned sanity check is
$\mathcal L = -\lambda\phi^4/4!$: here $c=-\lambda/4!$, $k=4$, so the
vertex is $i\cdot(-\lambda/4!)\cdot4! = -i\lambda$ — the familiar textbook
$\phi^4$ rule, recovered mechanically rather than hand-derived per model.

## 8.2 Momentum-space substitution

Any monomial that still contains `PartialMu` after `expand_derivatives`
({doc}`lagrangian`) gets the final substitution `∂_μφ → i p(φ) φ` via
`to_momentum_space`, called by `Model.interactions` whenever the physical
Lagrangian `has(PartialMu)`. The **sign** of this rule ($+i$, not $-i$) is
a deliberate convention choice, not a universal QFT fact — it corresponds
to momentum flowing *with* the field into the vertex under an $e^{+ip\cdot
x}$ plane-wave convention (matching the DLRSM1 reference this library was
built from), the opposite of the more common $\partial_\mu\phi\to-ip_\mu\phi$
convention some textbooks use for all-incoming momenta with $e^{-ip\cdot
x}$. The overall consistency of this choice — that it reproduces correct,
textbook-matching vertices once carried through every derivative coupling
in the model — is pinned by the VSS test cited below, not asserted on its
own.

## 8.3 The fermion bilinear track

`extract_fermion_vertices(L, boson_fields)` (`vertices/bilinear.py:135`)
first Leibniz/bilinear-expands `L`, then for every term in
`L.as_ordered_terms()`: finds the (exactly one — more than one raises,
since four-fermion operators are out of v1 scope) `Bilinear` atom, divides
it out (`sp.cancel(term / bil)`) to get a pure-boson coefficient, and
accumulates coefficients under the key `(bar, gamma, field)`. Each
accumulated coefficient is then handed to the *same*
`extract_interaction_coefficients` used for bosons, giving
`{(bar, gamma, field): {n_bosons: {boson_tuple: coeff}}}`.
`fermion_feynman_rule(coefficient, gamma, boson_tuple)`
(`vertices/bilinear.py:170`) applies the identical $i\times c\times\prod n!$
formula, now also carrying the Dirac structure $\Gamma$ along as an
explicit factor of the returned rule (it's not itself part of the
combinatorial counting — a `Bilinear` node counts as exactly one leg
regardless of $\Gamma$'s internal gamma-matrix content).

## 8.4 The closed vertex catalog

Every extracted `(field_tuple, coefficient)` pair is classified by
`classify_spins` (`vertices/vertex.py:33`) into one of exactly nine catalog
entries — `SSS, SSSS, VSS, VVS, VVSS, VVV, VVVV, FFS, FFV` — based purely on
the spins of the legs (looked up in a `{symbol: spin}` map built by
`Model.spin_map`). Any spin combination outside this list **raises**
`ValueError` rather than returning a best-guess classification: the catalog
is deliberately closed for v1 (R$_\xi$ ghosts, four-point fermion vertices,
and other combinations are out of scope, {doc}`pipeline`), and silently
mis-tagging an out-of-catalog vertex would be worse than refusing to
extract it. Each catalog entry also carries the UFO Lorentz-structure
name(s) it maps to (`LORENTZ_CATALOG`), consumed directly by
{doc}`export`.

## 8.5 Yang–Mills self-couplings

Rather than expanding $-\tfrac14F^a_{\mu\nu}F^{a\mu\nu}$ symbolically with
explicit Lorentz indices (which the rest of the pipeline deliberately
avoids, {doc}`lagrangian`), the pure-gauge cubic and quartic self-couplings
are computed **group-theoretically**, since their Lorentz structures are
universal (fixed once the spin content is VVV or VVVV) and only the
group-theory tensor multiplying them varies model to model.

### Derivation: cubic coupling

With $F^a_{\mu\nu} = \partial_\mu A^a_\nu - \partial_\nu A^a_\mu +
gf^{abc}A^b_\mu A^c_\nu$, the cubic self-interaction in $-\tfrac14F^aF^a$
comes from the cross term between the abelian piece and the non-abelian
piece:

$$
-\tfrac14\cdot2\cdot(\partial_\mu A^a_\nu-\partial_\nu A^a_\mu)\bigl(gf^{abc}A^{b\mu}A^{c\nu}\bigr)
\;=\; -g f^{abc}(\partial_\mu A^a_\nu)A^{b\mu}A^{c\nu},
$$

the group-index tensor being exactly $-g\,f^{abc}$. `cubic_couplings`
(`vertices/yangmills.py:70`) computes $f^{abc}$ from the fundamental
representation (`structure_constants`, $f^{abc}=-2i\,\mathrm{Tr}([T^a,T^b]T^c)$,
{doc}`declaration`) and, for a possibly-rotated physical basis $A^a =
\sum_iU_{ai}V_i$ (e.g. $W^1,W^2,W^3,B\to W^\pm,Z,\gamma$), contracts
$F_{ijk} = \sum_{abc}f^{abc}U_{ai}U_{bj}U_{ck}$ — the rotated tensor —
returning `{(V_i,V_j,V_k): -g·F_ijk}` for every nonzero triple.

### Derivation: quartic coupling

The quartic self-interaction comes from squaring the non-abelian piece
alone:

$$
-\tfrac14\bigl(gf^{abc}A^b_\mu A^c_\nu\bigr)\bigl(gf^{ade}A^{d\mu}A^{e\nu}\bigr)
\;=\; -\frac{g^2}{4}\sum_a f^{abc}f^{ade}\,A^b_\mu A^c_\nu A^{d\mu}A^{e\nu},
$$

summed (implicitly, repeated index) over the adjoint index $a$ that pairs
the two field-strength factors. `quartic_couplings`
(`vertices/yangmills.py:105`) builds the rotated pairwise tensor
$F^e_{ij} = \sum_{ab}f^{abe}U_{ai}U_{bj}$ and then $E_{(ij)(kl)} =
\sum_eF^e_{ij}F^e_{kl}$, matching the sum over $a$ above with $e$ playing
the role of the contracted adjoint index, returning
`{(V_i,V_j,V_k,V_l): -g²/4·Σ_e F^e_ij F^e_kl}` for that *ordered* quadruple.

### Derivation: the VVVV → 3 UFO Lorentz structures (`vvvv.py`)

UFO expresses a 4-vector self-coupling in exactly three independent metric
contractions, `VVVV1/2/3` (`export/ufo/lorentz_map.py`,
$g^{14}g^{23}-g^{13}g^{24}$ and cyclic). A single ordered quadruple
$(i,j,k,l)$ from `quartic_couplings` corresponds to one specific pairing of
the four legs; the other two independent pairings come from the two other
orderings, $(i,k,j,l)$ and $(i,l,j,k)$ — the three ways to partition 4
objects into 2 unordered pairs. `assemble_vvvv` (`export/ufo/vvvv.py:28`)
reads off exactly these three raw values:

```python
VVVV1 = -12 * quartic_raw[(i, j, k, l)]
VVVV2 = -12 * quartic_raw[(i, k, j, l)]
VVVV3 = -12 * quartic_raw[(i, l, j, k)]
```

The overall $-12$ normalization and the correctness of pairing three
*orderings* of the same raw tensor to the three UFO structures was not
assumed — it was **derived by direct functional differentiation** of
$-\tfrac14F^aF^a$ (an independent computation from `quartic_couplings`'s
group-theoretic route) and cross-checked against both SU(2) and SU(3),
including cases where more than one of the three terms is simultaneously
nonzero (`tests/test_yangmills.py`). This closes what `quartic_couplings`'s
own docstring had left as an unbuilt "Phase 5" step.

## Design gotchas

- **Never differentiate through `Bilinear`** — the extraction functions
  above only ever divide it out algebraically (`term / bil`) or match it
  structurally; see {doc}`invariance` for why differentiating through it
  is unsafe.
- **At most one `Bilinear` per term.** `extract_fermion_vertices` raises if
  it finds more than one — four-fermion operators are explicitly out of
  v1 scope (`Model.interactions`, {doc}`pipeline`), not silently mishandled.
- **`Model.interactions(..., min_legs=3)`** drops any extracted monomial
  with fewer than 3 legs by default — 1- and 2-point "interactions" are
  tadpole/mass terms, already handled in {doc}`ssb`/{doc}`masses`, and
  including them in a vertex table would be a spurious duplicate.

## Validation

- `tests/test_extract.py::test_pinned_symmetry_factor_phi4`,
  `::test_pinned_symmetry_factor_phi3`, `::test_round_trip_toy_potential`
  — the Feynman-rule formula, §8.1.
- `tests/test_gauge_sector_sm.py::test_ZhG0_vss_structure` — the momentum
  convention, §8.2.
- `tests/test_fermion_sector.py::TestGaugeCurrents` — the bilinear track,
  §8.3.
- `tests/test_gauge_sector_sm.py::test_vertex_objects_classified` — the
  closed catalog, §8.4.
- `tests/test_qcd.py::test_ggg_coupling_pinned`,
  `::test_gggg_coupling_pinned`, `tests/test_gauge_sector_sm.py::test_cubic_gauge_couplings`,
  `::test_quartic_gauge_couplings` — §8.5's cubic/quartic tensors, pinned
  numerically (`ggg = -g_s`).
- `tests/test_yangmills.py::TestVVVVAssemblyGroundTruth` — the independent
  functional-differentiation derivation of the VVVV assembly.

## Minimal snippet

```python
rules = model.feynman_rules([h, W1, W2])   # {field_tuple: i * coeff * n!}
vertices = model.vertices([h, W1, W2])     # typed Vertex objects
```
