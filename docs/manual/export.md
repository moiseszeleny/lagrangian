# 9. Export

## Physics statement

Extracted vertices ({doc}`vertices`) are only useful once they leave
`feynlag`'s internal representation: as a table for a paper, or as a model
directory a Monte Carlo generator can actually load. `feynlag` supports
both — a LaTeX vertex table, and a full UFO (Universal FeynRules Output)
model directory importable by MadGraph and compatible tools.

## LaTeX tables

`latex_feynman_table(interactions, simplify_coeff=None, extra_column=None)`
(`export/latex.py:27`) unifies three near-duplicate table generators from
the DLRSM1 reference into one configurable function. `_rows_from_interactions`
(`latex.py:12`) accepts either extraction output shape — the nested
`{n_fields: {field_tuple: coeff}}` from `extract_interaction_coefficients`
directly, or an already-flattened `{field_tuple: coeff}` dict — so it can
sit immediately downstream of either `Model.interactions` or a
hand-assembled rule table. `extra_column` optionally adds a third column
(e.g. a numerically simplified or series-approximated form of the same
coefficient) without duplicating the two-column code path.

## UFO export: the parameter closure requirement

A UFO `parameters.py` is a flat Python module where every `Parameter(...)`
object must be defined **before** any later parameter that references it in
its `value=` expression — i.e. it needs exactly the dependency-ordered list
{doc}`declaration` already builds. `_UFOBuilder._parameters_py`
(`export/ufo/writer.py:294`) calls `self.parameters.dependency_order()`
directly and emits one `Parameter(...)` line per entry in that order — the
UFO writer does not re-derive parameter ordering; it reuses
{class}`~feynlag.parameters.ParameterSet`'s topological sort verbatim.

## The vertex catalog → UFO Lorentz structures

`export/ufo/lorentz_map.py`'s `UFO_LORENTZ` dict is the concrete UFO string
for every entry of the closed catalog from {doc}`vertices` — e.g.
`VVS1 = "Metric(1,2)"`, `FFVL = "Gamma(3,2,-1)*ProjM(-1,1)"`. UFO's index
convention numbers *legs* (not Lorentz indices in the abstract): `P(l, n)`
is the momentum of leg `n` carried on Lorentz slot `l`; `Metric(m, n)`
contracts legs `m` and `n`; negative integers are UFO's convention for
internally-contracted (summed) indices, as in `Gamma(3,2,-1)*ProjM(-1,1)`
contracting the spinor index `-1` between the gamma matrix and the chiral
projector. `structures_for(vertex_type)` maps a catalog key to the list of
Lorentz-structure names UFO needs populated for that vertex type — e.g.
`FFV` always needs both `FFVL` and `FFVR` slots, even if one coupling is
zero (`_UFOBuilder.add_fermion_vertex` simply skips emitting a slot whose
coupling is exactly `0`).

**Four-fermion (FFFF)** operators export through
`_UFOBuilder.add_four_fermion_vertex` (and the `write_ufo(four_fermion_vertices=…)`
kwarg). The two Dirac chains sit on legs `(1,2)` and `(3,4)`; each chain mirrors
the single-fermion `FFSL`/`FFVL` spinor pattern, and the vector case shares one
internally-contracted Lorentz index across the chains, e.g.
`FFFFVLL = "Gamma(-1,2,-2)*ProjM(-2,1)*Gamma(-1,4,-3)*ProjM(-3,3)"` — the
repeated `-1` is the $\gamma^\mu\!\otimes\!\gamma_\mu$ contraction, `-2`/`-3`
the per-chain spinor dummies. There are eight such names (scalar⊗scalar and
vector⊗vector, each with the four L/R projector combinations); a mixed
scalar⊗vector structure is outside the v1 catalog and raises. As with the FFV
path, the adder applies the Feynman-rule `i` to each Lagrangian coefficient (its
omission is invisible in a single $|\mathcal M|^2$ but breaks interference).

**Export the h.c. vertex too.** A Hermitian Lagrangian's `op + h.c.` is *two*
four-fermion vertices (two distinct bilinear keys), and **MadGraph needs both**
to route fermion-number flow through the contact interaction — a UFO carrying
only one of the conjugate pair fails diagram generation with a
`NoDiagramException`, even though the single vertex is otherwise well-formed.
This is a MadGraph requirement, not a feynlag one (the exported Lorentz string
is byte-for-byte the one MadGraph's own `taudecay_UFO` uses for τ decay). The
validated recipe is in `scripts/madgraph_fermi.py`, which reproduces the muon
lifetime Γ(μ→eνν) = $G_F^2 m_\mu^5/192\pi^3$ from the exported Fermi UFO.

## SU(3): real color-tensor strings, one particle per gluon

Colored vertices need actual UFO color-tensor strings, not the hardcoded
singlet `'1'` used for EW self-couplings after EWSB:

- **`add_bosonic_vertex`** (the extractor-driven, color-singlet-only path
  used for Higgs/EW self-couplings) **raises** if any particle passed to it
  has `color != 1` — rather than silently emitting a wrong `color=['1']`
  for a vertex this code path genuinely doesn't know how to color-tensor.
  Colored self-couplings must go through the dedicated adders below.
- **`add_vvv_vertex`/`add_vvvv_vertex`** take an explicit `color=` (or
  `colors=`) string. A `ggg` vertex uses `color='f(1,2,3)'` — the three
  gluon legs at positions 1, 2, 3 directly contracted with the adjoint
  structure constant. A `gggg` vertex needs **three** color tensors, one
  per independent Lorentz structure (`VVVV1/2/3`, {doc}`vertices`),
  matching the three `f·f` color factors UFO expects for a 4-gluon vertex.
- **`add_fermion_vertex`**'s `color=` follows the leg ordering
  `[bar_symbol, field_symbol, boson]` (positions 1, 2, 3) — a `qqg` vertex
  uses `color='T(3,1,2)'`, i.e. $T^{a=\text{leg 3}}_{i=\text{leg 1},\,
  j=\text{leg 2}}$: the gluon's adjoint color index at leg 3, contracted
  with the quark/antiquark fundamental indices at legs 1 and 2 — matching
  both this adder's own argument order and `fermion_gauge_current`'s
  `T[r,c]` convention (`r` = bar-leg row index = leg 1, `c` = field-leg
  column index = leg 2, {doc}`vertices`).

Critically, **an unbroken non-abelian self-coupling is exported as one
physical particle referenced multiple times** (e.g. `ggg` triples the
single `g` UFO particle), color summed entirely inside the color-tensor
string — never as eight separate weak-basis adjoint components. The
group's full weak-basis component dictionary
(`group.bosons().components`) exists for internal symbolic verification
only ({doc}`vertices`'s `cubic_couplings`/`quartic_couplings`), not for
UFO particle declarations.

## Design gotchas

- **UFO cannot export symbolic gauge charges.** UFO's particle table calls
  `float()` on each particle's charge internally; a model like
  `examples/sm_u1x.py` that deliberately keeps its U(1)_X charge assignment
  symbolic (`X = a·Y + b·(B−L)`, {doc}`declaration`) cannot be exported to
  UFO without first fixing `a`, `b` to numbers. This is a hard limitation
  of the UFO format, not a gap in `feynlag`'s writer.
- **A `Parameter` referenced before its `dependency_order` slot is a bug,
  not a feature to special-case around** — if `_parameters_py` ever needs
  to reorder or retry, that indicates the underlying `ParameterSet` (or the
  caller's parameter list) is missing a dependency, and should be fixed at
  the declaration layer ({doc}`declaration`), not patched in the writer.

## Validation

- `tests/test_latex.py::test_table_from_nested_dict`,
  `::test_table_from_flat_dict_with_extra_column` — both accepted input
  shapes.
- `tests/test_ufo_export.py::test_ufo_imports`,
  `::test_ufo_parameters_resolve`, `::test_ufo_couplings_pinned` — the
  generated model actually imports, parameters resolve in dependency order,
  and `hWW` is pinned numerically.
- `tests/test_ufo_qcd.py::test_qqg_color_string`,
  `::test_ggg_color_string_and_coupling`,
  `::test_gggg_color_strings_and_couplings`,
  `::test_gluon_particle_is_color_octet_self_conjugate` — the SU(3)
  color-tensor strings and the one-particle-per-gluon convention.

## Minimal snippet

```python
from feynlag.export.ufo.writer import write_ufo

write_ufo("out/SM_UFO", "SM", model.parameters, particle_specs,
          bosonic_vertices=[...], vvv={...}, vvvv={...},
          fermion_vertices=[...])
```
