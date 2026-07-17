# 6. Mass Matrices

## Physics statement

Once the vacuum is fixed ({doc}`ssb`), the physical mass spectrum is read
off the second derivatives of the potential (scalars) or the kinetic sector
(gauge bosons) at that vacuum — the Hessian of $V$ is, by definition, the
mass-squared matrix of the fluctuation fields. Fermion masses come from a
different place: the coefficient of the bilinear mass term in the
Yukawa/mass sector, evaluated at the vacuum.

## Real scalar fluctuations

`scalar_mass_matrix(potential, vacuum, fields, tadpole_subs=None)`
(`vacuum/masses.py:38`) is the direct implementation:
$M^2_{ij} = \partial^2V/\partial\phi_i\partial\phi_j\vert_{\rm vacuum}$,
computed with `sp.derive_by_array` applied twice
(`build_mass_matrix`, `masses.py:20`) over the vacuum-shifted potential,
then evaluated at the vacuum and tadpole-substituted (`_at_vacuum_matrix`,
`masses.py:29` — factor each nonzero entry so downstream diagonalization
sees the simplest possible symbolic form). `fields` here are **real
fluctuation symbols** — a CP-even block, a CP-odd block, or any other real
sub-block the caller wants a Hessian for.

## The Dummy-conjugate trick (complex/charged fields)

A charged scalar's physical mass term comes from
$M^2_{ij} = \partial^2V/\partial\bar\phi_i\partial\phi_j$ — differentiating
with respect to the *conjugate* field. SymPy has no notion of
differentiating with respect to `conjugate(phi)` directly (it treats
`conjugate(phi)` as a function *of* `phi`, not an independent variable, so
`sp.diff` against it either errors or returns something meaningless).
`charged_mass_matrix` (`vacuum/masses.py:55`) works around this with the
**Dummy-conjugate trick**, ported from the DLRSM1 reference:

1. Replace every `conjugate(field)` in the potential with a fresh, unrelated
   `sp.Dummy` symbol: `dummies = {conjugate(f): Dummy(f"{f.name}_conj") for f in fields}`.
2. Differentiate `build_mass_matrix` with respect to the **Dummies** (rows)
   and the original `fields` (columns) — now a completely ordinary
   independent-variable derivative, since a `Dummy` has no algebraic
   relationship to `f` as far as SymPy is concerned.
3. Substitute the Dummies back to `conjugate(field)` in the resulting
   matrix entries.
4. Evaluate at the vacuum and apply tadpole substitutions as before.

The trick works because $\partial^2V/\partial\bar\phi_i\partial\phi_j$ only
needs $\bar\phi_i$ to behave as an *independent* variable from $\phi_i$ for
the duration of the differentiation — which is exactly what "treat $\phi$
and $\bar\phi$ as independent for holomorphic/antiholomorphic calculus"
means in the Wirtinger-derivative sense physicists use implicitly when they
write $\partial V/\partial\phi^*$. Introducing a literal SymPy `Dummy` in
place of `conjugate(phi)` makes that independence explicit and mechanical
rather than relying on SymPy inferring it (which it does not).

## Gauge boson masses

`gauge_mass_matrix(kinetic, vacuum, gauge_components, tadpole_subs=None)`
(`vacuum/masses.py:74`) starts from the **kinetic sector** rather than the
potential: $\mathcal L_{\rm kin} = (D_\mu\phi)^\dagger(D^\mu\phi)$
evaluated at the vacuum. At the vacuum every scalar is constant (all
fluctuations set to zero, VEVs are genuine constants), so **every**
`PartialMu` term vanishes identically — the algorithm applies
`at_vac.replace(PartialMu, lambda arg: 0)` directly rather than going
through the momentum-space machinery, since there is nothing left to carry
a momentum tag once the derivative terms are gone. What survives is purely
the $-igT^aA^a_\mu\langle\phi\rangle$ piece of $D_\mu\phi$, squared — a
pure quadratic form in the gauge boson components, whose Hessian
$M^2_{ab} = \partial^2\mathcal L_{\rm kin,vac}/\partial A^a\partial A^b$
**is** the gauge mass matrix, with **no sign flip** (the vector mass term
in the Lagrangian is conventionally written $+\tfrac12 M^2_{ab}A^aA^b$,
matching how $(D_\mu\phi)^\dagger(D^\mu\phi)$ already carries the correct
overall sign for a kinetic-derived mass term, unlike the potential-derived
scalar case where $\mathcal L \supset -V$ flips the sign relative to a
naive Hessian of $\mathcal L$ itself).

## Fermion mass matrices

Fermion masses live on the other track entirely ({doc}`vertices`):
`fermion_mass_matrix(L_fermionic, bar_base, field_base, vacuum, nflavors,
indices, gamma=None)` (`vertices/bilinear.py:218`) collects the coefficient
of `Bilinear(bar_base[i], Γ, field_base[j])` in the vacuum-evaluated
fermionic Lagrangian (after `expand_bilinear` distributes any composite
legs), normalizes each term's flavor indices to the canonical `(i, j)`
symbols, and builds the matrix as $M_{ij} = -\text{coefficient}$ — the
minus sign because the Lagrangian mass term is conventionally written
$\mathcal L \supset -\bar\psi M\chi$ (CONVENTIONS.md), the fermionic analog
of the potential's $\mathcal L \supset -V$ sign flip above.

## Validation

- `tests/test_scalar_pipeline_sm.py::test_higgs_and_goldstone_masses` — the
  SM real/charged scalar Hessians, $m_h^2 = 2\lambda v^2$.
- `tests/test_scalar_pipeline_thdm.py::test_cp_even_mass_matrix`,
  `::test_charged_mass_matrix_and_mHp` — the Dummy-conjugate trick exercised
  on the 2HDM charged sector.
- `tests/test_gauge_sector_sm.py::test_gauge_mass_matrix` — the SM
  $W$/$B$ mass matrix from the kinetic sector, $m_W = gv/2$.
- `tests/test_fermion_sector.py::TestYukawa` — the fermion mass matrix from
  a Yukawa sector.

## Minimal snippet

```python
h = sp.Symbol("H0_r", real=True)
Mh2 = model.mass_matrix([h])                 # [[2 * lam * v**2]]
MW2 = model.gauge_mass_matrix([W1, W2, W3, B])
```
