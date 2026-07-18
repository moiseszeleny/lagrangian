# feynlag

Tree-level Feynman rules from Beyond-Standard-Model Lagrangians, in pure
[SymPy](https://sympy.org).

You declare particle fields with their gauge and discrete-symmetry
representations, write the Lagrangian explicitly with library building
blocks (`Dmu`, `FieldStrength`, `dag`), and `feynlag` takes it from there:
gauge/discrete invariance and hermiticity checks, electroweak symmetry
breaking, mass-matrix diagonalization, rotation to the physical basis,
vertex extraction across the closed Lorentz catalog, and export to LaTeX
tables or UFO (MadGraph and friends).

The centerpiece of this site is the **Algorithms Manual**: one chapter per
pipeline stage, with the physics statement, the algorithm as it's actually
coded, a full derivation of the key results, the design gotchas that shaped
the implementation, and the pinned test that fixes each physical result.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc
Install, and the quick tour from the README.
:::

:::{grid-item-card} Algorithms Manual
:link: manual/pipeline
:link-type: doc
Full derivations, stage by stage: declare → write → check → break → diagonalize → extract → export.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Three executed notebooks walking the SM, VLL, and U(1)_X models end to end.
:::

:::{grid-item-card} Examples
:link: examples
:link-type: doc
Annotated gallery of the worked models in `examples/`.
:::

:::{grid-item-card} Conventions
:link: conventions
:link-type: doc
Every sign, metric, and normalization choice, pinned by tests.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc
Auto-generated reference for every public module.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting-started
manual/pipeline
manual/declaration
manual/lagrangian
manual/invariance
manual/ssb
manual/masses
manual/diagonalization
manual/vertices
manual/export
manual/verification
manual/suggest
manual/anomalies
manual/charges
manual/flavor
tutorials/index
examples
benchmark
roadmap
conventions
api/index
```
