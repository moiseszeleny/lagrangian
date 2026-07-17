# Tutorials

Three fully executed Jupyter notebooks, walking a worked model stage by
stage with real (stored) output — plots, mass matrices, Feynman rules. They
are tracked through the `nbstripout --keep-output` git filter (see the
repo's `CLAUDE.md`), so what you see below is exactly what re-running the
notebook produces.

```{toctree}
:maxdepth: 1

SM_Feynman_Rules_Tutorial
SM_VLL_Tutorial
SM_U1X_Tutorial
```

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
