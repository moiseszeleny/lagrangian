# Tutorials

Four fully executed Jupyter notebooks, walking a worked model stage by
stage with real (stored) output — plots, mass matrices, Feynman rules. They
are tracked through the `nbstripout --keep-output` git filter (see the
repo's `CLAUDE.md`), so what you see below is exactly what re-running the
notebook produces.

```{toctree}
:maxdepth: 1

SM_Feynman_Rules_Tutorial
SM_VLL_Tutorial
SM_U1X_Tutorial
ModelBuilding_Tutorial
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

## Model Building Tutorial

Goes one step earlier than the others: instead of analyzing a
hand-written Lagrangian, it shows the *model-building* tools. For a dark
`U(1)_D` sector with symbolic charges it uses `feynlag.anomalies` to derive
the anomaly-free charge assignment (forcing the dark fermion to be
vector-like), `feynlag.suggest` to enumerate the invariant operator basis
(and catch a mistuned charge that admits no mass term), and
`build_lagrangian` to assemble a validated model before running the full
pipeline to the dark-photon mass and `Z_D χχ` coupling.
