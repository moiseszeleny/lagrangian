# feynlag

Tree-level Feynman rules from Beyond-Standard-Model Lagrangians, in pure
[SymPy](https://sympy.org).

You declare particle fields with their gauge and discrete-symmetry
representations, write the Lagrangian explicitly with library building blocks
(`Dmu`, `FieldStrength`, `dag`), and `feynlag` takes it from there:

- gauge / discrete invariance and hermiticity checks,
- electroweak symmetry breaking: VEV expansion, tadpole conditions,
- mass-matrix extraction and diagonalization (orthogonal, unitary, SVD for
  Dirac fermions, Takagi for Majorana),
- rotation from the weak basis to the physical (mass) basis,
- vertex extraction (SSS, SSSS, VSS, VVS, VVSS, VVV, VVVV, FFS, FFV) with
  derivative couplings taken to momentum space,
- export: LaTeX vertex tables and UFO (MadGraph et al.).

Parameters are split into **external** (fixed by experiment, e.g. `v`, `m_h`,
`g`) and **internal** (derived: tadpole solutions, mixing angles, inverted
quartics), forming a dependency chain that closes the UFO parameter card.

## Documentation

Full docs, including an **Algorithms Manual** deriving the physics and
design of every pipeline stage (invariance checking, EWSB/tadpoles, mass
matrices, diagonalization, vertex extraction, export), tutorial notebooks,
an examples gallery, and the API reference:
**https://moiseszeleny.github.io/lagrangian/**

## Install (development)

```bash
pip install -e .[dev]
pytest
```

## Quick tour

```python
import sympy as sp
from feynlag import (ExternalParameter, InternalParameter, SU2, U1, Scalar,
                     Lagrangian, Model, Dmu, dag)

gw  = ExternalParameter("gw", 0.6535, positive=True)
g1  = ExternalParameter("g1", 0.3580, positive=True)
SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

v   = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
lam = ExternalParameter("lam", 0.129)
mu2 = InternalParameter("mu2", unit_dim=2)      # defined by the tadpole

H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
           component_names=["Gp", "H0"])
H.expand_vev({H.components[1]: v})              # H0 -> (v + h + i G0)/√2

HdH = (dag(H) * H.mat)[0]
DH  = Dmu(H)
L = Lagrangian()
L.add((dag(DH) * DH)[0], sector="kinetic")
L.add(mu2.s * HdH - lam.s * HdH**2, sector="potential")

m = Model("SM", gauge_groups=[SU2L, U1Y],
          fields=[H, SU2L.bosons("W"), U1Y.bosons("B")],
          parameters=[gw, g1, v, lam, mu2], lagrangian=L)

m.check_invariance()          # gauge invariance, hermiticity, dim ≤ 4
m.solve_tadpoles([mu2])       # mu2 = lam v², registered as internal
m.mass_matrix([sp.Symbol("H0_r", real=True)])   # [[2 lam v²]]
m.feynman_rules([...])        # vertices, momentum-space, i × n! included
```

See `examples/` for full runs: `sm_scalar_gauge.py` (complete SM: Higgs +
electroweak gauge + leptons + quark/QCD sector), `sm_vll.py` (SM + a
vector-like lepton doublet, biunitary mass-matrix diagonalization),
`sm_u1x.py` (SM × U(1)_X with a Z′, symbolic charges, chained rotations),
`thdm.py` (2HDM with the α rotation), `thdm_s3.py` (3HDM+S₃, where the
tadpole conditions force the √3 vacuum alignment). The
[docs site](https://moiseszeleny.github.io/lagrangian/) walks the SM, VLL,
and U(1)_X models stage by stage in three executed tutorial notebooks.

## Validation

The test suite pins the physics, not just the code (dual verification:
symbolic difference **and** random-point numeric checks):

- SM Higgs: `μ² = λv²`, `m_h² = 2λv²`, `h³ = −3i m_h²/v`, `h⁴ = −3i m_h²/v²`
- SM gauge: `m_W = gv/2`, Weinberg rotation, `hWW = i g m_W g^{μν}`,
  `γW⁺W⁻ = e`, `ZW⁺W⁻ = g cosθ_W`, scalar-QED Goldstone vertices
- SM leptons: `hℓℓ = −i m_ℓ/v`, `Wℓν = i g/√2 γ^μ P_L`,
  Z couplings ∝ `T³ − Q sin²θ_W`
- 2HDM: tadpoles, all three mass matrices and rotation angles vs the
  Gunion–Haber/Branco expressions
- 3HDM+S₃: invariant potential from the library's CG products; the tadpole
  system forces the `√3` alignment
- UFO: generated model imports cleanly; parameters resolve in dependency
  order; `hWW` coupling pinned numerically

## Status / roadmap

Working: scalars, gauge bosons, chiral fermions (bilinear track), tadpoles,
mass matrices (real/charged/gauge blocks), orthogonal/SVD/Takagi
diagonalization, momentum-space vertices for the closed catalog
(SSS SSSS VSS VVS VVSS VVV VVVV FFS FFV), LaTeX tables, UFO export
(including full SU(3) color-tensor strings for qqg/ggg/gggg).

Deferred (v2): R_ξ gauge fixing and ghosts, four-fermion operators,
NLO/UFO 2.0 extensions.
