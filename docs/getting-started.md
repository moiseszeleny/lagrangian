# Getting Started

## Install

```bash
pip install -e .[dev]
pytest
```

`feynlag` depends only on SymPy at runtime. The `dev` extra adds pytest,
numpy (for the numeric cross-checks in `feynlag.verify`), and nbstripout
(notebook diff hygiene — see the repo's `CLAUDE.md`).

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
H.expand_vev({H.components[1]: v})              # H0 -> (v + h + i G0)/sqrt(2)

HdH = (dag(H) * H.mat)[0]
DH  = Dmu(H)
L = Lagrangian()
L.add((dag(DH) * DH)[0], sector="kinetic")
L.add(mu2.s * HdH - lam.s * HdH**2, sector="potential")

m = Model("SM", gauge_groups=[SU2L, U1Y],
          fields=[H, SU2L.bosons("W"), U1Y.bosons("B")],
          parameters=[gw, g1, v, lam, mu2], lagrangian=L)

m.check_invariance()          # gauge invariance, hermiticity, dim <= 4
m.solve_tadpoles([mu2])       # mu2 = lam v^2, registered as internal
m.mass_matrix([sp.Symbol("H0_r", real=True)])   # [[2 lam v^2]]
m.feynman_rules([...])        # vertices, momentum-space, i x n! included
```

Every stage above corresponds to one chapter of the
{doc}`Algorithms Manual <manual/pipeline>`. For a complete worked model, read
`examples/sm_scalar_gauge.py` alongside {doc}`manual/pipeline`, or run the
{doc}`SM tutorial notebook <tutorials/index>`.

## Validation

The test suite pins the physics, not just the code (dual verification:
symbolic difference **and** random-point numeric checks — see
{doc}`manual/verification`):

- SM Higgs: `mu^2 = lam*v^2`, `m_h^2 = 2*lam*v^2`, `h^3 = -3i m_h^2/v`, `h^4 = -3i m_h^2/v^2`
- SM gauge: `m_W = g*v/2`, Weinberg rotation, `hWW = i g m_W g^{mu nu}`,
  `gamma*W+*W- = e`, `Z*W+*W- = g cos(theta_W)`, scalar-QED Goldstone vertices
- SM leptons: `h*l*l = -i m_l/v`, `W*l*nu = i g/sqrt(2) gamma^mu P_L`,
  Z couplings proportional to `T^3 - Q sin^2(theta_W)`
- 2HDM: tadpoles, all three mass matrices and rotation angles vs the
  Gunion-Haber/Branco expressions
- 3HDM+S3: invariant potential from the library's CG products; the tadpole
  system forces the sqrt(3) alignment
- UFO: generated model imports cleanly; parameters resolve in dependency
  order; `hWW` coupling pinned numerically

## Status / roadmap

Working: scalars, gauge bosons, chiral fermions (bilinear track), tadpoles,
mass matrices (real/charged/gauge blocks), orthogonal/SVD/Takagi
diagonalization, momentum-space vertices for the closed catalog
(SSS SSSS VSS VVS VVSS VVV VVVV FFS FFV), LaTeX tables, UFO export
(including full SU(3) color-tensor strings).

Deferred (v2): R_xi gauge fixing and ghosts, four-fermion operators,
NLO/UFO 2.0 extensions.
