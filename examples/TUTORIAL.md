# Tutorial: Feynman rules of the Standard Model with feynlag

This walks through `sm_scalar_gauge.py` line by line — declaring the SM
electroweak sector, checking it, breaking the symmetry, diagonalizing, and
extracting Feynman rules — then shows how to turn it into a BSM model.

Run the finished script at any point with:

```bash
python examples/sm_scalar_gauge.py
```

All conventions used below (metric, `P_L`, sign of `D_μ`, the explicit `1/√2`
in VEVs, the vertex normalization) are fixed once in `CONVENTIONS.md` — that
file is the source of truth if a sign looks unfamiliar.

## 0. What we're building

The scalar + electroweak-gauge sector of the SM: an `SU(2)_L × U(1)_Y` Higgs
doublet with its potential and covariant kinetic term. This is enough to
extract the Higgs self-couplings, the gauge-boson masses, the Weinberg
rotation, and every `hVV`/`hhVV`/Goldstone vertex — the same pipeline stages
you need for *any* BSM scalar sector.

## 1. Declare the symmetries and parameters

```python
import sympy as sp
from feynlag import ExternalParameter, InternalParameter, SU2, U1

gw = ExternalParameter("gw", 0.6535, positive=True)
g1 = ExternalParameter("g1", 0.3580, positive=True)
SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

v   = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
lam = ExternalParameter("lam", 0.129)
mu2 = InternalParameter("mu2", unit_dim=2)
```

- `SU2`/`U1` are `GaugeGroup`s carrying explicit generator matrices
  (`σ^a/2` for SU(2)); they build their own gauge-boson fields on demand
  (§3).
- **`ExternalParameter`** is a number you feed in from experiment — it needs
  a `value` for benchmarking and later for UFO export.
- **`InternalParameter`** is a symbol whose defining expression will be
  *derived* later (here, by the tadpole condition in §5). `unit_dim` is the
  parameter's mass dimension, used by the dimension-≤-4 invariance check.
- `positive=True` matters: it lets SymPy simplify `sqrt` expressions
  involving `v` and `gw`/`g1` without spurious branch-cut assumptions
  (CONVENTIONS.md's "positive dummy symbols" rule).

## 2. Declare the fields

```python
from feynlag import Scalar

H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
           component_names=["Gp", "H0"])
```

`reps` assigns representations: `2` is the SU(2) doublet, `1/2` the
hypercharge. This alone generates the two component symbols `Gp` (complex,
the charged Goldstone-to-be) and `H0` (complex, the neutral component) plus a
conjugate registry (`dag(H)` uses it, and it's how `sp.conjugate(Gp)` is
tracked through the pipeline).

Register the vacuum expectation value on the neutral component only —
charge conservation forbids a VEV on `Gp`:

```python
H.expand_vev({H.components[1]: v})   # H0 -> (v + h + i·G0) / √2
```

This is the explicit-`1/√2` convention from CONVENTIONS.md; it introduces
two new **real** fluctuation symbols, `H0_r` (will become the physical Higgs
`h`) and `H0_i` (the neutral Goldstone `G0`), but doesn't touch the
Lagrangian yet — `expand_vev` only *registers* the decomposition.

## 3. Write the Lagrangian — FeynRules style

feynlag doesn't auto-generate the potential; you write it with library
building blocks, and the library checks it's legal.

```python
from feynlag import Dmu, Lagrangian, dag

HdH = (dag(H) * H.mat)[0]          # H†H, an SU(2)×U(1) invariant scalar
V   = -mu2.s * HdH + lam.s * HdH**2

DH  = Dmu(H)                        # covariant derivative, D_μ = ∂_μ − i g Tᵃ Aᵃ_μ
L = Lagrangian()
L.add((dag(DH) * DH)[0], sector="kinetic")
L.add(-V, sector="potential")       # L ⊃ −V, per CONVENTIONS.md
```

- `dag(H)` returns the row-vector Hermitian conjugate; `H.mat` is the
  column of components — their product is the standard invariant bilinear.
- `Dmu(H)` reads `H.reps` and automatically pulls in **both** `SU2L` and
  `U1Y` gauge bosons (created lazily the first time — see §4) with the
  correct generator matrices. You never write `g Wᵃ_μ σᵃ/2` by hand.
- Lagrangian terms are tagged by **sector** (`kinetic`, `potential`,
  `yukawa`, `gauge`, `other`). Sectors matter later: hermiticity is checked
  per sector, and you can extract vertices from one sector at a time.

## 4. Assemble the Model and check invariance

```python
from feynlag import Model

model = Model("SM-EW", gauge_groups=[SU2L, U1Y],
              fields=[H, SU2L.bosons("W"), U1Y.bosons("B")],
              parameters=[gw, g1, v, lam, mu2], lagrangian=L)

report = model.check_invariance()
print(report)          # InvarianceReport(N checks, OK)
```

`SU2L.bosons("W")` / `U1Y.bosons("B")` materialize the gauge-boson `Field`s
(three real components `W_1, W_2, W_3` and one `B`) — the same objects
`Dmu(H)` used internally, so they must be listed explicitly as fields of the
model.

`check_invariance()` is a real check, not a formality: for every term it
verifies (a) gauge invariance under each declared group by an infinitesimal
variation `δφ = iαᵀφ` and requiring every generator's coefficient to vanish,
(b) hermiticity per sector, (c) mass dimension ≤ 4. If you get a sign or a
hypercharge wrong, this is where it's caught — before any dead-end algebra.

## 5. Break the symmetry: tadpoles and masses

```python
model.solve_tadpoles([mu2])
```

This evaluates `V` on the vacuum, differentiates with respect to each
registered VEV (`∂V/∂v = 0`), solves for `mu2`, and **registers the
solution on the `InternalParameter`** — `mu2.expr` is now `lam.s * v.s**2`.
Every later stage (mass matrices, the physical Lagrangian, UFO export)
automatically substitutes it.

Scalar masses come from the Hessian of `V` at the vacuum:

```python
h, G0 = sp.Symbol("H0_r", real=True), sp.Symbol("H0_i", real=True)
model.mass_matrix([h])[0, 0]      # -> 2*lam*v**2  (m_h²)
```

`mass_matrix` takes real fluctuation symbols for CP-even/CP-odd blocks; pass
`charged=True` with the raw complex component (e.g. `H.components[0]`) for
charged mass matrices — it uses the Dummy-conjugate trick internally so
SymPy can differentiate with respect to `conjugate(φ)`.

Gauge-boson masses come from the *kinetic* sector evaluated on the vacuum
(scalars are constant there, so all `∂_μ` terms vanish and only the
`g²v²·(gauge field)²` piece survives):

```python
W1, W2, W3 = SU2L.bosons().components
B = U1Y.bosons().components[0]
M2 = model.gauge_mass_matrix([W1, W2, W3, B])
```

`W1, W2` decouple from `W3, B` at this order; the `(W3, B)` block is the
familiar singular matrix whose non-zero eigenvalue is `m_Z²` and whose zero
eigenvalue is the photon.

## 6. Rotate to the physical basis

A `Rotation` is `new = R · old`; register it and every later Lagrangian
extraction rewrites the weak-basis fields for you.

```python
from feynlag import Rotation, rotation_2x2

Z, A = sp.symbols("Z A", real=True)
thetaW = sp.atan(g1.s / gw.s)
model.rotate(Rotation([W3, B], [Z, A], rotation_2x2(-thetaW)))

Wp, Wm = sp.symbols("Wp Wm")
U = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
model.rotate(Rotation([W1, W2], [Wp, Wm], U, kind="unitary"))
```

- `rotation_2x2(θ)` is the standard `[[cosθ, sinθ], [−sinθ, cosθ]]`; the
  minus sign on `thetaW` gives the textbook convention
  `Z = cosθ_W W³ − sinθ_W B`. When you build a rotation from a mass matrix
  yourself (rather than a known angle), use
  `diagonalize_orthogonal_2x2(M, old, new, angle=...)`, which *derives* the
  angle from `tan 2θ = 2M₁₂/(M₁₁−M₂₂)` and attaches it as
  `rot.angle_relation` — the CONVENTIONS.md rule that any mixing angle must
  be checked against its `tan 2θ` source, not just `sin²+cos²=1`.
- `kind="unitary"` tells `Rotation` to use `R†` (not `Rᵀ`) as the inverse
  when it builds the weak→physical substitution — needed here because the
  charged combination `W± = (W1 ∓ iW2)/√2` is complex.
- **Complex Goldstone**: `Gp` has no real/imaginary split (it never got a
  VEV), so it stays a single complex physical field; its antiparticle is
  handled in §7 with `conjugate_pair`, not with a `Rotation`.

## 7. Extract the Feynman rules

```python
from feynlag import conjugate_pair

Gp = H.components[0]
Gm, cmap = conjugate_pair(Gp, "Gm")     # {conjugate(Gp): Gm}
fields = [h, G0, Gp, Gm, Z, A, Wp, Wm]

rules = model.feynman_rules(fields, conjugate_map=cmap, simplifier=sp.simplify)
```

What happens inside `feynman_rules`:

1. `physical_lagrangian()` takes the full `L`, applies the vacuum shift
   (`H0 → (v+h+iG0)/√2`), substitutes the tadpole solution for `mu2`, then
   applies every registered `Rotation`'s substitution dict — all lazily
   cached, nothing computed until you ask.
2. Because the kinetic sector has `∂_μ` on every field, the result still
   contains `PartialMu` heads. These get Leibniz-expanded and sent to
   momentum space, `∂_μ φ → i·p(φ)·φ` (the convention pinned in
   CONVENTIONS.md), tagging each leg with its own symbolic momentum
   `p(φ)`.
3. `conjugate_map` rewrites `conjugate(Gp)` → the plain symbol `Gm` so the
   commuting-symbol extractor can see it as an ordinary field leg.
4. The extractor collects every monomial's coefficient, and each vertex
   gets the standard normalization `i × coefficient × ∏(leg multiplicities)!`
   — so e.g. `L ⊃ −λ/4! φ⁴` reads off as `−iλ`, not `−iλ/24`.

Dict keys are the field tuple in SymPy's **canonical sort order**, not the
order you write them in — look them up with a small helper (this is exactly
what the test suite does):

```python
def rule(*fields):
    key = tuple(sorted(fields, key=lambda s: s.sort_key()))
    return rules[key]

rule(h, Wp, Wm)        # i g² v / 2  =  i g m_W        (hWW)
rule(h, h, Wp, Wm)     # i g² / 2                       (hhWW)
rule(A, Gp, Gm)         # QED vertex, ∝ (p(Gp) − p(Gm))  — a momentum-dependent VSS rule
```

If you want typed objects instead of a flat dict (useful for UFO export or
for grouping by vertex topology), use `model.vertices(...)` — same
arguments, returns `Vertex` objects classified into the closed catalog
`SSS, SSSS, VSS, VVS, VVSS, VVV, VVVV, FFS, FFV` by inspecting the spin
content of each leg.

Cubic/quartic pure-gauge self-couplings (`γW⁺W⁻`, `W⁺W⁻W⁺W⁻`, …) are *not*
produced by `feynman_rules` — they come from the group's structure constants
directly, since their Lorentz structure is universal:

```python
from feynlag import cubic_couplings, quartic_couplings

U_mix = ...   # 3×4 matrix: adjoint (W1,W2,W3) -> physical (Wp,Wm,Z,A)
triple  = cubic_couplings(SU2L, physical=[Wp, Wm, Z, A], U=U_mix)
quartic = quartic_couplings(SU2L, physical=[Wp, Wm, Z, A], U=U_mix)
```

## 8. LaTeX and UFO output

```python
from feynlag import latex_feynman_table
print(latex_feynman_table(rules))
```

produces a ready-to-paste `\begin{array}` table. For a MadGraph-importable
model directory, use `feynlag.export.ufo.write_ufo(...)` — see
`tests/test_ufo_export.py` for a complete worked call (it needs a
`UFOParticle` spec per physical field: PDG code, spin, mass parameter name,
color, charge).

## 9. Adding a lepton (fermion) sector — optional extension

The scalar/gauge pipeline above only needs `Scalar`/`GaugeBoson`. Chiral
fermions use a parallel track because a fermion bilinear is a c-number
"atom", not a product of commuting symbols:

```python
from feynlag import Bilinear, WeylFermion, diracPL, diracPR, fermion_gauge_current

Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                 chirality="L", nflavors=3, component_names=["nuL", "eL"])
eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                 component_names=["eR"])

# gauge current i ψ̄ γ^μ D_μ ψ, interaction part, all groups in Ll.reps:
current = fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i)
```

Yukawa terms are written by hand as `Bilinear(ψ̄[i], Γ, χ[j])` sandwiches
(`Γ` built from `diracPL`/`diracPR`), and `extract_fermion_vertices` +
`fermion_mass_matrix` peel off the boson legs / build the mass matrix the
same way the bosonic extractor does. See `tests/test_fermion_sector.py` for
the full SM-lepton worked example (Yukawa mass, `W`/`Z`/`γ` currents with the
correct `T³ − Q sin²θ_W` structure), and `diagonalize_svd` /
`diagonalize_takagi` for Dirac/Majorana mass diagonalization.

## 10. Extending the SM: writing a BSM model

Every BSM model follows exactly the ten steps above; the only things that
change are *what you declare in steps 1–3*. A worked template already lives
in `examples/thdm.py` (a second Higgs doublet) and `examples/thdm_s3.py` (a
discrete `S₃` flavor symmetry) — read them alongside this checklist:

1. **New fields**: declare extra `Scalar`/`Fermion`/`GaugeBoson`s with their
   representations under your (possibly extended) symmetry groups. Extra
   gauge groups are just more `SU2`/`U1`/`SU3` objects passed to
   `gauge_groups=[...]`; a gauged extra `U(1)'` is created exactly like
   `U1Y` above.
2. **New symmetries**: for a discrete symmetry (softly-broken `Z₂` picking
   out Yukawa alignment in a 2HDM, or an `S₃` flavor symmetry across three
   doublets), instantiate `ZN(name, N)` or `S3()`, then `group.assign(irrep,
   *fields)`. `S3.doublet_product(...)` gives you the `2⊗2 = 1⊕1'⊕2`
   Clebsch–Gordan contractions to build invariant terms without doing the
   group theory by hand — `thdm_s3.py` builds its entire potential this way.
3. **Extend the potential/Yukawas**: write the new invariant contractions
   (`dag(H1)*H2.mat`, CG products, …) and add them to a `Lagrangian` sector.
   Call `model.check_invariance()` immediately — a forbidden term (wrong
   representation, missing `+ h.c.`, wrong discrete charge) fails loudly
   instead of silently producing wrong Feynman rules downstream.
4. **VEVs on every scalar that gets one**, then `solve_tadpoles([...])` for
   as many mass parameters as you have independent tadpole conditions. If
   the system is *over*-constrained (more tadpole equations than free mass
   parameters — the `thdm_s3.py` case), that's physical: it's forcing a
   vacuum alignment, and you'll see it as a residual condition on the VEV
   ratios after solving the others (exactly how `thdm_s3.py` recovers the
   `√3` alignment).
5. **Mass matrices grow with the field content**: `model.mass_matrix([...])`
   accepts any list of same-block fluctuation symbols — a 3×3 CP-even block
   for a 3HDM works identically to the SM's 1×1. For non-diagonalizable-in-
   closed-form blocks (≥3×3 symbolic), don't call `sp.Matrix.eigenvects`
   directly; either parametrize a `MixingAnsatz`-style rotation and solve
   the off-diagonal conditions numerically at benchmark points, or export to
   UFO and diagonalize numerically downstream.
6. **Register every new rotation** the same way as `alpha`/`beta` in
   `thdm.py`'s CP-even block.
7. **Extract vertices exactly as in §7**, just with a longer `fields` list
   and (if you added fermions) the bilinear track from §9.

The invariant is: nothing about `Model`, `check_invariance`,
`solve_tadpoles`, `mass_matrix`, `rotate`, or `feynman_rules` is SM-specific
— they operate purely on whatever fields, symmetries, and Lagrangian terms
you declared. Growing the model is additive, not a rewrite.
