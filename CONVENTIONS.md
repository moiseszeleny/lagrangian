# feynlag conventions

Fixed conventions for the whole library. Every item here has at least one pinned
test in `tests/`. Adapted from `bsm-calc/conventions/` and
`proyecto_3hdms3/CONVENTIONS.md`.

## Metric and Dirac algebra
- Metric signature **(+, −, −, −)**: `g = diag(1, -1, -1, -1)`.
- Clifford algebra: `{γ^μ, γ^ν} = 2 g^{μν} I₄`.
- `γ₅ = i γ⁰γ¹γ²γ³`.
- Chiral projectors: `P_L = (1 − γ₅)/2`, `P_R = (1 + γ₅)/2`.

## Lagrangian signs (mostly-plus for kinetic terms)
- Scalar kinetic: `+ (D_μ φ)† (D^μ φ)`.
- Fermion kinetic: `+ i ψ̄ γ^μ D_μ ψ`.
- Gauge kinetic: `− ¼ F_{μν} F^{μν}`.
- Fermion mass: `− m ψ̄ ψ`.
- Covariant derivative: `D_μ = ∂_μ − i g T^a A^a_μ` (all couplings with this sign).
- Yukawa: `L_Yuk = − Y ψ̄_L Φ ψ_R + h.c.`; `Φ̃ = i σ₂ Φ*`.

## VEVs and field expansion
- Neutral complex scalar expands with the **explicit 1/√2**:
  `φ⁰ → (v + h + i a)/√2`.
- VEV symbols and physical masses are declared `positive=True`.

## Feynman rules
- Vertex = `i × ∂ⁿL/∂φ₁…∂φₙ` evaluated at zero fields — equivalently
  `i × (monomial coefficient) × ∏_f (multiplicity of f)!`.
  Pinned test: `L = −λ/4! φ⁴` ⇒ vertex `−iλ`.
- All momenta **incoming**; `∂_μ φ → −i p_μ φ` for an incoming momentum
  convention with `e^{-ip·x}` plane waves. **feynlag uses `∂_μ φ → i p(φ) φ`
  matching the DLRSM1 convention** (momenta flowing with the field into the
  vertex via `e^{+ip·x}`); the overall convention is fixed by the pinned VSS
  test and documented in `operators.py`.

## 2→2 scattering kinematics
- `1(k₁) + 2(k₂) → 3(k₃) + 4(k₄)`, all-incoming/outgoing per the standard
  external-fermion-line table (not an all-incoming convention): incoming
  particle / outgoing antiparticle → field (ψ) slot; outgoing particle /
  incoming antiparticle → bar (ψ̄) slot.
- Mandelstam invariants `s = (k₁+k₂)² = (k₃+k₄)²`, `t = (k₁−k₃)² = (k₂−k₄)²`;
  `u = m₁²+m₂²+m₃²+m₄² − s − t` is always a **derived** quantity, never an
  independent free symbol — momentum conservation then holds by construction.
  `s` is `positive=True`; `t` is `real=True` (it is negative throughout the
  physical region — the "positive dummies under `sqrt`" rule does not apply
  to it).
- Flux factor `1/(2√λ(s,m₁²,m₂²))`; `dσ/dt = ⟨|M|²⟩/(16πλ(s,m₁²,m₂²))`.
- **Squared-amplitude functions/methods return the spin/colour-summed
  `|M|²`, never averaged.** Averaging over the initial state is declared
  data (`feynlag.pheno.particles.ExternalState.dof()`) and applied exactly
  once, by `feynlag.pheno.scattering.cross_section`/
  `differential_cross_section` — see `docs/manual/scattering_roadmap.md`.

## SymPy hygiene
- Arguments of `sqrt` must be manifestly positive: introduce positive dummy
  symbols for differences (e.g. `p1 = μ₃ − m_μ` with `p1 > 0`), never feed a
  raw difference to `sqrt`.
- Simplification hierarchy, cheapest first: `expand` → `collect` → `factor` →
  `simplify` (last resort).
- Rotation angles must be verified against `tan(2θ)` from the defining
  off-diagonal condition, not only `sin² + cos² = 1`.
- Results are exposed as **functions/lazy properties**, never computed at
  module import time.
- Dual verification everywhere: symbolic difference **and** random-point
  numeric check (`feynlag.verify.numeric_equal`).
