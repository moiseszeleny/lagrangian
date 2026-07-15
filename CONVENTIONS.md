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
