# 14. Quark Flavour Mixing (CKM)

```{note}
A **modelling** chapter, not a pipeline stage: it shows how feynlag handles
three-generation quark mixing without a symbolic diagonalization that does not
exist in closed form.
```

## Why insertion, not diagonalization

The CKM matrix is $V = U_u^\dagger U_d$, the mismatch between the unitary
rotations that diagonalize the up- and down-type Yukawas. For a *generic*
three-generation complex Yukawa those rotations have no closed symbolic form —
`sympy`'s `Matrix.diagonalize` returns unusable nested radicals (the same wall
`diagonalize_svd` hits, flagged in `examples/sm_scalar_gauge.py`). So feynlag
does what FeynRules' Standard Model does: work in the quark **mass basis** with
diagonal Yukawas and **insert** the CKM matrix directly into the charged
current,

$$\mathcal L_W = \frac{g}{\sqrt2}\,W^+_\mu\;\bar u_i\,\gamma^\mu P_L\,V_{ij}\,d_j
  \;+\;\text{h.c.}$$

The neutral currents ($Z$, $\gamma$, gluon) are written flavour-diagonal in the
mass basis, so they carry no CKM factor — the **GIM mechanism**, no
flavour-changing neutral currents.

## The unitarity trap

$V$ must be unitary. A matrix of nine independent symbols is *not*:
$\sum_k V^*_{ki}V_{kj}\neq\delta_{ij}$ symbolically, which would fake a
$Z$-current FCNC through the same rotation that produces the $W$ mixing. There
are two escapes, and feynlag uses both:

1. **Exact standard parametrization** —
   {func}`~feynlag.flavor.standard_ckm` builds $V$ from three mixing angles and
   one CP phase in the PDG convention, unitary as a trigonometric identity, so
   `V†V` collapses to the identity under `sympy.simplify`.
2. **Mass-basis insertion** — putting $V$ only in the $W$ current, never in the
   neutral currents, so GIM holds by construction.

The two-generation Cabibbo case *can* be driven through the ordinary real
mass-basis rotation machinery ({doc}`diagonalization`), and there the GIM
cancellation is an exact symbolic result: rotating the flavour-diagonal neutral
current by `rotation_2x2(θ_c)` gives back a diagonal current
($\cos^2+\sin^2=1$), while the charged current picks up the off-diagonal
$\sin\theta_c$ term. This is the physics ground truth pinned in
`tests/test_ckm.py::TestTwoGenerationGIM`.

## Parameters and UFO export

CKM elements are exported as **scalar** parameters — never an
`IndexedBase`/matrix, because `ParameterSet` is scalar-only and the UFO code
printer would emit an invalid `V[0,1]` subscript. `standard_ckm` returns:

- four **real external** parameters — the angles `th12`, `th13`, `th23` and the
  phase `deltaCP` (PDG central values);
- nine **complex internal** parameters `Vud … Vtb`, each defined by its
  standard-parametrization expression.

Internals are emitted `type='complex'` in dependency order, so the phase in
`Vub`/`Vtd` survives; {func}`~feynlag.verify.verify_ufo_numeric` evaluates the
whole set at export.

## Usage

```python
from feynlag import standard_ckm

params, V = standard_ckm()          # V is a 3×3 sympy Matrix, exactly unitary
L_W = g / sqrt(2) * Wp * sum(
    V[a, b] * Bilinear(ubar[a], gamma_mu_PL, d[b])
    for a in range(3) for b in range(3))
```

See `examples/sm_ckm.py` for the full worked demo (charged-current vertices,
GIM, UFO round-trip).

## Verification

`tests/test_ckm.py` pins: the two-generation GIM cancellation (mixing in $W$,
none in $Z$); the three-generation `V†V = 1` symbolically plus PDG magnitudes
$|V_{ud}|\approx0.974$, $|V_{us}|\approx0.225$, $|V_{cb}|\approx0.042$ and a
non-zero CP phase in $V_{ub}$; the inserted $W\bar u d$ vertex
$= i\,g/\sqrt2\,V_{ud}\,\gamma^\mu P_L$; and the complex CKM internals
round-tripping through UFO export.
```
