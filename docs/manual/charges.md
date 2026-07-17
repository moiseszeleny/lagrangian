# 13. Electric Charge and the Vertex Checks

```{note}
Like {doc}`anomalies`, these are **model-consistency** checks that run on the
extracted vertices — they close out the {doc}`verification` umbrella
(`Model.validate`) with the two checks that need per-leg electric charge.
```

## Physics statement

Two things every consistent theory must satisfy show up at the vertex level:
**electric charge is conserved at every vertex** ($\sum_\text{legs} Q = 0$), and
the interaction Lagrangian is **hermitian**, so every vertex is accompanied by
its conjugate with the conjugated coupling. Checking these needs the electric
charge of each *physical* leg — but physical fields (`h`, `Z`, `W±`) are ad-hoc
symbols created during diagonalization and carry no quantum numbers.

`feynlag.charges` supplies them two ways, and cross-checks one against the
other:

- a {class}`~feynlag.charges.ChargeRegistry` — the user's declared
  `{physical field: charge}` map (the same information `UFOParticle.charge`
  needs at export), auto-completed with conjugate and Dirac-adjoint partners;
- an independent **derivation from the vacuum**: electric charge is the
  combination of diagonal generators and U(1) charges that leaves the vacuum
  invariant.

## Charge from the vacuum — no hard-coded `Q = T3 + Y`

Electric charge is *defined* by the symmetry breaking: it is the generator
combination that annihilates every VEV,

$$Q = \sum_a c_a\,T^a_\text{diag} + \sum_i c_i\,q_{U(1)_i},\qquad Q\,\langle\phi\rangle = 0.$$

{func}`~feynlag.charges.derive_charge_operator` builds the candidate diagonal
operators (each U(1) charge; the Cartan generators of each non-abelian group —
excluding any group under which no VEV'd scalar transforms, so colour and
spectator U(1)s drop out), imposes $Q\langle\phi\rangle=0$ for every VEV as a
linear system, and returns the **null space**. This is model-independent:

- **Standard Model** — the Higgs VEV in the neutral doublet component forces
  $c_{T3}=c_Y$, i.e. $Q\propto T3+Y$ falls out;
- **Left–right models** — the bidoublet/triplet VEVs give $Q\propto T3_L+T3_R+(B-L)/2$;
- **SM×U(1)_X** — an X-charged Higgs pushes X out of the unbroken combination.

The null space fixes $Q$ only up to scale (pinned against the declared map) and
raises loudly if the vacuum breaks everything (no unbroken charge) or leaves two
independent U(1)s (charge genuinely ambiguous).

{func}`~feynlag.charges.physical_charges` propagates these weak-basis charges to
the physical fields through the registered `Rotation`s
($Q_\text{new}=R\,Q_\text{old}\,R^{-1}$) — this is what turns the neutral,
non-eigenstate `W1`/`W2` into `W±` with charge $\pm1$ **automatically**, and
raises if a physical field is not a charge eigenstate.

## The three checks

- {func}`~feynlag.charges.check_charge_consistency` — declared charges vs the
  vacuum-derived operator, catching an assignment inconsistent with the reps.
- {func}`~feynlag.charges.check_charge_conservation` — $\sum Q=0$ on every
  bosonic vertex and every fermion bilinear (`ψ̄ Γ ψ` + bosons), using the
  registry's automatic $-Q$ for the Dirac-adjoint leg.
- {func}`~feynlag.charges.check_hermiticity_pairing` — every vertex pairs with
  its conjugate (legs relabelled to antiparticles) with coupling
  $(-1)^{1+d}[\text{coupling}]^*$, where $d$ is the number of derivative
  (momentum) tags; a missing partner or a mismatched coupling is a dropped
  `h.c.`.

## Usage

```python
from feynlag import ChargeRegistry

report = model.validate(
    charges={h: 0, Gp: 1, Gm: -1, Z: 0, A: 0, Wp: 1, Wm: -1},
    fields=[h, G0, Gp, Gm, Z, A, Wp, Wm],
    conjugate_map=cmap,
    conjugates={Gp: Gm, Gm: Gp, Wp: Wm, Wm: Wp},
)
print(report.summary())
#   charge_conservation: ok — ChargeConservationReport(38 vertices, ok)
#   charge_consistency: ok — ChargeConsistencyReport(consistent)
#   hermiticity_pairing: ok — HermiticityPairingReport(30 vertices, ok)
```

`conjugates` is the full antiparticle pairing (both directions) for every
non-self-conjugate physical field — the charged `W±` come from a rotation, not
`conjugate_pair`, so the checker cannot infer the pairing on its own.

## Verification

`tests/test_charges.py` pins the physics: the derived operator is $T3+Y$ (with
`W±` charges appearing automatically from the rotation), the declared charges
are consistent, every SM-lite vertex conserves charge and pairs hermitian, a
fermion current $\bar\nu\gamma e\,W^+$ conserves charge, and the guards fire on
a charge-breaking vacuum, a mistuned declaration, an ambiguous charge, and an
unknown leg.
