# 12. Gauge-Anomaly Cancellation

```{note}
Like {doc}`suggest`, this is a **model-building** check that sits *beside* the
declaration stage rather than inside the Feynman-rule pipeline: given only the
fermion representations, it decides whether the gauge theory is even
quantum-mechanically consistent — before any Lagrangian is written.
```

## Physics statement

A chiral gauge theory is consistent only if its triangle anomalies cancel.
Each anomaly is a sum over the (Weyl) fermion content of a group-theory factor
times the fermions' charges; if any sum fails to vanish the gauge current is
not conserved at the quantum level and the theory is inconsistent. For a new
U(1) — a $Z'$, gauged $B-L$, a flavour symmetry — anomaly cancellation is the
first non-trivial constraint on the allowed charge assignment, and with a
symbolic charge it becomes a set of polynomial equations to solve.

`feynlag.anomalies` computes every coefficient symbolically straight from the
representations already declared for a {class}`~feynlag.lagrangian.Model`, so
it works hand-in-hand with the symbolic-charge models the rest of the library
supports (e.g. `examples/sm_u1x.py`'s $X=aY+b(B-L)$).

## Conventions

Everything is reduced to **left-handed Weyl** fermions. A field declared
`chirality='R'` is the conjugate of a left-handed field, so in the reduction
all its U(1) charges flip sign and every non-abelian representation goes to its
conjugate. The Dynkin index $T(R)=T(\bar R)$ is conjugation-invariant, so only
the cubic $[SU(N)]^3$ anomaly ($A(\bar R)=-A(R)$) and the charge signs feel the
flip.

## Coefficients

For non-abelian groups $G$ (with Dynkin index $T$ and cubic index $A$) and
abelian factors $U(1)_a$, over all left-reduced fermions with total
multiplicity $M$ (flavours × product of representation dimensions):

| Anomaly | Coefficient |
|---|---|
| $[U(1)_a][U(1)_b][U(1)_c]$ | $\sum M\, q_a q_b q_c$ |
| $\mathrm{grav}^2\text{–}U(1)_a$ | $\sum M\, q_a$ |
| $[G]^2\text{–}U(1)_a$ | $\sum (M/\dim G)\, T(R_G)\, q_a$ |
| $[G]^3$ | $\sum (M/\dim G)\, \mathrm{sign}\, A(R_G)$ |
| Witten–$G$ (SU(2) only) | number of doublets (must be even) |

The Dynkin index is computed generically from the generators as
$T(R)=\frac{1}{\dim G}\sum_a \mathrm{Tr}(T^a T^a)$; the cubic index is non-zero
only for a complex representation — among the supported reps that is the SU(3)
fundamental ($A(\mathbf 3)=1$).

## Usage

```python
from feynlag import Model, check_anomaly_free, anomaly_coefficients

model = Model("SM_BL", gauge_groups=[SU3c, SU2L, U1Y, U1BL], fields=fermions)

report = check_anomaly_free(model)     # AnomalyReport
report.ok                              # True for the SM + 3 nuR
report.nonzero                         # {} — nothing failed

coeffs = anomaly_coefficients(model)   # {name: symbolic coefficient}
coeffs["[U1BL][U1BL][U1BL]"]           # 0  (cancels only once nuR is added)
```

With a symbolic charge the coefficient is a polynomial constraint:
`coeffs["[U1X][U1X][U1X]"]` for a single right-handed field of charge $x$ is
$-x^3$, i.e. the anomaly-free condition is a genuine equation in $x$ rather
than a number. This is the ingredient for solving for anomaly-free $Z'$ charge
assignments symbolically.

## Verification

`tests/test_anomalies.py` pins the textbook results: the Standard Model with
three right-handed neutrinos is anomaly-free under
$SU(3)_c\times SU(2)_L\times U(1)_Y\times U(1)_{B-L}$; every individual
coefficient (including the mixed $[SU(N)]^2$–U(1) and the colour cubic)
vanishes exactly; dropping $\nu_R$ leaves $[U(1)_{B-L}]^3\neq 0$; and a
mistuned hypercharge breaks cancellation.
```
