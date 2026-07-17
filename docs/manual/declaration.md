# 2. Declaration: Parameters, Fields, Groups

## Physics statement

Before any Lagrangian can be written, a model needs three kinds of
declarations: **parameters** (what's fixed by experiment vs. derived from
the rest of the model), **fields** (particle content, with explicit gauge
and discrete representations), and **symmetry groups** (generator matrices
that tell the invariance checker, in {doc}`invariance`, how each field
transforms). `feynlag` treats all three as concrete SymPy objects from the
start — no abstract index notation, no deferred symbolic group theory.

## Parameters: external vs. internal

{class}`~feynlag.parameters.ExternalParameter` wraps a `sympy.Symbol` with a
numeric benchmark value — the couplings, VEVs, and Yukawas an experiment has
actually measured. {class}`~feynlag.parameters.InternalParameter` wraps a
symbol whose defining expression (`expr`) is filled in **later**, once a
downstream pipeline stage derives it — a tadpole solution, a mixing angle
from `tan 2θ`, an inverted quartic coupling. This mirrors the physics: you
don't know `μ²` until you've imposed the tadpole condition, but you need the
symbol `μ²` to exist before you can write down the potential that contains
it.

### Algorithm: `ParameterSet.dependency_order`

A UFO parameter card must assign every internal parameter *after* every
parameter it depends on. `ParameterSet` computes this order with **Kahn's
algorithm** for topological sorting (`parameters.py:176`):

1. For every internal parameter `p`, collect `deps[p] = free_symbols(p.expr) ∩ internal_symbols` (external-only dependencies need no ordering).
2. Repeatedly move any parameter whose entire `deps` set is already `resolved` into the output list, and add it to `resolved`.
3. If a pass finds no such parameter but some remain, the remaining set contains a cycle — raise, don't guess an order.

```python
ordered, resolved = [], set()
remaining = dict(deps)
while remaining:
    ready = [s for s, d in remaining.items() if d <= resolved]
    if not ready:
        raise ValueError(f"dependency cycle among internal parameters: {...}")
    for s in sorted(ready, key=str):
        ordered.append(...); resolved.add(s); del remaining[s]
```

`ParameterSet.resolve()` then substitutes progressively through this order
to express every internal parameter purely in terms of externals — exactly
what `Parameter.numeric()` needs to produce floats, and what
`export/ufo/writer.py` needs to emit a valid `parameters.py`.

## Fields: component expansion

A {class}`~feynlag.fields.Field` is declared with a `reps` dict
(`{GaugeGroup: representation_label}`) and expands eagerly into explicit
**component symbols** — the representation the rest of the pipeline works
in. There is no lazy "doublet" object surviving past construction; a
`Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1,2)})` immediately becomes two
plain symbols `H_1`, `H_2`.

- **Bosons** (`Scalar`, `GaugeBoson`) get plain `sympy.Symbol` components —
  they commute, so they can go through SymPy's `Poly` machinery directly
  ({doc}`vertices`).
- **Fermions** get `sympy.IndexedBase` components (`fields.py:239`,
  `Fermion._component_symbol`) — one `IndexedBase` per gauge component, each
  carrying an implicit flavor index (`psi[i]`), because a fermion field
  generically comes in several generations and every vertex-extraction and
  invariance check needs to preserve that index symbolically rather than
  unrolling it into `nflavors` separate symbols.

`Field.dim` (`fields.py:79`) is the product of `group.rep_dim(rep)` over
every non-abelian group in `reps` — a field charged under `SU2L: 2` and
`SU3c: 3` simultaneously (a quark doublet) has `dim = 6`.

### Generator matrices on the full component space

`Field.generators(group)` (`fields.py:106`) must return one matrix per
generator, each acting on the field's **full** component vector — but a
field can carry several simultaneous non-abelian reps (e.g. `SU2L: 2,
SU3c: 3`). The construction is a **Kronecker product** over the declared
reps, in declaration order:

$$
T^a_{\text{full}} \;=\; T^a_{G} \otimes \mathbb{1}_{d_1} \otimes \mathbb{1}_{d_2} \otimes \cdots
$$

with $T^a_G$ placed at the slot matching `group` and identity blocks
everywhere else — this is exactly the statement that the joint
representation of a field charged under $G_1 \times G_2$ is the tensor
product representation, and a generator of $G_1$ acts trivially on the
$G_2$ factor. The code builds this iteratively with
`sp.kronecker_product`, one factor at a time, rather than special-casing
the single-group case.

## Gauge groups: explicit generators, PDG normalization

`groups/gauge.py` hard-codes generator matrices rather than deriving them
from Lie-algebra data at runtime, in the standard hermitian, PDG-normalized
convention:

- **U(1)**: the single "generator" is the 1×1 matrix `[charge]` — a U(1)
  representation label *is* its charge.
- **SU(2)**: fundamental (`rep=2`) uses $T^a = \sigma^a/2$ (Pauli matrices);
  adjoint (`rep=3`) is built from the structure constants,
  $(T^a)_{bc} = -i f^{abc} = -i\,\varepsilon_{abc}$ for SU(2).
- **SU(3)**: fundamental (`rep=3`) uses $T^a = \lambda^a/2$ (Gell-Mann
  matrices); adjoint (`rep=8`) again from $(T^a)_{bc} = -if^{abc}$, with
  $f^{abc}$ computed directly from the fundamental generators via
  $f^{abc} = -2i\,\mathrm{Tr}([T^a,T^b]T^c)$.

### Why the adjoint formula is correct

The defining Lie-algebra relation is $[T^a, T^b] = if^{abc}T^c$. Writing
the adjoint generators as $(T^a_{\rm adj})_{bc} \equiv -if^{abc}$ and using
total antisymmetry of $f^{abc}$, one checks directly that these matrices
satisfy the same commutation relation with the *same* structure constants
— i.e. the adjoint representation is a genuine representation of the
algebra, not merely a bookkeeping convenience. `groups/gauge.py` computes
$f^{abc}$ *from* the fundamental (`-2i Tr([T^a,T^b]T^c)`) and then builds
the adjoint generators from that same $f^{abc}$, so the two representations
are guaranteed consistent by construction rather than by two independently
hand-typed numerical tables.

## Discrete groups: finite generators, multiplet assignment

`groups/discrete.py` follows the same "explicit generator matrix" design
for finite symmetries. `ZN` builds the irrep-$k$ generator as the $1\times1$
phase $\omega^k$, $\omega = e^{2\pi i/N}$. `S3` uses the real-orthogonal
2-dimensional irrep basis $\rho(a) = R(2\pi/3)$ (rotation, for the 3-cycle),
$\rho(b) = \mathrm{diag}(1,-1)$ (reflection, for a transposition) — chosen
specifically so mixed scalar/fermion multiplets never need complex
generator entries. `S3.doublet_product` supplies the $2\otimes 2 = 1 \oplus
1' \oplus 2$ Clebsch–Gordan decomposition needed to write an S₃-invariant
potential without hand-deriving it (used end to end in
`examples/thdm_s3.py`).

Discrete-symmetry membership is registered separately from `reps`, via
`group.assign(irrep, *multiplet)` — **not** as a field constructor
argument — because a discrete multiplet routinely spans *several*
`Field` objects (e.g. an S₃ doublet made of two separate Higgs SU(2)
doublets, `S3.assign('2', H1, H2)`), whereas `reps` is inherently
per-field. `assign()` also rejects multiplets that mix `Fermion` and
non-`Fermion` members: the two use structurally different substitution
mechanisms downstream (flat `xreplace` dict for bosons vs. index-preserving
`.replace()` for `Indexed` fermion components — see {doc}`invariance`), so a
mixed assignment would silently drop invariance checking for one side
rather than raising.

## Design gotchas

- **`bar_partner` registry** (`fields.py:32`): every `Fermion` populates a
  module-level dict at construction time mapping each component
  `IndexedBase` to its Dirac-adjoint `IndexedBase` and back. This lets
  `Bilinear._eval_conjugate` ({doc}`invariance`) find "the bar of this
  field" without threading a `Fermion` reference through every call site —
  a convenience that only works because component names are unique per
  model.
- **`DiracFermion` is untested and known-broken**: its `chirality=None`
  default makes `fermion_gauge_current` build `diracI*γ^μ` gauge-current
  structures that `dirac_conjugate` has no rule for (see {doc}`invariance`).
  Model a vector-like fermion as **two** `WeylFermion`s with identical reps
  instead (`examples/sm_vll.py`).
- **`GaugeBoson.reps`** only ever contains its *own* group in the adjoint —
  a `W` boson's `reps` don't record that it's also an SU(3) singlet, because
  singlet transformation is the default (`Field.generators` returns zero
  matrices for any group not in `reps`).

## Validation

- `tests/test_parameters.py::test_dependency_order_and_resolve`,
  `::test_cycle_detection` — the topological sort and its failure mode.
- `tests/test_fields.py::test_generators_doublet`,
  `::test_gauge_boson_from_group`, `::test_vev_expansion` — component and
  generator construction.
- `tests/test_groups.py::TestSU2`, `::TestSU3`, `::TestS3` — generator
  matrices satisfy the expected commutation relations and CG products.

## Minimal snippet

```python
import sympy as sp
from feynlag import SU2, U1, ExternalParameter, Scalar

gw = ExternalParameter("gw", 0.6535, positive=True)
g1 = ExternalParameter("g1", 0.3580, positive=True)
SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
           component_names=["Gp", "H0"])
H.generators(SU2L)   # [sigma^1/2, sigma^2/2, sigma^3/2] as 2x2 Matrices
```
