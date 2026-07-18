# Roadmap: Status and Remaining Work

This page is the durable "what's done, what's left, how to resume" reference
for feynlag's adoption-focused development (packaging, validation, and physics
coverage beyond the core tree-level pipeline). Unlike a session's ephemeral
plan file, this is committed and discoverable in any future session.

## Status

Completed (branch `explore/adoption-roadmap`, 244 tests green, `pytest` ~6 min):

- **Packaging & CI** — PyPI metadata, GitHub Actions test matrix + build/publish
  workflow; `DiracFermion` fails fast with a pointer to the two-`WeylFermion`
  recipe instead of emitting unverified vertices.
- **`Model.validate()` umbrella** — aggregates gauge/discrete invariance,
  hermiticity, mass-dimension, gauge-anomaly cancellation
  (`feynlag.anomalies`), electric-charge conservation + vacuum-derived
  consistency (`feynlag.charges`), vertex-level hermiticity pairing, and a UFO
  numeric round-trip (`feynlag.verify.verify_ufo_numeric`) into one report.
- **CKM / quark flavor** (`feynlag.flavor`) — the FeynRules-SM insertion route
  (mass basis + CKM in the charged current only), not a symbolic 3×3 SVD;
  `standard_ckm()` gives an exactly-unitary matrix from the PDG parametrization.
- **MadGraph validation** (`docs/benchmark.md`, `scripts/madgraph_roundtrip.py`)
  — the exported SM UFO reproduces MadGraph's stock `sm` cross sections
  (e⁺e⁻→μ⁺μ⁻ and the gauge-cancelling e⁺e⁻→W⁺W⁻) to Monte-Carlo precision. This
  caught and fixed two real UFO-export bugs (relative imports, a missing
  Feynman-rule `i` in fermion couplings) — see `CLAUDE.md`'s "MadGraph
  round-trip" section for the full account.
- **D.3 — model-building tutorial** (`examples/ModelBuilding_Tutorial.ipynb`,
  executed, symlinked into `docs/tutorials/`) — walks the *model-building*
  workflow (as opposed to the analysis-pipeline tutorials): for a dark `U(1)_D`
  sector with symbolic charges, `feynlag.anomalies` derives the anomaly-free
  assignment (forcing the dark fermion vector-like), `feynlag.suggest`
  enumerates the invariant operator basis (and returns empty on a mistuned
  charge), and `build_lagrangian` assembles a validated model before the full
  pipeline gives `m_ZD²=g_D²q_S²v_D²` and `Z_D χχ = i g_D q_χ γ^μ`. No library
  changes — pure showcase of the already-built `suggest`/`anomalies` tools.

### How to resume

1. Check out `explore/adoption-roadmap` (or branch fresh off `main` if it has
   been merged) and run `pytest` — confirm still green before starting.
2. Read `CLAUDE.md` for architecture and the accumulated gotchas; the project's
   auto-memory file (outside this repo, in the Claude Code memory store)
   carries a session-by-session account of every phase and the bugs found
   along the way — a fresh session should consult it.
3. Pick a phase below; each entry names the concrete files/lines already
   scouted and the open decision, so a session can start implementing directly
   rather than re-exploring.

## Remaining phases

### C2 — four-fermion operators (medium effort)

Lifts the "exactly one `Bilinear` per term" restriction to support effective
operators like `(ψ̄Γψ)(χ̄Γ′χ)`.

- The guard is `vertices/bilinear.py:156-158`
  (`extract_fermion_vertices` raises "four-fermion operators are outside v1
  scope"); the `(bar, gamma, field)` grouping key at `bilinear.py:163-164`
  becomes a **two-bilinear** key for FFFF.
- `expand_bilinear` (`bilinear.py:104-132`) needs **no change** — it distributes
  per-`Bilinear`-node already, so it handles a product of two independently.
- `fermion_feynman_rule` (`bilinear.py:170-172`) needs a new rule combining
  Γ⊗Γ′ and identical-fermion Wick/symmetry factors — not just the existing
  boson-multiplicity factorial.
- Catalog: add `"FFFF"` to `LORENTZ_CATALOG` (`vertices/vertex.py:18-28`).
  `classify_spins` currently raises on four F's (`vertex.py:55-58`) because the
  sorted-letter scheme can't distinguish which fermion pairs with which — carry
  the pairing in `Vertex.meta`.
- UFO export: add `FFFF*` Lorentz structures (four spinor indices, e.g.
  `ProjM(2,1)*ProjM(4,3)` for a scalar-scalar current pair) to
  `export/ufo/lorentz_map.py` + `structures_for`, plus a four-fermion vertex
  adder and `write_ufo` kwarg in `export/ufo/writer.py`.
- Gauge/discrete invariance checking needs **no change** — `invariance.py`
  already transforms every fermion leg tree-wide regardless of bilinear count
  (`:165, 173-175`). `sympy.conjugate` distributes over a product of two
  `Bilinear`s automatically (`bilinear.py:65-78`), but the vertex-level
  hermiticity-pairing check (`charges.py:478-488`) is keyed on a single
  bilinear and needs extending to two-bilinear keys.
- Mass-dimension: `check_mass_dimension`'s math already gives the correct `u**6`
  for a four-fermion term, but `Model.check_invariance` hardcodes `max_dim=4`
  when calling it (`lagrangian.py:398-400`; the parameter already exists at
  `invariance.py:233`) — needs an EFT/`max_dim` flag threaded through.
- **The open decision**: spinor-contraction ordering / Fierz-identity basis —
  this is the actual physics reason the one-bilinear guard exists
  (`bilinear.py:11-12`), not just an arbitrary restriction. Resolve this before
  writing the extractor changes.

### C3 — R_ξ gauge fixing, Goldstone couplings, and ghosts (large effort — own plan)

The biggest remaining gap; expect a dedicated multi-session plan rather than a
single pass.

- **Decide first**: the README advertises a `FieldStrength` building block that
  was never implemented (`operators.py:10` defers it perpetually) — either
  build it now (the natural home for ghost-kinetic and gauge-fixing terms) or
  keep bolting gauge-fixing onto the existing group-theoretic route
  (`vertices/yangmills.py` builds VVV/VVVV from structure constants directly,
  never from a `−¼F_{μν}F^{μν}` Lagrangian term).
- The `V·∂G` kinetic-mixing term that R_ξ gauge-fixing exists to cancel is
  currently **invisible** in the pipeline: `gauge_mass_matrix` zeroes every
  derivative before differentiating (`vacuum/masses.py:92`), and
  `Model.interactions`'s `min_legs=3` default silently drops the 2-leg
  monomial (`lagrangian.py:271,297`). So correctness can't be checked via
  "mixing cancels to zero" — verify instead via ξ-dependent Goldstone/ghost
  masses and catalog completeness.
- Ghosts are spin-0 **Grassmann** fields, which collide with ordinary real
  scalars in the spin-letter vertex classifier (`vertices/vertex.py:30-59`) —
  there is no Grassmann/ghost marker on `Field` today. Needs: a ghost flag,
  new `UUV`/`UUS` catalog entries kept in sync across `vertex.py` and
  `export/ufo/lorentz_map.py`, and a dedicated ghost extraction track (ghost
  bilinears are Grassmann, so they can't go through the commuting-symbol
  `extract.py`).
- UFO ghost export is roughly 40% there: `export/ufo/static.py`'s vendored
  MadGraph `Particle` class already understands `spin=-1` (draws a dotted
  line), but feynlag's own `UFOParticle` (`writer.py:32-62`) has no ghost field
  and the writer's vertex-ordering/Lorentz-selection code assumes spin ∈
  {1,2,3}.

### D.2 — dim-5 Weinberg operator (small effort; do after C2)

Extends `suggest.py`'s operator enumeration (currently ≤ dimension 4) to the
dimension-5 `LLHH` operator, which generates a Majorana neutrino mass after
EWSB — pairing naturally with the existing `diagonalize_takagi`
(`vacuum/diagonalize.py`). Needs the EFT/`max_dim` opt-out that C2 introduces
for its own dim-6 four-fermion operators, so sequence this after C2 (or after
just that flag lands).

### ~~D.3 — model-building tutorial notebook~~ ✅ done

Delivered — see the Status section above (`examples/ModelBuilding_Tutorial.ipynb`).

## Suggested order

**~~D.3~~ → C2 (+ its `max_dim` flag) → D.2 → C3.** D.3 is done; C2 is
self-contained once the Fierz/ordering decision is made (resolved in the C2
session plan: as-written bilinear basis, distinct-legs-only for v1); D.2 rides
on C2's dimension-check plumbing; C3 is the long pole and deserves its own
dedicated plan, informed by whichever `FieldStrength` decision gets made.
