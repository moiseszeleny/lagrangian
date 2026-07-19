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
- **D.2 — Majorana infrastructure + the dim-5 Weinberg operator**
  (`feynlag.dirac.diracC`, `MajoranaBilinear`, `majorana_mass_matrix`,
  `extract_majorana_vertices`; `examples/sm_weinberg.py`, `tests/test_majorana.py`)
  — the roadmap under-scoped this as "just enumeration + Takagi wiring"; it in
  fact needed charge-conjugation machinery (`C=iγ²γ⁰`) and a same-chirality
  `ψᵀCΓψ` Majorana bilinear, built as first-class support (also unlocks type-I
  seesaw `½M_R ν_Rᵀ C ν_R` and triplet-LRSM masses). The Weinberg operator
  `(LᵀCεL)(HH)` gives `m_ν=−c v²/Λ` (Takagi-diagonalized) + `ν̄νh`/`ν̄νhh`
  couplings; `suggest_yukawa(max_dim=5)` enumerates it. **UFO export of the
  Majorana vertices is deferred** (see below). Fixed a latent
  `check_mass_dimension` bug (per-additive-term counting) en route.

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

### ~~C2 — four-fermion operators~~ ✅ done

Lifted the "exactly one `Bilinear` per term" restriction to support dim-6
effective operators like `(ψ̄Γψ)(χ̄Γ′χ)` (Fermi theory, SMEFT contact terms).
Delivered — see the Status section above and `CLAUDE.md`'s "The two-track
extraction design" for the full account.

- **The open decision (resolved)**: the as-written bilinear basis (no Fierz
  canonicalisation) restricted to **four distinct fermion components**; a
  repeated component raises `NotImplementedError` (cross-chain Wick
  contractions would need spinor-index Fierz algebra the opaque-`Bilinear`
  design can't express). With distinct legs there are no exchange
  contractions, so no new Wick/symmetry factor was needed — the rule is the
  plain scalar `i·coeff·∏(boson mult)!` with the two Dirac structures carried
  separately.
- Landed the **`max_dim` EFT flag** on `check_invariance`/`validate` (default
  4; pass 6 for four-fermion, 5 for Weinberg) — this is what **D.2 reuses**.
- `test_four_fermion.py` (22 tests), `examples/fermi_theory.py`. Optional
  end-to-end MadGraph muon-decay-width cross-check:
  `scripts/madgraph_fermi.py` (not in CI).

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

### ~~D.2 — dim-5 Weinberg operator~~ ✅ done (was mis-scoped as "small")

Delivered as the **Majorana-infrastructure** phase (see the Status section) — it
needed charge-conjugation support (`diracC`) and a same-chirality `ψᵀCΓψ`
`MajoranaBilinear`, not just an enumeration tweak. `suggest_yukawa(max_dim=5)`
enumerates `LᵀCεL HH`; `majorana_mass_matrix` + `diagonalize_takagi` give the
physical Majorana ν masses.

### ✅ Type-I seesaw + heavy-neutrino couplings (done, on top of D.2)

SM + 3 `ν_R` (Dirac Yukawa + large Majorana mass) end to end: `seesaw_mass_matrix`
/ `seesaw_light_mass` (`vacuum/masses.py`) build the `[[0,m_D],[m_Dᵀ,M_R]]` block
matrix and the `m_ν≈−m_D M_R⁻¹ m_Dᵀ` formula; `diagonalize_takagi` gives the
light+heavy spectrum; and a new charge-conjugation-aware `MajoranaRotation`
(`vacuum/diagonalize.py`) rotates the weak `ν_L`/`ν_R` into the physical Majorana
mass eigenstates (mixing `ν_L` with `ν_R^c`), so `extract_fermion_vertices`
yields the heavy-neutrino couplings `W ℓ̄ N=(g/√2)·V`, `Z ν̄ N∝(g_Z/2)V` with the
light–heavy mixing `V≈m_D/M_R` and its decoupling `M_R→∞`. `examples/sm_seesaw.py`,
`examples/SM_Seesaw_Tutorial.ipynb`, `tests/test_seesaw.py`.

### E — UFO export of Majorana vertices (new follow-up, from D.2)

The symbolic Majorana pipeline is complete, but `ν̄νh`/`ν̄νhh` Majorana vertices
are **not yet emitted to UFO**. Needs MadGraph's Majorana-fermion conventions
(`spin=2` self-conjugate particles, the `C`-carrying Lorentz structures, and the
fermion-flow handling MadGraph applies to Majorana lines) — its own area, akin to
the FFFF h.c.-pairing lesson (`docs/benchmark.md`). Until then
`MajoranaFermion`/`MajoranaBilinear` are symbolic-only.

### ~~D.3 — model-building tutorial notebook~~ ✅ done

Delivered — see the Status section above (`examples/ModelBuilding_Tutorial.ipynb`).

## Suggested order

**~~D.3~~ → ~~C2~~ → ~~D.2~~ → C3 / E.** D.3, C2, and D.2 are done. Remaining:
**C3** (R_ξ gauge fixing + ghosts — the long pole, deserves its own dedicated
plan, informed by the `FieldStrength` decision) and **E** (UFO Majorana export,
a smaller self-contained follow-up to D.2).
