"""3HDM-S₃ decay sector: physical basis, gauge couplings and LFV rates.

Third module of the arc.  `model.py` carries the scalar sector, `fermions.py`
the Yukawa sectors; this one finally *registers the physical basis on the
Model* — the step `NOTES.md` recorded as notebook 03's blocker — adds the
electroweak kinetic terms, and turns the resulting vertices into widths.

**General δ, not the aligned limit.**  The physical CP-even basis is
``R_S = R_A · R_H(δ)`` with ``δ = α − θ_v`` ([LFVHD] Eqs. RS_AH, R_H).  δ is
kept **symbolic** throughout: the draft's Scenario A is δ→0 and Scenario B is
δ→π/2, and both are recovered as limits rather than assumed up front.  δ is
also not a free dial — α is the residual CP-even 2×2 mixing angle that
`model.cp_even_angle` computes, so at any benchmark point the λ's *predict*
where the model sits among the scenarios.

Two exact, δ-independent consequences fall straight out (both asserted in
`03_scalar_decays.ipynb`): ``R_H`` acts only in the 1–3 plane, so ``h_0``'s
column of ``R_S`` is untouched and stays orthogonal to the vacuum direction.
Hence for **any** δ

    g(h_0 V V) = 0   exactly,        Q_2 is independent of δ,

i.e. ``h_0`` is gauge-phobic while keeping its lepton-flavour-violating
entries.

Performance note: the bosonic extractor is run over the ``kinetic`` and
``potential`` sectors only.  Letting it see the Yukawa sector as well costs
~4 minutes for six boson legs (measured) because it must walk past every
``Bilinear``; the fermionic vertices come from the separate bilinear track
anyway, so the sectors are genuinely disjoint.

References
----------
[LFVHD] M. Zeleny-Mora, M. Mondragón, T. A. Valencia-Pérez, "Exploring LFV
    Higgs decays in the Three Higgs Doublet Model", draft (`LFVHD_3HDMS3.tex`).
[GomezBock21] Eur. Phys. J. C 81, 942 (2021), arXiv:2102.02800.  Eq. (29) the
    geometric rotation; Eqs. (54)–(55) the alignment scenarios.
[CMS21] CMS Collaboration, Phys. Rev. D 104, 032013 (2021), arXiv:2105.03007 —
    B(H→μτ) < 0.15%, B(H→eτ) < 0.22% at 95% CL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _field
from pathlib import Path

import sympy as sp

from feynlag import (Dmu, ExternalParameter, Lagrangian, Model, Rotation,
                     charged_current_rotation, conjugate_pair, dag,
                     weinberg_rotation)

import fermions as F
from model import build_model, cp_even_angle, rotation

__all__ = [
    "DecayModel", "build_decay_model", "boson_vertices", "vv_couplings",
    "delta_of_point", "load_points", "point_substitution", "LFV_LIMITS",
]

#: 95% CL observed upper limits on LFV Higgs branching ratios, [CMS21].
LFV_LIMITS = {("tau", "mu"): 1.5e-3, ("tau", "e"): 2.2e-3}


@dataclass(eq=False)
class DecayModel:
    """The full 3HDM-S₃ with gauge + Yukawa sectors and a physical basis."""

    scalar: object                 # the model.py S3Model bundle
    model: Model
    lepton: object                 # fermions.FermionSector
    delta: sp.Symbol
    R: sp.Matrix                   # the geometric rotation R_A
    RS: sp.Matrix                  # R_A · R_H(δ)
    cp_even: tuple                 # (h_1, h_0, h_2)
    cp_odd: tuple                  # (G0, A_1, A_2)
    charged: tuple                 # (Gp, Hp_1, Hp_2)
    gauge: dict                    # {"Z":…, "A":…, "Wp":…, "Wm":…}
    negative: tuple = ()           # (Gm, Hm_1, Hm_2), the antiparticles
    cmap: dict = _field(default_factory=dict)   # {conjugate(H+): H−}
    _cache: dict = _field(default_factory=dict, repr=False)

    @property
    def bosons(self):
        return list(self.cp_even) + list(self.cp_odd) + list(self.charged) \
            + list(self.negative) \
            + [self.gauge["Z"], self.gauge["Wp"], self.gauge["Wm"]]

    def on_vacuum(self, expr):
        """Impose the √3 alignment (the exact-S₃ vacuum)."""
        return sp.expand(expr).subs(self.scalar.align)


def build_decay_model(delta=None, name="3HDM-S3-decays") -> DecayModel:
    """Scalars + electroweak kinetic terms + charged-lepton Yukawas, rotated.

    Args:
        delta: the CP-even Higgs-basis angle.  ``None`` (default) keeps it a
            free symbol ``δ`` — the general Scenario C.  Pass ``0`` for the
            draft's Scenario A or ``sp.pi/2`` for Scenario B.
    """
    m = build_model()
    H1, H2, HS = m.doublets
    SU2L, U1Y, s3 = m.SU2L, m.U1Y, m.s3

    d = sp.Symbol("delta", real=True) if delta is None else sp.sympify(delta)
    R = rotation(m)
    RS = sp.simplify(R * F.R_H(d))

    lep = F.build_lepton_sector(s3, SU2L, U1Y, m.doublets)

    V = -sum(t.expr for t in m.model.lagrangian.terms)
    kinetic = sum((dag(Dmu(H)) * Dmu(H))[0] for H in m.doublets)
    L = (Lagrangian().add(kinetic, sector="kinetic")
         .add(-V, sector="potential")
         .add(lep.lagrangian(), sector="yukawa"))

    # the gauge bosons must be declared fields, or `Model.spin_map` has no
    # spin for W/B and every vertex touching them fails to classify
    W, B = SU2L.bosons(), U1Y.bosons()

    model = Model(name, gauge_groups=[SU2L, U1Y], discrete_groups=[s3],
                  fields=[H1, H2, HS, *lep.left, *lep.right, W, B],
                  parameters=list(m.model.parameters)
                  + list(lep.couplings.values()),
                  lagrangian=L)

    def flucts(part):
        idx = 1 if part == "r" else 2
        return [H.vev_expansions[H.components[1]][idx] for H in m.doublets]

    cp_even = sp.symbols("h_1 h_0 h_2", real=True)
    cp_odd = sp.symbols("G0 A_1 A_2", real=True)
    charged = sp.symbols("Gp Hp_1 Hp_2")

    # new = Rᵀ old, i.e. the columns of R are the physical directions
    model.rotate(Rotation(flucts("r"), list(cp_even), RS.T))
    model.rotate(Rotation(flucts("i"), list(cp_odd), R.T))
    model.rotate(Rotation([H.components[0] for H in m.doublets],
                          list(charged), R.T))
    Z, A = weinberg_rotation(model, SU2L, U1Y)
    Wp, Wm = charged_current_rotation(model, SU2L)

    # The charged scalars are complex, so a vertex like h H⁺H⁻ reaches the
    # extractor as `Hp · conjugate(Hp)`.  Without this map the extractor never
    # forms it and silently returns only the legs linear in H⁺ — which is how
    # the h_i H⁺H⁻ trilinears came back empty the first time.
    negative, cmap = [], {}
    for sym, name in zip(charged, ("Gm", "Hm_1", "Hm_2")):
        neg, part = conjugate_pair(sym, name)
        negative.append(neg)
        cmap.update(part)

    return DecayModel(scalar=m, model=model, lepton=lep, delta=d, R=R, RS=RS,
                      cp_even=cp_even, cp_odd=cp_odd, charged=charged,
                      gauge={"Z": Z, "A": A, "Wp": Wp, "Wm": Wm},
                      negative=tuple(negative), cmap=cmap)


# --------------------------------------------------------------------------
# vertices
# --------------------------------------------------------------------------

def boson_vertices(dm: DecayModel, fields, sectors=("kinetic", "potential"),
                   simplifier=None):
    """Three-leg bosonic vertices, extracted per sector and merged.

    Restricted to the bosonic sectors on purpose — see the module docstring.
    The charged-scalar conjugate map is always passed, so ``h H⁺H⁻``-type
    vertices actually form.  ``simplifier`` defaults to ``None``: simplifying
    every coupling during extraction costs ~5× (390 s vs 69 s for the
    ``h_i H⁺H⁻`` set), and callers only need a handful simplified.

    Cached on the `DecayModel` per (fields, sectors) key.
    """
    key = ("bv", tuple(map(str, fields)), tuple(sectors))
    if key in dm._cache:
        return dm._cache[key]
    out = {}
    for sector in sectors:
        for v in dm.model.vertices(list(fields), sector=sector, min_legs=3,
                                   conjugate_map=dm.cmap,
                                   simplifier=simplifier):
            if len(v.particles) != 3:
                continue
            out.setdefault(v.particles, []).append(v)
    dm._cache[key] = out
    return out


def vv_couplings(dm: DecayModel, vector="Wp"):
    """``{h_i: g(h_i V V)}`` on the aligned vacuum, at general δ.

    The headline check of §2: ``h_0`` must come out **exactly zero** for any δ,
    and the other two must share the SM strength as cos δ / sin δ.
    """
    key = ("vv", vector)
    if key in dm._cache:
        return dm._cache[key]
    pair = {"Wp": (dm.gauge["Wp"], dm.gauge["Wm"]),
            "Z": (dm.gauge["Z"], dm.gauge["Z"])}[vector]
    fields = list(dm.cp_even) + list(set(pair))
    table = boson_vertices(dm, fields, sectors=("kinetic",))
    out = {}
    for h in dm.cp_even:
        got = sp.S.Zero
        for particles, vs in table.items():
            if h in particles and all(p in particles for p in set(pair)):
                got += sum(v.coupling for v in vs)
        out[h] = sp.simplify(dm.on_vacuum(got))
    dm._cache[key] = out
    return out


# --------------------------------------------------------------------------
# the lepton mass basis
# --------------------------------------------------------------------------

#: physical charged-lepton names, in the order `fermions.two_stage_diagonalize`
#: produces (the O₁₂ eigenvector with m_e first, then the 2–3 block).
LEPTON_NAMES = ("e", "mu", "tau")


def register_lepton_basis(dm: DecayModel, O):
    """Rotate the charged leptons to their mass basis.

    ``O`` is the orthogonal matrix of `fermions.two_stage_diagonalize`
    (``O = O₁₂O₂₃``).  **Two rotations per chirality** — field-side and
    bar-side with the same matrix — as `CLAUDE.md` documents and
    `examples/sm_vll.py` demonstrates; registering only one silently leaves
    half the legs in the weak basis.

    Returns ``(fields, bars)``: the physical ``(e, mu, tau)`` leg symbols.
    """
    if "leptons" in dm._cache:
        return dm._cache["leptons"]

    lep = dm.lepton
    bars, rights = lep.mass_legs()
    lefts = [f.components[1][0] for f in lep.left]      # charged SU(2) slot

    # The physical legs must stay `Indexed`, not plain Symbols: the whole
    # bilinear track (`expand_bilinear`, `_split_indexed_term`) matches on
    # `Indexed` nodes, and rotating onto Symbols makes it raise
    # "expected exactly one Indexed factor".  Same shape as sm_vll.py's
    # `Rotation([eL[i], EL[i]], [e1L[i], e2L[i]], …)`.
    idx = lambda suffix: [sp.IndexedBase(f"{n}{suffix}")[0]
                          for n in LEPTON_NAMES]
    phys_L, phys_R = idx("_L"), idx("_R")
    phys_Lbar, phys_Rbar = idx("_Lbar"), idx("_Rbar")

    right_bars = [f.bar_components[0][0] for f in lep.right]

    Ot = sp.Matrix(O).T
    for old, new in ((lefts, phys_L), (bars, phys_Lbar),
                     (rights, phys_R), (right_bars, phys_Rbar)):
        dm.model.rotate(Rotation(list(old), list(new), Ot))

    dm._cache["leptons"] = (phys_L, phys_R, phys_Lbar, phys_Rbar)
    return dm._cache["leptons"]


# --------------------------------------------------------------------------
# benchmark points
# --------------------------------------------------------------------------

def load_points(path="results/viable_points.json"):
    """The viable benchmark points of `01_scalar_parameter_space.ipynb`."""
    return json.loads(Path(path).read_text())["points"]


def point_substitution(dm: DecayModel, point):
    """``{symbol: value}`` for one benchmark point (λ's and VEVs)."""
    lam = dm.scalar.lam_symbols
    sub = {lam[k]: point["lambdas"][f"lambda_{k + 1}"] for k in range(8)}
    sub[dm.scalar.v1.s] = point["vevs_GeV"]["v1"]
    sub[dm.scalar.v2.s] = point["vevs_GeV"]["v2"]
    sub[dm.scalar.vS.s] = point["vevs_GeV"]["vS"]
    return sub


def delta_of_point(dm: DecayModel, point):
    """The **predicted** δ = α − θ_v at a benchmark point.

    α is the residual CP-even 2×2 mixing angle (`model.cp_even_angle`) and
    θ_v the vacuum angle, so δ is fixed by the λ's and the vacuum — the model
    says which of the draft's scenarios it realises, rather than being told.
    """
    alpha_expr = cp_even_angle(dm.scalar)[0]
    sub = point_substitution(dm, point)
    alpha = complex(sp.N(alpha_expr.subs(dm.scalar.align).subs(sub))).real
    v2, vS = point["vevs_GeV"]["v2"], point["vevs_GeV"]["vS"]
    v12 = 2 * v2 / sp.sqrt(3)                    # v12 = 2 v1, v2 = √3 v1
    theta_v = float(sp.N(sp.atan2(v12, vS)))
    return alpha - theta_v
