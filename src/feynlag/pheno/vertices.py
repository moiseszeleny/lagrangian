"""Unified decay-vertex view over feynlag's two extraction tracks.

feynlag extracts bosonic and fermionic vertices through deliberately separate
machinery (see the two-track design in ``CLAUDE.md``):

- bosons are commuting symbols → :meth:`~feynlag.lagrangian.Model.vertices`
  returns typed :class:`~feynlag.vertices.vertex.Vertex` objects;
- fermions live inside opaque :class:`~feynlag.vertices.bilinear.Bilinear`
  atoms → :func:`~feynlag.vertices.bilinear.extract_fermion_vertices` returns a
  raw ``{(bar, Γ, field): {n: {bosons: coeff}}}`` dict, and every caller so far
  (``scripts/export_sm_ufo.py``, ``examples/sm_u1x.py``, ``examples/sm_vll.py``)
  has hand-picked keys and hand-split ``P_L``/``P_R``.

A decay calculator needs both, in one list, with the chiral split already done.
:class:`DecayVertex` is that common view and :func:`collect_decay_vertices`
builds it.  Doing the ``P_L``/``P_R`` split **once** here is the point.
"""

from dataclasses import dataclass, field as dc_field

import sympy as sp

from ..dirac import DiracGamma, DiracGammaLower, DiracIdentity, PL, PR
from ..vertices.bilinear import extract_fermion_vertices
from ..vertices.extract import vertex_multiplicity

__all__ = ["DecayVertex", "classify_gamma", "collect_decay_vertices",
           "fermion_decay_vertices", "resolve_leg"]


@dataclass
class DecayVertex:
    """One three-leg vertex, usable by :mod:`~feynlag.pheno.amplitudes`.

    Attributes:
        particles: the leg symbols. Bosonic vertices carry the field symbols;
            fermionic ones carry ``(bar_leg, field_leg, boson)``.
        vertex_type: closed-catalog key (``'FFV'``, ``'VVS'``, …).
        coupling: full Feynman rule for a bosonic vertex (carries the ``i``).
        g_left / g_right: coefficients of ``P_L`` / ``P_R`` (or ``γ^μP_L`` /
            ``γ^μP_R``) for a fermionic vertex, each carrying the Feynman-rule
            ``i`` and the boson multiplicity factor.
        meta: extra bookkeeping (``'bar'``, ``'field'``, ``'boson'``, …).
    """

    particles: tuple
    vertex_type: str
    coupling: sp.Expr = sp.S.Zero
    g_left: sp.Expr = sp.S.Zero
    g_right: sp.Expr = sp.S.Zero
    meta: dict = dc_field(default_factory=dict)

    def __repr__(self):
        legs = " ".join(str(p) for p in self.particles)
        if self.vertex_type.startswith("FF"):
            return (f"DecayVertex({legs} [{self.vertex_type}] : "
                    f"L={self.g_left}, R={self.g_right})")
        return f"DecayVertex({legs} [{self.vertex_type}] : {self.coupling})"


def classify_gamma(gamma):
    """Split a bilinear ``Γ`` slot into ``(structure, chirality)``.

    Returns ``structure`` in ``{'S', 'V'}`` (scalar sandwich vs. vector
    current, i.e. whether a ``γ^μ`` is present) and ``chirality`` in
    ``{'L', 'R', None}``.

    Raises:
        NotImplementedError: for structures outside the set feynlag builds —
            the same policy as :func:`~feynlag.dirac.dirac_conjugate`.
    """
    structure, chirality = "S", None
    for factor in sp.Mul.make_args(sp.sympify(gamma)):
        if isinstance(factor, DiracIdentity) or factor == sp.S.One:
            continue
        if isinstance(factor, PL):
            chirality = "L"
        elif isinstance(factor, PR):
            chirality = "R"
        elif isinstance(factor, (DiracGamma, DiracGammaLower)):
            if structure == "V":
                raise NotImplementedError(
                    f"classify_gamma: {gamma!r} has more than one gamma "
                    f"matrix; only a single vector current is supported")
            structure = "V"
        else:
            raise NotImplementedError(
                f"classify_gamma: no rule for the factor {factor!r} in "
                f"{gamma!r}")
    return structure, chirality


def resolve_leg(leg, particle_map):
    """Map a Weyl leg to its physical particle, falling back to the leg.

    A Dirac fermion is modelled in feynlag as two :class:`WeylFermion`s (see
    ``CLAUDE.md``: ``DiracFermion`` raises on construction), so ``e_L`` and
    ``e_R`` are *different* legs even though ``Z → e⁺e⁻`` is one channel fed by
    both. ``particle_map`` is how the caller states that identification:
    ``{eL[i]: e, eR[i]: e, eLbar[i]: ebar, ...}``. Without it the left- and
    right-handed currents would be counted as two separate decays, each missing
    the other's interference with the mass terms.

    Lookup falls back to the leg's ``IndexedBase`` so a flavour-indexed leg can
    be mapped once for all generations.
    """
    if not particle_map:
        return leg
    if leg in particle_map:
        return particle_map[leg]
    base = getattr(leg, "base", None)
    if base is not None and base in particle_map:
        return particle_map[base]
    return leg


def fermion_decay_vertices(table, spin_map=None, particle_map=None):
    """Turn an :func:`extract_fermion_vertices` table into
    :class:`DecayVertex` objects with the chiral split applied.

    Keys sharing the same ``(bar, field, boson)`` but differing in chirality
    are merged into a single vertex carrying both ``g_left`` and ``g_right``.
    With ``particle_map`` supplied the merge happens on **physical** particles,
    so a Dirac fermion's ``L`` and ``R`` currents combine — see
    :func:`resolve_leg`.

    Args:
        table: the ``{(bar, Γ, field): {n: {bosons: coeff}}}`` dict.
        spin_map: optional ``{symbol: spin}``; when absent the structure of
            ``Γ`` decides the type (``γ^μ`` → FFV).
        particle_map: ``{weyl_leg: physical_particle}``.

    Returns:
        list of :class:`DecayVertex`.
    """
    merged = {}
    for key, by_count in table.items():
        if isinstance(key[0], tuple):
            continue                    # four-fermion (FFFF): not a 1→2 decay
        bar, gamma, field = key
        structure, chirality = classify_gamma(gamma)
        if chirality is None:
            raise NotImplementedError(
                f"fermion_decay_vertices: the structure {gamma!r} has no "
                f"chiral projector; feynlag's FFS/FFV vertices always carry "
                f"P_L or P_R, so this is unexpected input")
        for n_bosons, terms in by_count.items():
            if n_bosons != 1:
                continue                # only three-leg vertices decay 1→2
            for bosons, coeff in terms.items():
                if coeff == 0:
                    continue
                vtype = "FFV" if structure == "V" else "FFS"
                legs = (resolve_leg(bar, particle_map),
                        resolve_leg(field, particle_map))
                slot = merged.setdefault(
                    (legs[0], legs[1], bosons[0], vtype),
                    {"L": sp.S.Zero, "R": sp.S.Zero})
                # the Feynman-rule i and the boson multiplicity, exactly as
                # fermion_feynman_rule applies them (the Dirac structure is
                # carried separately, in g_left/g_right)
                slot[chirality] += sp.I * coeff * vertex_multiplicity(bosons)

    out = []
    for (bar, field, boson, vtype), couplings in merged.items():
        out.append(DecayVertex(
            particles=(bar, field, boson),
            vertex_type=vtype,
            g_left=couplings["L"],
            g_right=couplings["R"],
            meta={"bar": bar, "field": field, "boson": boson},
        ))
    return out


def collect_decay_vertices(model, boson_fields, fermion_sectors=(),
                           conjugate_map=None, simplifier=None,
                           particle_map=None):
    """All three-leg vertices of ``model``, bosonic and fermionic, in one list.

    Args:
        model: a built :class:`~feynlag.lagrangian.Model`.
        boson_fields: physical boson symbols to extract over.
        fermion_sectors: Lagrangian sectors holding fermion bilinears
            (typically ``('gauge', 'yukawa')``); each is run through
            :func:`~feynlag.vertices.bilinear.extract_fermion_vertices`.
        conjugate_map: ``{conjugate(φ): φ̄}`` normalization passed through to
            the bosonic extractor.
        simplifier: optional callable applied to bosonic couplings.
        particle_map: ``{weyl_leg: physical_particle}`` identifying the two
            Weyl legs of a Dirac fermion — see :func:`resolve_leg`.

    Returns:
        list of :class:`DecayVertex`.
    """
    out = []
    spins = model.spin_map(conjugate_map=conjugate_map)

    for vertex in model.vertices(boson_fields, conjugate_map=conjugate_map,
                                 min_legs=3, simplifier=simplifier):
        if len(vertex.particles) != 3:
            continue
        out.append(DecayVertex(
            particles=vertex.particles,
            vertex_type=vertex.vertex_type,
            coupling=vertex.coupling,
            meta={"source": "bosonic"},
        ))

    for sector in fermion_sectors:
        L = sp.expand(model.physical_lagrangian(sector=sector))
        if L == 0:
            continue
        table = extract_fermion_vertices(L, list(boson_fields))
        for vertex in fermion_decay_vertices(table, spin_map=spins,
                                             particle_map=particle_map):
            vertex.meta["source"] = f"fermionic:{sector}"
            out.append(vertex)

    return out
