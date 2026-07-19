"""Vertex objects and the closed Lorentz-structure catalog.

The vertex catalog is CLOSED: SSS, SSSS, VSS, VVS, VVSS, VVV, VVVV, FFS,
FFV, and FFFF (four-fermion, dim-6 effective operators).  Classification is
rule-based on the spin content of the legs; anything outside the catalog raises
loudly (no silent drops).  Each catalog entry carries the UFO ``lorentz.py``
structure name used at export.

``FFFF`` cannot be reconstructed from the sorted spin letters alone — the four
letters don't say which fermion pairs into which Dirac chain.  A caller that
builds a :class:`Vertex` for a four-fermion operator must record the chain
pairing in ``Vertex.meta['pairing']`` (the two ``(bar, gamma, field)`` subkeys
produced by :func:`~feynlag.vertices.bilinear.extract_fermion_vertices`).
"""

from dataclasses import dataclass, field as dc_field

import sympy as sp

from .extract import vertex_multiplicity

__all__ = ["Vertex", "classify_spins", "LORENTZ_CATALOG"]

#: vertex type → UFO lorentz structure name(s) (Phase 5 fills the strings)
LORENTZ_CATALOG = {
    "SSS": ["SSS1"],
    "SSSS": ["SSSS1"],
    "VSS": ["VSS1"],
    "VVS": ["VVS1"],
    "VVSS": ["VVSS1"],
    "VVV": ["VVV1"],
    "VVVV": ["VVVV1", "VVVV2", "VVVV3"],
    "FFS": ["FFS1", "FFS2"],      # P_L / P_R slots
    "FFV": ["FFV1", "FFV2"],      # γ^μ P_L / γ^μ P_R slots
    # four-fermion: two Dirac chains, each scalar (S) or vector (V), each with a
    # L/R projector → the 2×(2×2) chain-structure pairs (see export/ufo)
    "FFFF": ["FFFFSLL", "FFFFSLR", "FFFFSRL", "FFFFSRR",
             "FFFFVLL", "FFFFVLR", "FFFFVRL", "FFFFVRR"],
}

_SPIN_LETTER = {0: "S", sp.Rational(1, 2): "F", 1: "V"}


def classify_spins(field_tuple, spin_map):
    """Vertex type string from the spins of the legs.

    Args:
        field_tuple: physical field symbols.
        spin_map: ``{symbol: spin}`` with spin in {0, 1/2, 1}.

    Returns:
        catalog key like ``'VVS'`` (letters sorted V < S..., i.e. fermions
        first, vectors next, scalars last — FFV, VSS, VVS conventions).

    Raises:
        KeyError: a leg has no spin assigned.
        ValueError: the spin content is outside the closed catalog.
    """
    letters = []
    for f in field_tuple:
        spin = spin_map[f]
        letters.append(_SPIN_LETTER[sp.Rational(spin)])
    # canonical order: F, V, S
    order = {"F": 0, "V": 1, "S": 2}
    key = "".join(sorted(letters, key=lambda c: order[c]))
    if key not in LORENTZ_CATALOG:
        raise ValueError(f"vertex {field_tuple} with spin content {key!r} "
                         f"is outside the v1 catalog "
                         f"{sorted(LORENTZ_CATALOG)}")
    return key


@dataclass
class Vertex:
    """One Feynman vertex.

    Attributes:
        particles: canonically sorted tuple of physical field symbols.
        coupling: the full rule ``i × coefficient × ∏ multiplicities!``
            (may contain ``p(φ)`` momentum tags for derivative couplings).
        vertex_type: catalog key (``'VVS'``, …).
    """

    particles: tuple
    coupling: sp.Expr
    vertex_type: str = ""
    meta: dict = dc_field(default_factory=dict)

    @classmethod
    def from_coefficient(cls, field_tuple, coefficient, spin_map=None):
        """Build from a monomial coefficient (extractor output)."""
        coupling = sp.I * coefficient * vertex_multiplicity(field_tuple)
        vtype = classify_spins(field_tuple, spin_map) if spin_map else ""
        return cls(tuple(field_tuple), coupling, vtype)

    @property
    def lorentz_structures(self):
        return LORENTZ_CATALOG.get(self.vertex_type, [])

    def __repr__(self):
        names = " ".join(str(p) for p in self.particles)
        return f"Vertex({names} [{self.vertex_type}] : {self.coupling})"

    def _repr_latex_(self):
        fields_tex = "\\, ".join(sp.latex(p) for p in self.particles)
        return f"$\\displaystyle {fields_tex} \\, : \\; {sp.latex(self.coupling)}$"
