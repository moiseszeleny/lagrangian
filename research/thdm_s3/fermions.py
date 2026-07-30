"""S₃ fermion sectors of the 3HDM: charged leptons and quarks.

Companion to `model.py` (which carries the scalar sector).  This module builds
the S₃-invariant Yukawa Lagrangian, extracts the 3×3 mass matrices, and runs the
two-stage diagonalization that `LFVHD_3HDMS3.tex` performs by hand — so every
step of that draft can be checked against an independent derivation.

**Basis.** The draft and feynlag use *different* bases for the S₃ doublet.  The
draft's generators are ``a = R(−2π/3)`` and ``b`` = reflection about 30°;
feynlag's are ``ρ(a) = R(2π/3)``, ``ρ(b) = diag(1,−1)``.  They are related by a
unique O(2) **reflection at π/6** (`tex_basis_map`), which is why the draft's
vacuum alignment reads ``v₁ = √3 v₂`` where feynlag's reads ``v₂ = √3 v₁``.
Four of the five Yukawa structures are built from the *dot product* of two S₃
doublets and are therefore basis-independent; only the triple-doublet structure
(``Y₂``) changes, exactly as ``λ₄`` does in the scalar potential — see
`01_scalar_parameter_space.ipynb` §4.  Under the reflection the draft's ``Y₂``
structure maps onto feynlag's with coefficient **+1**, so the full coupling
dictionary is the identity.

References
----------
[LFVHD] M. Zeleny-Mora, M. Mondragón, T. A. Valencia-Pérez, "Exploring LFV Higgs
    decays in the Three Higgs Doublet Model", draft (`LFVHD_3HDMS3.tex`).
    Eq. (S3_representations_leptons) the S₃ assignment; the Yukawa Lagrangian and
    Eq. (ML_mass_matrix); Eq. (O12_definition)/(O23) the two-stage
    diagonalization; Eq. (RS)/(RAT)/(R_H) the scalar rotations;
    Eq. (Q_general) the LFV couplings.
[DasDeyPal16] D. Das, U. K. Dey, P. B. Pal, "S₃ symmetry and the quark mixing
    matrix", Phys. Lett. B 753, 315 (2016), arXiv:1507.06509.  The quark-sector
    companion: an unbroken Z₂ leaves the CKM matrix block-diagonal, and soft S₃
    breaking is what generates the small elements.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field

import sympy as sp

from feynlag import (Bilinear, ExternalParameter, S3, SU2, SU3, Scalar, U1,
                     WeylFermion, diracPL, diracPR)
from feynlag.vertices.bilinear import expand_bilinear

__all__ = [
    "TEX_GENERATORS", "tex_basis_map", "FermionSector", "build_lepton_sector",
    "build_quark_sector", "mass_matrix_from_bilinears", "o12_angle", "o12",
    "o23", "two_stage_diagonalize", "g_matrices", "q_matrices",
    "R_A", "R_H", "R_S",
]


# --------------------------------------------------------------------------
# the basis map to the draft
# --------------------------------------------------------------------------

#: The draft's S₃ doublet generators — ``a`` of order 3, ``b`` of order 2.
TEX_GENERATORS = (
    sp.Matrix([[sp.Rational(-1, 2), sp.sqrt(3) / 2],
               [-sp.sqrt(3) / 2, sp.Rational(-1, 2)]]),
    sp.Matrix([[sp.Rational(1, 2), sp.sqrt(3) / 2],
               [sp.sqrt(3) / 2, sp.Rational(-1, 2)]]),
)


def tex_basis_map(s3: S3 | None = None):
    """The O(2) matrix conjugating feynlag's S₃ doublet basis into the draft's.

    Returns ``(O, ok)`` where ``O⁻¹ ρ_feynlag(g) O = ρ_draft(g)`` for both
    generators, and ``ok`` records that this was *verified*, not assumed.  The
    solution is the reflection at π/6; ``det O = −1``, which is exactly why the
    draft's order-3 generator is feynlag's inverse.

    Doublet components translate as ``v_draft = Oᵀ v_feynlag``.
    """
    s3 = s3 or S3()
    a_fl, b_fl = s3._irrep_generators["2"]
    a_tex, b_tex = TEX_GENERATORS
    t = sp.pi / 6
    O = sp.Matrix([[sp.cos(t), sp.sin(t)], [sp.sin(t), -sp.cos(t)]])
    ok = (sp.simplify(O.inv() * a_fl * O - a_tex) == sp.zeros(2, 2)
          and sp.simplify(O.inv() * b_fl * O - b_tex) == sp.zeros(2, 2))
    return O, ok


# --------------------------------------------------------------------------
# field content
# --------------------------------------------------------------------------

@dataclass(eq=False)
class FermionSector:
    """One S₃-structured Yukawa sector: (2 ⊕ 1) left, (2 ⊕ 1) right."""

    name: str
    left: tuple                 # (F1, F2, FS)  — the SU(2) doublets
    right: tuple                # (R1, R2, RS)  — the SU(2) singlets
    couplings: dict             # {1..5: ExternalParameter}
    terms: dict                 # {1..5: expression}  (no h.c.)
    tilde: bool                 # built with H̃ (up-type) rather than H
    su2_slot: int               # which SU(2) component carries the mass term
    ncolors: int = 1
    _cache: dict = _field(default_factory=dict, repr=False)

    def lagrangian(self, prefactors=None):
        """``−Σ_k p_k Y_k T_k + h.c.`` — the sign convention of ``L = −ψ̄Mψ``.

        ``prefactors`` overrides the coefficient of each structure, so the
        draft's own normalization (a ``1/√2`` on every term except ``Y₃``) can
        be reproduced verbatim when checking its ``μ ↔ Y`` dictionary.
        Defaults to 1, i.e. ``μ_k = Y_k v/√2`` straight from the VEV expansion.
        """
        p = prefactors or {}
        core = sum(p.get(k, 1) * self.couplings[k].s * self.terms[k]
                   for k in range(1, 6))
        return -(core + sp.conjugate(core))

    #: The draft's prefactors: 1/√2 on every structure except Y₃.
    DRAFT_PREFACTORS = {1: 1 / sp.sqrt(2), 2: 1 / sp.sqrt(2), 3: sp.Integer(1),
                        4: 1 / sp.sqrt(2), 5: 1 / sp.sqrt(2)}

    @property
    def coupling_symbols(self):
        return [self.couplings[k].s for k in range(1, 6)]

    def mass_legs(self):
        """(bar_legs, field_legs) — the 3+3 `Indexed` legs of the mass term."""
        bars = [f.bar_components[self.su2_slot * self.ncolors][0]
                for f in self.left]
        fields = [r.components[0][0] for r in self.right]
        return bars, fields


def _sandwich(Lf, Hf, Rf, tilde=False, ncolors=1, color=None):
    """``(L̄·H) ψ_R`` — SU(2) contracted, colour summed (or one colour picked).

    ``tilde`` uses H̃ = (H⁰*, −H⁺*), the conjugate doublet the up-type quarks
    need; built inline exactly as ``examples/sm_scalar_gauge.py`` does, since no
    ``Htilde`` abstraction exists in the library.
    """
    Hp, H0 = Hf.components
    upper, lower = (sp.conjugate(H0), -sp.conjugate(Hp)) if tilde else (Hp, H0)
    bar = Lf.bar_components
    cols = range(ncolors) if color is None else [color]
    out = 0
    for c in cols:
        # component order is (SU(2) slot outer, colour inner)
        out += (upper * Bilinear(bar[c][0], diracPR, Rf.components[c][0])
                + lower * Bilinear(bar[ncolors + c][0], diracPR,
                                   Rf.components[c][0]))
    return out


def _yukawa_terms(left, right, scalars, tilde=False, ncolors=1):
    """The five S₃-invariant Yukawa structures, in **feynlag's** doublet basis.

    With ``L = (L₁,L₂)``, ``R = (R₁,R₂)`` S₃ doublets and ``L_S``, ``R_S``,
    ``H_S`` singlets, and writing ``[X]_k`` for the k-th CG channel of 2⊗2
    (`S3.doublet_product`):

    ==  ============================================  =====================
    k   structure                                     CG channel
    ==  ============================================  =====================
    1   ``(L̄ · R)₁ H_S``                              1 of L̄⊗R
    2   ``[L̄ ⊗ H]₂ · R``                              the unique 1 of 2⊗2⊗2
    3   ``L̄_S H_S R_S``                               all singlets
    4   ``L̄_S (H · R)₁``                              1 of H⊗R
    5   ``(L̄ · H)₁ R_S``                              1 of L̄⊗H
    ==  ============================================  =====================

    Structures 1, 3, 4, 5 are dot products of two doublets and are invariant
    under *any* orthogonal change of doublet basis; only structure 2 is
    basis-sensitive (see the module docstring).
    """
    L1, L2, LS = left
    R1, R2, RS = right
    H1, H2, HS = scalars
    S = lambda Lf, Hf, Rf: _sandwich(Lf, Hf, Rf, tilde, ncolors)

    # [L̄ ⊗ H]₂ = (x₁₁ − x₂₂, −(x₁₂ + x₂₁)) in feynlag's real orthogonal basis
    return {
        1: S(L1, HS, R1) + S(L2, HS, R2),
        2: ((S(L1, H1, R1) - S(L2, H2, R1))
            - (S(L1, H2, R2) + S(L2, H1, R2))),
        3: S(LS, HS, RS),
        4: S(LS, H1, R1) + S(LS, H2, R2),
        5: S(L1, H1, RS) + S(L2, H2, RS),
    }


def _declare(prefix, s3, reps_L, reps_R, comp_L, comp_R, ncolors):
    """Three left doublets and three right singlets, assigned to 2 ⊕ 1.

    Component names span the full gauge Kronecker product, SU(2) slot outer and
    colour inner — the ordering `_sandwich` assumes.
    """
    def names(stems, tag):
        if ncolors == 1:
            return [f"{c}{tag}" for c in stems]
        return [f"{c}{tag}_{k + 1}" for c in stems for k in range(ncolors)]

    def mk(name, reps, chirality, comp_names):
        return WeylFermion(name, reps=reps, chirality=chirality, nflavors=1,
                           component_names=comp_names)

    left, right = [], []
    for tag in ("1", "2", "S"):
        nm = f"{prefix}{tag}"
        left.append(mk(nm, reps_L, "L", names(comp_L, nm)))
        rn = f"{prefix}{tag}R"
        right.append(mk(rn, reps_R, "R", names(comp_R, rn)))
    s3.assign("2", left[0], left[1]); s3.assign("1", left[2])
    s3.assign("2", right[0], right[1]); s3.assign("1", right[2])
    return tuple(left), tuple(right)


def build_lepton_sector(s3, SU2L, U1Y, scalars, tex=False):
    """L_i = (ν_i, ℓ_i) doublets + ℓ_iR singlets, S₃ 2 ⊕ 1 — [LFVHD] Eq. (2)."""
    left, right = _declare(
        "L", s3,
        {SU2L: 2, U1Y: -sp.Rational(1, 2)}, {U1Y: -1},
        ["nu", "e"], ["e"], 1)
    Y = {k: ExternalParameter(f"Yl{k}", 0.01 * k, real=True) for k in range(1, 6)}
    terms = _yukawa_terms(left, right, scalars, tilde=False, ncolors=1)
    return FermionSector("lepton", left, right, Y, terms,
                         tilde=False, su2_slot=1, ncolors=1)


def build_quark_sector(s3, SU2L, U1Y, SU3c, scalars, kind):
    """Q_i doublets + u_iR / d_iR singlets, S₃ 2 ⊕ 1 — the analogue of [LFVHD].

    ``kind='down'`` uses H (mass from the lower SU(2) slot); ``kind='up'`` uses
    H̃ (mass from the upper slot).  The left-handed doublets are declared once
    per call, so build the down sector first and pass its ``left`` back in via
    `share_left` if a single Q_L is wanted for both.
    """
    if kind not in ("up", "down"):
        raise ValueError("kind must be 'up' or 'down'")
    hyper = sp.Rational(2, 3) if kind == "up" else -sp.Rational(1, 3)
    tag = "u" if kind == "up" else "d"
    left, right = _declare(
        f"Q{tag}", s3,
        {SU2L: 2, U1Y: sp.Rational(1, 6), SU3c: 3}, {U1Y: hyper, SU3c: 3},
        ["uL", "dL"], [tag], 3)
    Y = {k: ExternalParameter(f"Y{tag}{k}", 0.01 * k, real=True)
         for k in range(1, 6)}
    terms = _yukawa_terms(left, right, scalars, tilde=(kind == "up"), ncolors=3)
    return FermionSector(f"quark-{kind}", left, right, Y, terms,
                         tilde=(kind == "up"),
                         su2_slot=0 if kind == "up" else 1, ncolors=3)


# --------------------------------------------------------------------------
# mass matrices
# --------------------------------------------------------------------------

def mass_matrix_from_bilinears(L_yuk, bar_legs, field_legs, vacuum,
                               gamma=diracPR):
    """3×3 mass matrix by reading off Bilinear coefficients at the vacuum.

    ``vertices/bilinear.py::fermion_mass_matrix`` assumes a **single**
    flavour-indexed ``IndexedBase`` pair, which this model does not have: its
    three generations sit in *different* S₃ irreps and so are separate
    ``WeylFermion``s.  This is the same extraction (Lagrangian mass term
    ``−ψ̄ M χ``, hence ``M[i,j] = −coefficient``) over a list of distinct legs.
    """
    L0 = sp.expand(expand_bilinear(vacuum.at_vacuum(sp.expand(L_yuk))))
    n = len(bar_legs)
    M = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = -sp.expand(L0.coeff(Bilinear(bar_legs[i], gamma,
                                                   field_legs[j])))
    return M


# --------------------------------------------------------------------------
# the two-stage diagonalization  ([LFVHD] Eqs. O12_definition, O23)
# --------------------------------------------------------------------------

def o12_angle(r):
    """ψ with ``tan 2ψ = −r``, the angle that block-diagonalizes the 1–2 sector.

    The upper-left block is ``B = μ₁·1 + a₂[[1, r],[r, −1]]`` with
    ``r = v₁/v₂``, so the diagonalizing angle depends on **the vacuum alone** —
    not on the Yukawa couplings.  That is what makes ``O₁₂`` universal across
    the charged-lepton, up- and down-quark sectors, and hence what forces the
    CKM matrix to be block diagonal in the exactly-symmetric model.

    At ``r = √3`` this gives ψ = 60°, reproducing [LFVHD] Eq. (O12_definition).
    """
    return (sp.pi - sp.atan(r)) / 2


def o12(r):
    """[LFVHD] Eq. (O12_definition), generalized to arbitrary ``r = v₁/v₂``."""
    psi = o12_angle(r)
    c, s = sp.cos(psi), sp.sin(psi)
    return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def o23(theta):
    """[LFVHD] Eq. (O23) — the residual 2–3 rotation."""
    c, s = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[1, 0, 0], [0, c, s], [0, -s, c]])


def two_stage_diagonalize(M, r=sp.sqrt(3), simplify=True):
    """``(O, D, theta)`` with ``O = O₁₂O₂₃`` and ``D = OᵀMO`` diagonal.

    Only valid when the residual 2×2 block is symmetric — which the draft shows
    is forced by matching the singular values to the physical masses
    (``μ₅ = μ₄``).  Raises if it is not, rather than silently returning a
    non-diagonal ``D``.
    """
    O1 = o12(r)
    B = O1.T * M * O1
    if simplify:
        B = B.applyfunc(sp.simplify)
    a, b, c, d = B[1, 1], B[1, 2], B[2, 1], B[2, 2]
    if sp.simplify(b - c) != 0:
        raise ValueError("residual 2x2 block is not symmetric (mu4 != mu5); "
                         "a single orthogonal rotation cannot diagonalize it")
    theta = sp.atan2(2 * b, d - a) / 2 if b != 0 else sp.Integer(0)
    O2 = o23(theta)
    O = O1 * O2
    D = O.T * M * O
    if simplify:
        D = D.applyfunc(sp.simplify)
    return O, D, theta


# --------------------------------------------------------------------------
# scalar rotations and the LFV couplings  ([LFVHD] Eqs. RS, RAT, R_H, Q_general)
# --------------------------------------------------------------------------

def R_A(phi, theta_v):
    """[LFVHD] Eq. (RAT), transposed — the geometric (Higgs-basis) rotation.

    Identical in form to `model.rotation` / [GomezBock21] Eq. (29); its first
    column is the vacuum direction.
    """
    cp, sp_ = sp.cos(phi), sp.sin(phi)
    ct, st = sp.cos(theta_v), sp.sin(theta_v)
    return sp.Matrix([[st * cp, -sp_, -ct * cp],
                      [st * sp_, cp, -ct * sp_],
                      [ct, 0, st]])


def R_H(delta):
    """[LFVHD] Eq. (R_H) — the Higgs-basis → mass-basis rotation, ``δ = α − θ_v``."""
    c, s = sp.cos(delta), sp.sin(delta)
    return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R_S(phi, alpha):
    """[LFVHD] Eq. (RS) — equals ``R_A(φ,θ_v) R_H(α−θ_v)`` identically."""
    return R_A(phi, alpha)


def g_matrices(M, vevs):
    """``(G₁, G₂, G_S)`` with ``M = Σ_k v_k G_k`` — the per-doublet Yukawa matrices.

    Each ``G_k`` is the coefficient matrix of the *real* neutral field φ_{6+k},
    because the ``1/√2`` of the VEV expansion and that of the field expansion
    cancel between ``M = Σ v_k G_k`` and ``L = −ℓ̄ G_k φ_{6+k} ℓ``.
    """
    G = [M.applyfunc(lambda e, vk=vk: sp.diff(sp.expand(e), vk)) for vk in vevs]
    rebuilt = sp.zeros(*M.shape)
    for vk, Gk in zip(vevs, G):
        rebuilt += vk * Gk
    residual = (M - rebuilt).applyfunc(lambda e: sp.simplify(sp.expand(e)))
    if residual != sp.zeros(*M.shape):
        raise ValueError("M is not linear in the VEVs; G_k decomposition failed")
    return tuple(G)


def q_matrices(G, RS, O):
    """[LFVHD] Eq. (Q_general): ``Q_i = Σ_j (R_S)_{ji} Õ_j`` with ``Õ_j = OᵀG_jO``.

    ``Q₁`` in the SM-like scenario must come out ``diag(m)/v`` exactly — the
    first column of ``R_S`` is the vacuum direction, so
    ``Q₁ = (1/v) Oᵀ(Σ v_j G_j) O = (1/v) Oᵀ M O``.
    """
    Gt = [O.T * Gj * O for Gj in G]
    out = []
    for i in range(3):
        Qi = sp.zeros(*Gt[0].shape)
        for j in range(3):
            Qi += RS[j, i] * Gt[j]
        out.append(Qi)
    return tuple(out)
