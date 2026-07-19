"""Quark flavour mixing — the CKM matrix as input, not as a symbolic SVD.

A fully generic three-generation complex Yukawa does not diagonalize in closed
symbolic form (nested radicals from ``Matrix.diagonalize``), so feynlag takes
the pragmatic FeynRules-SM route: work in the quark **mass basis** with diagonal
Yukawas and insert the CKM matrix directly into the charged (W) current. The
neutral (Z, γ, gluon) currents stay flavour-diagonal by construction — the GIM
mechanism — so no spurious flavour-changing neutral currents appear.

The matrix must be **unitary**; a matrix of nine independent symbols is not
(``Σ_k V*_{ki} V_{kj} ≠ δ_{ij}`` symbolically, which would fake Z-FCNCs). This
module therefore builds ``V`` from the exact PDG standard parametrization in
three mixing angles and one CP phase — unitary as a trigonometric identity — so
that ``V†V = 1`` collapses to ``1`` under :func:`sympy.simplify`.

CKM elements are exported as **scalar** parameters (the angles are real
externals; the nine ``V_{ij}`` are complex internals). They can never be an
``IndexedBase``/matrix parameter: ``ParameterSet`` is scalar-only and the UFO
code printer would emit an invalid ``V[0,1]`` subscript.
"""

import math

import sympy as sp

from .parameters import ExternalParameter, InternalParameter
from .vacuum.diagonalize import rotation_2x2

__all__ = ["standard_ckm", "cabibbo_2x2", "CKM_ELEMENT_NAMES"]

#: row/column order (up-type rows u,c,t; down-type columns d,s,b)
CKM_ELEMENT_NAMES = (
    ("Vud", "Vus", "Vub"),
    ("Vcd", "Vcs", "Vcb"),
    ("Vtd", "Vts", "Vtb"),
)

# PDG central values of the sines of the mixing angles and the CP phase.
_S12, _S13, _S23, _DELTA = 0.22500, 0.003675, 0.04182, 1.144


def standard_ckm(prefix=""):
    """CKM matrix in the exact PDG standard parametrization.

    Builds four real external mixing parameters (``th12``, ``th13``, ``th23``,
    ``deltaCP``) and nine complex internal elements ``V_{ij}`` defined by the
    standard parametrization, so the returned matrix is unitary as a
    trigonometric identity (``V†V`` simplifies to the identity).

    Args:
        prefix: optional string prepended to every parameter name (to run more
            than one CKM-like matrix in one model).

    Returns:
        ``(params, V)``: ``params`` is the list of all 13 parameters (4
        externals + 9 internals, the order UFO export wants — externals first,
        then internals in dependency order), and ``V`` is the ``3×3``
        :class:`sympy.Matrix` of the internal element symbols.
    """
    th12 = ExternalParameter(prefix + "th12", value=math.asin(_S12))
    th13 = ExternalParameter(prefix + "th13", value=math.asin(_S13))
    th23 = ExternalParameter(prefix + "th23", value=math.asin(_S23))
    delta = ExternalParameter(prefix + "deltaCP", value=_DELTA)

    c12, s12 = sp.cos(th12.s), sp.sin(th12.s)
    c13, s13 = sp.cos(th13.s), sp.sin(th13.s)
    c23, s23 = sp.cos(th23.s), sp.sin(th23.s)
    ph = sp.exp(sp.I * delta.s)

    exprs = sp.Matrix([
        [c12 * c13, s12 * c13, s13 / ph],
        [-s12 * c23 - c12 * s23 * s13 * ph,
         c12 * c23 - s12 * s23 * s13 * ph,
         s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * ph,
         -c12 * s23 - s12 * c23 * s13 * ph,
         c23 * c13],
    ])

    internals = []
    V = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            p = InternalParameter(prefix + CKM_ELEMENT_NAMES[i][j], real=False)
            p.define(exprs[i, j])
            internals.append(p)
            V[i, j] = p.s

    return [th12, th13, th23, delta] + internals, V


def cabibbo_2x2(theta):
    """The real 2×2 Cabibbo rotation ``[[cosθ, sinθ], [−sinθ, cosθ]]``.

    A thin alias of :func:`~feynlag.vacuum.diagonalize.rotation_2x2` for the
    two-generation demonstration, where CKM mixing *is* derivable through the
    ordinary real mass-basis rotation machinery (unlike the three-generation
    complex case).
    """
    return rotation_2x2(theta)
