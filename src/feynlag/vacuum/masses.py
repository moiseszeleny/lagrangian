"""Mass-matrix extraction.

- :func:`build_mass_matrix`: double ``derive_by_array`` kernel (DLRSM1 port).
- :func:`scalar_mass_matrix`: real-fluctuation mass matrix
  ``M²_ij = ∂²V/∂φᵢ∂φⱼ |_vacuum`` with tadpole substitutions.
- :func:`charged_mass_matrix`: complex-field mass matrix
  ``M²_ij = ∂²V/∂φ̄ᵢ∂φⱼ |_vacuum`` via the Dummy-conjugate trick
  (SymPy cannot differentiate with respect to ``conjugate(φ)`` directly).
"""

import sympy as sp
from sympy import derive_by_array

from ..operators import PartialMu

__all__ = ["build_mass_matrix", "scalar_mass_matrix", "charged_mass_matrix",
           "gauge_mass_matrix"]


def build_mass_matrix(potential, fields1, fields2=None):
    """``M_ij = ∂²(potential)/∂fields1_i ∂fields2_j`` as a Matrix."""
    if fields2 is None:
        fields2 = fields1
    elements = derive_by_array(derive_by_array(potential, list(fields1)),
                               list(fields2))
    return elements.tomatrix()


def _at_vacuum_matrix(M, vacuum, tadpole_subs=None):
    """Evaluate a matrix at the vacuum and apply tadpole substitutions."""
    M = M.applyfunc(vacuum.at_vacuum)
    if tadpole_subs:
        M = M.applyfunc(lambda e: sp.expand(e.subs(tadpole_subs)))
    return M.applyfunc(lambda e: sp.factor(sp.expand(e))
                       if e != 0 else sp.S.Zero)


def scalar_mass_matrix(potential, vacuum, fields, tadpole_subs=None):
    """Mass matrix of real fluctuation fields.

    Args:
        potential: scalar potential ``V`` in weak-basis components.
        vacuum: :class:`Vacuum`.
        fields: real fluctuation symbols (a CP-even / CP-odd block).
        tadpole_subs: optional dict from :func:`solve_tadpoles`.

    Returns:
        Symmetric ``M²`` Matrix.
    """
    V_shifted = vacuum.shift(potential)
    M = build_mass_matrix(V_shifted, fields)
    return _at_vacuum_matrix(M, vacuum, tadpole_subs)


def charged_mass_matrix(potential, vacuum, fields, tadpole_subs=None):
    """Mass matrix of complex (charged) fields: ``∂²V/∂φ̄ᵢ∂φⱼ``.

    Implements the DLRSM1 Dummy-conjugate trick: ``conjugate(φᵢ)`` is
    replaced by a Dummy, the derivative is taken with respect to the Dummies
    (rows) and the fields (columns), and the Dummies are restored before
    vacuum evaluation.

    Returns:
        Hermitian ``M²`` Matrix (rows: ∂/∂φ̄, columns: ∂/∂φ).
    """
    dummies = {sp.conjugate(f): sp.Dummy(f"{f.name}_conj") for f in fields}
    V_dummied = potential.xreplace(dummies)
    M = build_mass_matrix(V_dummied, list(dummies.values()), list(fields))
    restore = {d: c for c, d in dummies.items()}
    M = M.applyfunc(lambda e: e.xreplace(restore))
    return _at_vacuum_matrix(M, vacuum, tadpole_subs)


def gauge_mass_matrix(kinetic, vacuum, gauge_components, tadpole_subs=None):
    """Gauge boson mass matrix from the scalar kinetic sector.

    Evaluates ``L_kin = (D_μ φ)†(D^μ φ)`` on the vacuum (where every scalar
    is constant, so all ``∂_μ`` terms drop) and takes
    ``M²_ab = ∂² L_kin,vac / ∂A^a ∂A^b`` — the vector mass term in the
    Lagrangian is ``+½ M²_ab A^a A^b``, no sign flip.

    Args:
        kinetic: the scalar kinetic Lagrangian (with ``PartialMu`` heads).
        vacuum: :class:`Vacuum`.
        gauge_components: gauge boson component symbols (e.g. W1, W2, W3, B).

    Returns:
        Symmetric ``M²`` Matrix over ``gauge_components``.
    """
    at_vac = vacuum.at_vacuum(kinetic)
    # every scalar is constant on the vacuum: ∂_μ(anything) → 0 here
    at_vac = sp.expand(at_vac.replace(PartialMu, lambda arg: 0))
    M = build_mass_matrix(at_vac, list(gauge_components))
    if tadpole_subs:
        M = M.applyfunc(lambda e: sp.expand(e.subs(tadpole_subs)))
    return M.applyfunc(lambda e: sp.factor(sp.expand(e))
                       if e != 0 else sp.S.Zero)
