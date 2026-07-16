"""Assemble the 4-boson (VVVV) self-coupling into the 3 UFO Lorentz
structures — the "Phase 5" step yangmills.py's ``quartic_couplings``
docstring deferred and that was never built for any gauge group.

Derivation (verified by direct functional differentiation of
``-1/4 F^a_{mu nu} F^{a mu nu}`` against ``quartic_couplings``'s raw output,
for both SU(2) and SU(3), including cases where more than one of the three
terms is nonzero — see tests/test_yangmills.py for the ground-truth check):
for a representative external ordering ``(i, j, k, l)`` of 4 physical
bosons,

    VVVV1 = -12 * quartic_couplings(...)[(i, j, k, l)]
    VVVV2 = -12 * quartic_couplings(...)[(i, k, j, l)]
    VVVV3 = -12 * quartic_couplings(...)[(i, l, j, k)]

matching lorentz_map.py's ``VVVV1/2/3`` definitions
(``Metric(1,4)Metric(2,3) - Metric(1,3)Metric(2,4)`` etc). These are raw
Lagrangian-level coefficients (no extra "i"), consistent with how
``cubic_couplings``'s raw output is already used as-is for the VVV1 UFO
coupling elsewhere in this codebase.
"""

__all__ = ["assemble_vvvv"]

_K = -12


def assemble_vvvv(quartic_raw, quadruple):
    """Build ``{lorentz_name: coeff}`` (VVVV1/2/3) for one boson quadruple.

    Args:
        quartic_raw: the raw dict returned by
            :func:`~feynlag.vertices.yangmills.quartic_couplings`.
        quadruple: ``(i, j, k, l)`` — one representative ordering of the 4
            physical boson symbols.

    Returns:
        dict with only the nonzero VVVV1/2/3 entries.
    """
    i, j, k, l = quadruple
    raw = {
        "VVVV1": _K * quartic_raw.get((i, j, k, l), 0),
        "VVVV2": _K * quartic_raw.get((i, k, j, l), 0),
        "VVVV3": _K * quartic_raw.get((i, l, j, k), 0),
    }
    return {name: coeff for name, coeff in raw.items() if coeff != 0}
