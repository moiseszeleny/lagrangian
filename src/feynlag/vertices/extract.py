"""Interaction-coefficient extraction from a Lagrangian.

Ported from ``bsm-calc/models/DLRSM1/symbolic_tools.py``.  The extractor
treats fields as commuting SymPy Symbols — fermions never enter here; they are
handled by the bilinear track (Phase 4).

The result is a nested dict::

    {n_fields: {(field, field, ...): coefficient}}

where the inner key is the canonically sorted tuple of fields in the monomial
and the coefficient is the *monomial* coefficient (symmetry factorials NOT yet
multiplied out — see :func:`feynman_rule`).
"""

from collections import Counter

from sympy import Add, I, Integer, Mul, S, Symbol, cancel, expand, factorial

__all__ = ["extract_interaction_coefficients", "vertex_multiplicity",
           "feynman_rule"]


def _extract_fallback(L, fields_set):
    """Term-by-term fallback used when the fast Poly path fails."""
    L_expanded = expand(L)
    interaction_dict = {}

    for term in L_expanded.as_ordered_terms():
        if term == S.Zero:
            continue

        detected_fields_for_term = []
        for f_symbol in fields_set:
            if f_symbol not in term.free_symbols:
                continue

            exponent_of_f = 0
            if term.is_Mul:
                for factor in term.args:
                    if factor == f_symbol:
                        exponent_of_f += 1
                    elif factor.is_Pow and factor.base == f_symbol:
                        if isinstance(factor.exp, (int, Integer)):
                            exponent_of_f += int(factor.exp)
            elif term.is_Pow:
                if term.base == f_symbol and isinstance(term.exp, (int, Integer)):
                    exponent_of_f = int(term.exp)
            elif term == f_symbol:
                exponent_of_f = 1

            if exponent_of_f > 0:
                detected_fields_for_term.extend([f_symbol] * exponent_of_f)

        term_fields_tuple = tuple(
            sorted(detected_fields_for_term, key=lambda s: s.sort_key()))
        num_fields = len(term_fields_tuple)

        field_product = Mul(*term_fields_tuple) if term_fields_tuple else S.One
        coefficient = cancel(term / field_product)

        bucket = interaction_dict.setdefault(num_fields, {})
        bucket[term_fields_tuple] = bucket.get(term_fields_tuple, S.Zero) + coefficient

    return interaction_dict


def extract_interaction_coefficients(L, fields, parameters=None):
    """Extract interaction coefficients from a Lagrangian.

    Tries a fast ``Poly``-based method (``domain='EX'``) first, falling back
    to a robust term-by-term method.

    Args:
        L: SymPy expression (the Lagrangian, or one sector of it).
        fields: iterable of SymPy ``Symbol`` field components.  Non-Symbol
            entries are ignored (fermion bilinears take the other track).
        parameters: unused; kept for signature compatibility with DLRSM1.

    Returns:
        dict: ``{n_fields: {sorted-field-tuple: coefficient}}``.
    """
    fields = {f for f in fields if isinstance(f, Symbol)}

    if not fields:
        return {0: {(): L}} if L != S.Zero else {}

    fields_tuple = tuple(sorted(fields, key=lambda s: s.sort_key()))

    poly_L = None
    try:
        poly_L = L.as_poly(*fields_tuple, domain="EX")
    except Exception:
        poly_L = None

    if poly_L is None:
        return _extract_fallback(L, fields)

    interaction_dict = {}
    for monom_exps, coeff in poly_L.terms():
        if coeff == S.Zero:
            continue

        detected_fields_list = []
        for field_symbol, exponent in zip(fields_tuple, monom_exps):
            detected_fields_list.extend([field_symbol] * exponent)

        term_fields_tuple = tuple(
            sorted(detected_fields_list, key=lambda s: s.sort_key()))
        num_fields = len(term_fields_tuple)

        bucket = interaction_dict.setdefault(num_fields, {})
        bucket[term_fields_tuple] = bucket.get(term_fields_tuple, S.Zero) + coeff

    return interaction_dict


def vertex_multiplicity(field_tuple):
    """Product of ``k!`` over the multiplicities of identical fields.

    This is the combinatorial factor relating a monomial coefficient to the
    Feynman vertex: for ``L ⊃ c · φ₁^{k₁} … φₙ^{kₙ}`` the vertex is
    ``i · c · ∏ kᵢ!``.
    """
    result = S.One
    for count in Counter(field_tuple).values():
        result *= factorial(count)
    return result


def feynman_rule(coefficient, field_tuple):
    """Feynman rule for one monomial: ``i × coefficient × ∏ (multiplicity)!``.

    Pinned convention (CONVENTIONS.md): ``L = −λ/4! φ⁴`` gives vertex ``−iλ``.
    """
    return I * coefficient * vertex_multiplicity(field_tuple)
