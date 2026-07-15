"""LaTeX vertex-table generation.

Unifies the three ``generate_latex_table_*`` variants from
``bsm-calc/models/DLRSM1/symbolic_tools.py`` into one configurable function.
"""

from sympy import latex, simplify

__all__ = ["latex_feynman_table"]


def _rows_from_interactions(interactions):
    """Yield ``(field_tuple, coefficient)`` from either dict layout.

    Accepts the nested ``{n_fields: {fields: coeff}}`` output of
    ``extract_interaction_coefficients`` or a flat ``{fields: coeff}`` dict.
    """
    for key in sorted(interactions.keys(), key=str):
        value = interactions[key]
        if isinstance(value, dict):
            for fields, coeff in value.items():
                yield fields, coeff
        else:
            yield key, value


def latex_feynman_table(interactions, simplify_coeff=None, extra_column=None,
                        extra_header=r"\textbf{Simplified}"):
    """Generate a LaTeX table of interactions and their coefficients.

    Args:
        interactions: nested dict from ``extract_interaction_coefficients``
            or a flat ``{field_tuple: coefficient}`` dict.
        simplify_coeff: optional callable applied to each coefficient before
            printing (default: print raw).
        extra_column: optional callable producing a third column from each
            coefficient (e.g. ``sympy.simplify`` or a series approximation).
            ``None`` gives a two-column table.
        extra_header: header of the third column.

    Returns:
        str: LaTeX ``array`` environment.
    """
    ncols = 3 if extra_column is not None else 2
    header_cells = [r"\textbf{Interaction}", r"\textbf{Coefficient}"]
    if extra_column is not None:
        header_cells.append(extra_header)

    table = r"\begin{array}{|" + "c|" * ncols + "}\n"
    table += "\\hline\n"
    table += " & ".join(header_cells) + r" \\" + "\n"
    table += "\\hline\n"

    for fields, coefficient in _rows_from_interactions(interactions):
        interaction_str = " ".join(latex(f) for f in fields)
        coeff = simplify_coeff(coefficient) if simplify_coeff else coefficient
        cells = [f"${interaction_str}$", f"${latex(coeff)}$"]
        if extra_column is not None:
            cells.append(f"${latex(extra_column(coefficient))}$")
        table += " & ".join(cells) + " \\\\\n"
        table += "\\hline\n"

    table += r"\end{array}"
    return table
