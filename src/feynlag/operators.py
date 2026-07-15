"""Derivative operators and momentum-space substitution.

Ported from ``bsm-calc/models/DLRSM1``: ``symbolic_tools.PartialMu`` /
``momentum`` and ``feynman_rules.D_linear`` (here with the field set passed
explicitly instead of read from module globals — the original recursion also
silently dropped the ``fields`` argument, fixed here).

Momentum convention (see CONVENTIONS.md): ``∂_μ φ → i p(φ) φ`` where ``p(φ)``
is the (symbolic) momentum associated with the leg carrying field ``φ``.
Covariant derivatives ``Dmu`` and field strengths arrive in Phase 3.
"""

from sympy import Function, I, Matrix, Mul

__all__ = ["momentum", "PartialMu", "D_linear", "expand_derivatives",
           "to_momentum_space", "Dmu"]

#: Symbolic momentum carried by a field: ``momentum(phi)`` prints as ``p(phi)``.
momentum = Function("p")


class PartialMu(Function):
    """The differential operator ``∂_μ`` acting on a field expression.

    Stays unevaluated on construction; :meth:`doit` applies the momentum-space
    rule ``∂_μ φ → i p(φ) φ``.
    """

    @classmethod
    def eval(cls, arg):
        # ∂_μ of a manifestly constant expression vanishes.
        if arg.is_number:
            return arg * 0
        return None

    def doit(self, **hints):
        field = self.args[0]
        return I * momentum(field) * field

    def _eval_conjugate(self):
        # ∂_μ is a real operator: (∂_μ φ)* = ∂_μ (φ*)
        from sympy import conjugate
        return PartialMu(conjugate(self.args[0]))


def D_linear(expr, fields):
    """Distribute ``∂_μ`` over sums and products (linearity + Leibniz rule).

    ``expr`` is the *argument* of a ``PartialMu`` head; ``fields`` is the
    collection of field symbols the derivative acts on.  Non-field factors are
    treated as constants.

    Returns the expanded expression with ``PartialMu`` applied to single
    fields only.
    """
    fields = set(fields)
    expr = expr.expand()
    if expr.is_Add:
        return sum(D_linear(arg, fields) for arg in expr.args)

    def _split_factor(f):
        """Return the list of field factors a single factor contributes.

        ``phi`` -> [phi];  ``phi**n`` -> [phi]*n (integer n > 0);
        anything else -> None (constant w.r.t. the fields).
        """
        if f in fields:
            return [f]
        if (f.is_Pow and f.base in fields and f.exp.is_Integer
                and f.exp > 0):
            return [f.base] * int(f.exp)
        return None

    if expr.is_Pow:
        split = _split_factor(expr)
        if split is not None:
            base, n = expr.base, int(expr.exp)
            return n * base ** (n - 1) * PartialMu(base)

    if expr.is_Mul:
        factors = expr.as_ordered_factors()
        field_part = []
        coeff_part = []
        for f in factors:
            split = _split_factor(f)
            if split is None:
                coeff_part.append(f)
            else:
                field_part.extend(split)

        if len(field_part) == 1:
            coeff = Mul(*coeff_part) if coeff_part else 1
            return coeff * PartialMu(field_part[0])
        terms = []
        for i in range(len(field_part)):
            new_factors = field_part.copy()
            new_factors[i] = PartialMu(new_factors[i])
            terms.append(Mul(*coeff_part, *new_factors))
        return sum(terms)

    if expr in fields:
        return PartialMu(expr)
    # Constant with respect to the fields (parameters, VEVs after the shift):
    # ∂_μ(const) = 0.  (The original DLRSM version returned the expression
    # unchanged — a latent bug once VEV-shifted arguments reach here.)
    return expr * 0


def expand_derivatives(expr, fields):
    """Rewrite every ``PartialMu(<product>)`` in ``expr`` via the Leibniz rule."""
    return expr.replace(PartialMu, lambda arg: D_linear(arg, fields)).expand()


def to_momentum_space(expr, fields):
    """Full derivative pipeline: Leibniz-expand then ``∂_μ φ → i p(φ) φ``.

    Derivatives of expressions containing none of the ``fields`` (constants,
    VEVs) are dropped.
    """
    expr = expand_derivatives(expr, fields)
    fields = tuple(fields)
    return expr.replace(PartialMu,
                        lambda arg: (I * momentum(arg) * arg)
                        if arg.has(*fields) else 0)


def Dmu(field, gauge_groups=None):
    """Covariant derivative of a field: column Matrix over components.

    Sign convention (CONVENTIONS.md): ``D_μ = ∂_μ − i g T^a A^a_μ``.

    Args:
        field: a :class:`~feynlag.fields.Field`.
        gauge_groups: groups to gauge (default: every group in
            ``field.reps``).  Each group's boson field is created on demand
            via ``group.bosons()``.

    Returns:
        Matrix ``(D_μ φ)_i = ∂_μ φ_i − i Σ_G g_G Σ_a A^a_μ (T^a φ)_i``.

    The Lorentz index is implicit (carried to momentum space by the
    ``p(φ)`` tagging and reconstructed by the vertex classifier).
    """
    groups = list(gauge_groups) if gauge_groups is not None \
        else list(field.reps.keys())
    vec = field.mat
    result = Matrix([[PartialMu(c)] for c in field.components])
    for group in groups:
        if group not in field.reps:
            continue
        boson = group.bosons()
        for a, T in enumerate(field.generators(group)):
            result -= I * group.g * boson.components[a] * (T * vec)
    return result
