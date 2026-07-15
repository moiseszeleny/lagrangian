"""Dirac / Clifford algebra layer.

Ported from ``bsm-calc/models/DLRSM1/dirac.py``.  Non-commutative symbolic
gamma matrices with a rule-based Clifford simplifier; chiral projectors with
their algebra baked into ``__mul__``/``__add__``.

Changes with respect to the original:
- the ``diracPR = None`` forward-declaration wart is gone: projector algebra
  uses ``isinstance`` checks against the singleton classes;
- no import-time dependency on the momentum function;
- metric signature defaults to the house convention ``'+---'``
  (``feynlag.conventions.METRIC_SIGNATURE``).
"""

from sympy import (
    Add, Expr, Integer, KroneckerDelta, Mul, Pow, S, expand,
    preorder_traversal, sympify, Function,
)
from sympy.printing.latex import LatexPrinter

from .conventions import METRIC_SIGNATURE

__all__ = [
    "MetricTensor", "minkowski_metric", "DiracGamma", "DiracGammaLower",
    "DiracIdentity", "DiracZero", "PL", "PR", "diracI", "dirac0",
    "diracPL", "diracPR", "gamma_simplify", "dirac_conjugate",
]

#: Symbolic metric tensor head used inside expressions: ``MetricTensor(mu, nu)``.
MetricTensor = Function("MetricTensor")


def minkowski_metric(idx1, idx2, convention=METRIC_SIGNATURE):
    """Minkowski metric component ``g^{idx1 idx2}``.

    Concrete integer indices (0..3) give ±1/0; symbolic indices give a
    KroneckerDelta expression assuming 4D spacetime with index 0 time-like.

    >>> minkowski_metric(0, 0), minkowski_metric(1, 1), minkowski_metric(0, 1)
    (1, -1, 0)
    """
    idx1_s = sympify(idx1)
    idx2_s = sympify(idx2)

    time_sign = S.One if convention == "+---" else -S.One
    space_sign = -S.One if convention == "+---" else S.One

    if isinstance(idx1_s, Integer) and isinstance(idx2_s, Integer):
        if idx1_s == idx2_s:
            if idx1_s == 0:
                return time_sign
            if idx1_s in (1, 2, 3):
                return space_sign
            # out-of-range integer: fall through to symbolic formula
        else:
            return S.Zero

    if convention == "+---":
        diag_symbolic = 2 * KroneckerDelta(idx1_s, 0) - 1
    elif convention == "-+++":
        diag_symbolic = 1 - 2 * KroneckerDelta(idx1_s, 0)
    else:
        raise ValueError("Unsupported metric convention. Use '+---' or '-+++'.")

    if idx1_s == idx2_s:
        return diag_symbolic
    return KroneckerDelta(idx1_s, idx2_s) * diag_symbolic


class DiracLatexPrinter(LatexPrinter):
    """LaTeX printer that parenthesizes powers of gamma matrices."""

    def _print_Pow(self, expr, rational=False):
        base, exp = expr.base, expr.exp
        if isinstance(base, (DiracGamma, DiracGammaLower)):
            base_str = self.doprint(base)
            exp_str = self.doprint(exp)
            if exp.is_Rational and exp.p < 0:
                return rf"\frac{{1}}{{{self._print_Pow(base ** (-exp), rational)}}}"
            if exp == -1:
                return rf"\frac{{1}}{{{base_str}}}"
            return rf"({base_str})^{{{exp_str}}}"
        return super()._print_Pow(expr, rational=rational)


dirac_printer = DiracLatexPrinter()


class DiracGamma(Expr):
    """Dirac gamma matrix ``γ^μ`` (upper Lorentz index).

    The full Clifford algebra is NOT applied automatically on multiplication;
    use :func:`gamma_simplify`.
    """

    is_commutative = False

    def __new__(cls, index):
        return super().__new__(cls, sympify(index))

    @property
    def index(self):
        return self.args[0]

    def __repr__(self):
        return f"DiracGamma({repr(self.index)})"

    def _sympystr(self, printer):
        return f"gamma({printer.doprint(self.index)})"

    def _latex(self, printer):
        return rf"\gamma^{{{printer.doprint(self.index)}}}"


class DiracGammaLower(Expr):
    """Dirac gamma matrix ``γ_μ`` (lower Lorentz index)."""

    is_commutative = False

    def __new__(cls, index):
        return super().__new__(cls, sympify(index))

    @property
    def index(self):
        return self.args[0]

    def __repr__(self):
        return f"DiracGammaLower({repr(self.index)})"

    def _sympystr(self, printer):
        return f"gamma_({printer.doprint(self.index)})"

    def _latex(self, printer):
        return rf"\gamma_{{{printer.doprint(self.index)}}}"


class DiracIdentity(Expr):
    """4x4 identity in Dirac space (singleton ``diracI``)."""

    is_commutative = True
    is_Identity = True

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __mul__(self, other):
        if isinstance(other, (DiracGamma, DiracGammaLower)) or \
                getattr(other, "is_projector", False):
            return other
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, (DiracGamma, DiracGammaLower)) or \
                getattr(other, "is_projector", False):
            return other
        return super().__mul__(other)

    def __add__(self, other):
        if isinstance(other, DiracZero):
            return self
        if other == self:
            return 2 * self
        return super().__add__(other)

    def __repr__(self):
        return "DiracIdentity()"

    def _sympystr(self, printer):
        return "DiracI"

    def _latex(self, printer):
        return r"I_{4}"


class DiracZero(Expr):
    """4x4 zero matrix in Dirac space (singleton ``dirac0``)."""

    is_commutative = True
    is_Zero = True

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __add__(self, other):
        if isinstance(other, (DiracGamma, DiracGammaLower, DiracIdentity, PL, PR)):
            return other
        raise NotImplementedError(
            f"Dirac0 lives in Dirac space; the sum {self} + {other} is not defined"
        )

    def __radd__(self, other):
        return other

    def __repr__(self):
        return "DiracZero()"

    def _sympystr(self, printer):
        return "Dirac0"

    def _latex(self, printer):
        return r"0_{4}"


class PR(Expr):
    """Right-handed chiral projector ``P_R = (1 + γ₅)/2`` (singleton).

    Algebra baked into operators: ``P_R² = P_R``, ``P_R P_L = 0``,
    ``P_R + P_L = I₄``.  Rules involving gamma matrices are handled by
    :func:`gamma_simplify`, not here.
    """

    is_commutative = False
    is_projector = True

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __mul__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PR):
            return self                     # PR * PR = PR
        if isinstance(other, PL):
            return dirac0                   # PR * PL = 0
        if isinstance(other, DiracIdentity):
            return self                     # PR * I = PR
        if isinstance(other, DiracZero):
            return dirac0                   # PR * 0 = 0
        return super().__mul__(other)

    def __rmul__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PL):
            return dirac0                   # PL * PR = 0
        if isinstance(other, DiracIdentity):
            return self                     # I * PR = PR
        if isinstance(other, DiracZero):
            return dirac0                   # 0 * PR = 0
        return super().__rmul__(other)

    def __add__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PL):
            return diracI                   # PR + PL = I
        if isinstance(other, DiracZero):
            return self                     # PR + 0 = PR
        if isinstance(other, PR):
            return 2 * self                 # PR + PR = 2 PR
        return super().__add__(other)

    def __radd__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PL):
            return diracI
        if isinstance(other, DiracZero):
            return self
        return super().__radd__(other)

    def __repr__(self):
        return "PR()"

    def _sympystr(self, printer):
        return "PR"

    def _latex(self, printer):
        return r"P_R"


class PL(Expr):
    """Left-handed chiral projector ``P_L = (1 − γ₅)/2`` (singleton).

    Algebra baked into operators: ``P_L² = P_L``, ``P_L P_R = 0``,
    ``P_L + P_R = I₄``.
    """

    is_commutative = False
    is_projector = True

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __mul__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PL):
            return self                     # PL * PL = PL
        if isinstance(other, PR):
            return dirac0                   # PL * PR = 0
        if isinstance(other, DiracIdentity):
            return self                     # PL * I = PL
        if isinstance(other, DiracZero):
            return dirac0                   # PL * 0 = 0
        return super().__mul__(other)

    def __rmul__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PR):
            return dirac0                   # PR * PL = 0
        if isinstance(other, DiracIdentity):
            return self                     # I * PL = PL
        if isinstance(other, DiracZero):
            return dirac0                   # 0 * PL = 0
        return super().__rmul__(other)

    def __add__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PR):
            return diracI                   # PL + PR = I
        if isinstance(other, DiracZero):
            return self                     # PL + 0 = PL
        if isinstance(other, PL):
            return 2 * self                 # PL + PL = 2 PL
        return super().__add__(other)

    def __radd__(self, other):
        other = sympify(other, strict=True)
        if isinstance(other, PR):
            return diracI
        if isinstance(other, DiracZero):
            return self
        return super().__radd__(other)

    def __repr__(self):
        return "PL()"

    def _sympystr(self, printer):
        return "PL"

    def _latex(self, printer):
        return r"P_L"


# --- Singleton instances -----------------------------------------------------
diracI = DiracIdentity()
dirac0 = DiracZero()
diracPR = PR()
diracPL = PL()


def gamma_simplify(expr, metric_func=minkowski_metric, identity=diracI, zero=dirac0):
    """Apply Clifford algebra rules to ``expr`` with canonical ordering.

    Rules applied iteratively until a fixed point:
    - ``{γ^μ, γ^ν} = 2 g^{μν} I₄`` (adjacent gammas reordered canonically),
    - metric contraction ``g_{μν} γ^ν → γ_μ``,
    - identity / zero matrix absorption.

    Ported verbatim in behavior from DLRSM1.
    """
    current_expr = expand(expr)

    while True:
        simplified_expr_prev_iter = current_expr

        # --- identity and zero rules ---
        new_expr = current_expr
        new_expr = new_expr.replace(
            lambda x: isinstance(x, Mul) and identity in x.args,
            lambda x: Mul(*[arg for arg in x.args if arg != identity]),
        )
        new_expr = new_expr.replace(
            lambda x: isinstance(x, Mul) and zero in x.args,
            lambda x: zero,
        )
        new_expr = new_expr.replace(
            lambda x: isinstance(x, Add) and zero in x.args,
            lambda x: Add(*[arg for arg in x.args if arg != zero]),
        )
        current_expr = new_expr

        # --- metric contraction g(mu,nu) * gamma(nu) -> gamma_(mu) ---
        contraction_replacements = {}
        for term in preorder_traversal(current_expr):
            if term in contraction_replacements:
                continue
            if not isinstance(term, Mul):
                continue
            args = list(term.args)
            metric_positions = [i for i, a in enumerate(args)
                                if isinstance(a, Function) and a.func == MetricTensor]
            gamma_positions = [i for i, a in enumerate(args)
                               if isinstance(a, DiracGamma)]

            new_gamma_lower = None
            contracted = (-1, -1)
            for m_pos in metric_positions:
                metric_arg = args[m_pos]
                if len(metric_arg.args) != 2:
                    continue
                m_idx1, m_idx2 = metric_arg.args
                for g_pos in gamma_positions:
                    g_idx = args[g_pos].index
                    if g_idx == m_idx2:
                        contracted = (m_pos, g_pos)
                        new_gamma_lower = DiracGammaLower(m_idx1)
                        break
                    if g_idx == m_idx1:
                        contracted = (m_pos, g_pos)
                        new_gamma_lower = DiracGammaLower(m_idx2)
                        break
                if new_gamma_lower is not None:
                    break

            if new_gamma_lower is not None:
                new_args = [a for i, a in enumerate(args) if i not in contracted]
                new_args.append(new_gamma_lower)
                contraction_replacements[term] = Mul(*new_args).expand()

        if contraction_replacements:
            current_expr = current_expr.xreplace(contraction_replacements)

        # --- Clifford algebra with canonical ordering ---
        clifford_replacements = {}
        for term in preorder_traversal(current_expr):
            if (isinstance(term, Pow) and isinstance(term.base, DiracGamma)
                    and term.exp == 2):
                mu = term.base.index
                if term not in clifford_replacements:
                    clifford_replacements[term] = metric_func(mu, mu) * identity
                continue

            if isinstance(term, Mul):
                args = list(term.args)
                i = 0
                while i < len(args) - 1:
                    g1, g2 = args[i], args[i + 1]
                    if isinstance(g1, DiracGamma) and isinstance(g2, DiracGamma):
                        mu, nu = g1.index, g2.index
                        if mu == nu:
                            term_to_insert = metric_func(mu, mu) * identity
                            new_args = args[:i] + [term_to_insert] + args[i + 2:]
                            if term not in clifford_replacements:
                                clifford_replacements[term] = Mul(*new_args).expand()
                            break
                        def index_is_greater(idx1, idx2):
                            s1, s2 = str(idx1), str(idx2)
                            try:
                                return int(s1) > int(s2)
                            except ValueError:
                                return s1 > s2
                        if index_is_greater(mu, nu):
                            g_mu_nu = metric_func(mu, nu)
                            term_to_insert = (2 * g_mu_nu * identity
                                              - DiracGamma(nu) * DiracGamma(mu))
                            new_args = args[:i] + [term_to_insert] + args[i + 2:]
                            if term not in clifford_replacements:
                                clifford_replacements[term] = Mul(*new_args).expand()
                            break
                    i += 1

        if clifford_replacements:
            current_expr = current_expr.xreplace(clifford_replacements)

        if current_expr == simplified_expr_prev_iter:
            break

    # I₄ is the multiplicative unit of the Dirac-space algebra: normalize it
    # to the scalar 1 so results are uniform (the in-loop rules already strip
    # it inside products; this covers standalone occurrences too).
    current_expr = current_expr.xreplace({identity: S.One})
    return expand(current_expr)


def dirac_conjugate(gamma):
    """Dirac-conjugate Γ̄ = γ⁰ Γ† γ⁰ of a Bilinear middle slot Γ.

    Used to build the hermitian conjugate of a fermion sandwich:
    ``(ψ̄₁ Γ ψ₂)^† = ψ̄₂ Γ̄ ψ₁``.  Covers exactly the Γ structures this
    library builds (``diracI``/``1``, ``diracPL``, ``diracPR``, bare
    ``DiracGamma(mu)``, and ``DiracGamma(mu)*diracPL``/``diracPR``):

    - ``γ⁰`` is hermitian and ``γ^{μ†} = γ⁰γ^μγ⁰`` (standard Dirac
      representation with the (+,−,−,−) metric), so a bare ``γ^μ`` is
      self-conjugate: ``γ̄^μ = γ^μ``.
    - ``P_L``/``P_R`` are hermitian but anticommute with ``γ⁰`` (since
      ``γ₅`` anticommutes with every ``γ^μ``), so ``γ⁰P_Lγ⁰ = P_R`` and vice
      versa: **bare** chiral projectors swap.
    - Combined with a gamma matrix the two effects cancel — ``γ^μP_L =
      P_Rγ^μ`` — so a **vector current** ``γ^μP_L``/``γ^μP_R`` is
      self-conjugate (the standard "V−A ports to V−A" result: the h.c. of
      ``ν̄γ^μP_Lℓ`` is ``ℓ̄γ^μP_Lν``, same Γ, bar/field swapped).

    Raises:
        NotImplementedError: for any other Γ structure (no silent guess).
    """
    if gamma == diracI or gamma == S.One:
        return diracI
    if gamma == diracPL:
        return diracPR
    if gamma == diracPR:
        return diracPL

    factors = Mul.make_args(gamma)
    gammas = [f for f in factors if isinstance(f, DiracGamma)]
    projectors = [f for f in factors if isinstance(f, (PL, PR))]
    rest = [f for f in factors if f not in gammas and f not in projectors]
    if len(gammas) == 1 and len(projectors) <= 1 and not rest:
        # exactly one gamma^mu, at most one projector: the two conjugation
        # effects cancel (gamma^mu P_L = P_R gamma^mu), self-conjugate.
        return gamma
    raise NotImplementedError(
        f"dirac_conjugate: no rule for Γ = {gamma!r}; only diracI, diracPL, "
        f"diracPR, DiracGamma(mu), and DiracGamma(mu)*diracPL/diracPR are "
        f"supported (e.g. products of two or more gamma matrices need "
        f"{{γ^μ,γ^ν}}=2g^{{μν}} reduction first, and reverse order under "
        f"conjugation — not handled here)")
