"""Covariant Dirac-trace engine for squared amplitudes.

The computation path is **manifestly covariant**: traces are evaluated with
SymPy's own Clifford-algebra engine (:mod:`sympy.physics.hep.gamma_matrices`,
which ships with SymPy — no new dependency), producing Lorentz-invariant
contractions that are reduced to on-shell dot products ``p_i·p_j`` at the very
end.  Explicit 4×4 Dirac matrices are deliberately **not** on this path; they
appear only as the independent test oracle (``tests/test_pheno.py``).

SymPy's engine does not cover three things this module needs, all handled here:

1. **Mass terms.**  ``gamma_trace`` chokes on ``p̸ + m``: the sum is a
   ``TensAdd`` whose scalar argument has no ``.sorted_components()``.
   :func:`dirac_trace` splits the scalar (identity) pieces off and applies
   ``Tr[c·I₄] = 4c`` to them directly.

2. **γ₅ / chiral projectors.**  SymPy has no ``G5``, so ``P_L``/``P_R`` cannot
   be represented at all.  :func:`reduce_projectors` implements the projector
   algebra and reduces any chain to a γ₅-free trace times a factor — see that
   function for the derivation and the precondition it enforces.  Which
   projector a vertex carries is decided upstream by
   :func:`~feynlag.pheno.vertices.classify_gamma`.

3. **Reduction to scalars.**  ``TensExpr.replace_with_arrays`` raises
   ``ValueError`` on contracted dummy indices (SymPy 1.14), so
   :func:`contract_to_dots` walks the tensor structure directly.
"""

import sympy as sp
from sympy.physics.hep.gamma_matrices import GammaMatrix, LorentzIndex
from sympy.physics.hep.gamma_matrices import gamma_trace as _sympy_gamma_trace
from sympy.tensor.tensor import (
    TensAdd, TensExpr, TensMul, TensorHead, tensor_indices,
)

__all__ = [
    "LorentzIndex", "contract_to_dots", "dirac_trace", "index", "momentum",
    "reduce_projectors", "slashed",
]

#: cache so repeated calls share one ``TensorHead`` per momentum name — two
#: distinct heads with the same name do not contract with each other.
_MOMENTA = {}
_INDICES = {}


def momentum(name):
    """The (cached) rank-1 ``TensorHead`` for a momentum called ``name``."""
    if name not in _MOMENTA:
        _MOMENTA[name] = TensorHead(name, [LorentzIndex])
    return _MOMENTA[name]


def index(name):
    """A (cached) Lorentz ``TensorIndex`` called ``name``."""
    if name not in _INDICES:
        _INDICES[name] = tensor_indices(name, LorentzIndex)
    return _INDICES[name]


def slashed(p, dummy):
    """``p̸ = p_μ γ^μ`` for the momentum head ``p``, summed over ``dummy``."""
    return p(dummy) * GammaMatrix(-dummy)


# ------------------------------------------------------------------- traces

def dirac_trace(expr):
    """Trace of a Dirac chain, tolerating scalar (identity) pieces.

    ``sympy.physics.hep.gamma_trace`` handles only pure gamma products, so a
    spin-sum factor ``p̸ + m`` breaks it.  Here the expression is expanded and
    each additive term is traced separately: terms carrying gamma matrices go
    to SymPy, purely scalar terms use ``Tr[c·I₄] = 4c``.
    """
    if not isinstance(expr, TensExpr):
        return 4 * sp.sympify(expr)
    expr = expr.expand()
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    total = sp.S.Zero
    for term in terms:
        if isinstance(term, TensExpr) and term.atoms(GammaMatrix):
            total += _sympy_gamma_trace(term)
        else:
            total += 4 * term
    return total


def reduce_projectors(expr, chirality, n_free_indices=0, n_momenta=2):
    """Reduce a chiral chain to a γ₅-free trace times a scalar factor.

    ``P_{L,R} = (1 ∓ γ₅)/2``, and SymPy's tensor engine has no γ₅ at all.  The
    reduction used here is exact algebra up to one dropped term:

    - projectors are moved to one end with ``P_{L,R} γ^μ = γ^μ P_{R,L}``,
    - ``P² = P`` collapses repeats and ``P_L P_R = 0`` kills mixed chains
      (classified upstream by
      :func:`~feynlag.pheno.vertices.classify_gamma`),
    - a single remaining projector gives
      ``Tr[X P_{L,R}] = ½ Tr[X] ∓ ½ Tr[X γ₅]``.

    The ``Tr[X γ₅]`` term is a totally antisymmetric ε tensor.  It **vanishes
    identically for a 1→2 decay**: ε needs four independent four-vectors, but a
    two-body final state supplies only two independent momenta (``P = p₁ + p₂``
    is not independent), and any leftover free index is contracted with a
    polarization sum that is *symmetric* in its two indices.  So the ε term is
    dropped — but only after checking that precondition.

    Args:
        expr: the γ₅-free chain (the projector is carried in ``chirality``,
            not inside ``expr``).
        chirality: ``'L'``, ``'R'`` or ``None``.
        n_free_indices: uncontracted Lorentz indices that will meet a
            **symmetric** polarization sum.
        n_momenta: independent momenta in the process (2 for a 1→2 decay).

    Returns:
        ``(expr, factor)`` — trace ``expr`` and multiply by ``factor``.

    Raises:
        NotImplementedError: if the ε term is not provably zero.  This is a
            hard guard, not a comment: a later 1→3 extension fails loudly here
            instead of silently returning a wrong number.
    """
    if chirality is None:
        return expr, sp.S.One
    if chirality not in ("L", "R"):
        raise ValueError(f"chirality must be 'L', 'R' or None, got {chirality!r}")
    # ε^{μνρσ} is totally antisymmetric: it needs four independent vectors.
    # Each free index is contracted with a symmetric tensor and so cannot
    # supply one; only independent momenta can.
    if n_momenta > 3:
        raise NotImplementedError(
            f"reduce_projectors: the γ₅ (ε-tensor) term is only provably zero "
            f"for at most 3 independent momenta, got {n_momenta}. A 1→3 or "
            f"2→2 process needs genuine ε-tensor algebra, which this engine "
            f"does not implement — refusing to silently drop it.")
    if n_free_indices > 2:
        raise NotImplementedError(
            f"reduce_projectors: {n_free_indices} free Lorentz indices — the "
            f"γ₅ term is only provably killed by symmetric polarization sums "
            f"for at most 2. Refusing to silently drop it.")
    return expr, sp.Rational(1, 2)


# ------------------------------------------------------- scalar reduction

def contract_to_dots(expr, dot):
    """Reduce a fully contracted tensor expression to a plain scalar.

    ``TensExpr.replace_with_arrays`` raises ``ValueError: p1(L_0) not found``
    on contracted dummy indices (SymPy 1.14), so the structure is walked
    directly: for each ``TensMul``, every slot of every component is paired
    with its index, dummy indices are matched up, and each matched pair of
    momentum heads becomes ``dot(head_a, head_b)``.

    Any surviving ``metric`` component is rejected rather than guessed at: the
    caller must run ``.contract_metric(LorentzIndex.metric)`` first (a naive
    per-pair rule would turn ``g^{ab}g_{ab} = 4`` into ``16``).

    .. note::
       This is the one function touching semi-private SymPy tensor API
       (``TensMul.components``/``.get_indices()``/``.coeff``).  It is isolated
       here on purpose and pinned by unit tests, so a future SymPy change
       breaks exactly one place, loudly.

    Args:
        expr: tensor expression with no free indices.
        dot: callable ``(head_a, head_b) -> scalar`` giving ``p_a·p_b``.

    Raises:
        ValueError: an index is left uncontracted (a bug upstream).
    """
    if not isinstance(expr, TensExpr):
        return sp.sympify(expr)
    expr = expr.expand()
    terms = expr.args if isinstance(expr, TensAdd) else [expr]
    total = sp.S.Zero
    for term in terms:
        if not isinstance(term, TensMul):
            total += term
            continue
        # one head entry per index slot, aligned with get_indices()
        heads = []
        for component in term.components:
            if component.name == "metric":
                raise ValueError(
                    f"contract_to_dots: an uncontracted metric survives in "
                    f"{term}; call .contract_metric(LorentzIndex.metric) on "
                    f"the expression first")
            heads.extend([component] * component.rank)
        indices = term.get_indices()
        if len(heads) != len(indices):
            raise ValueError(
                f"contract_to_dots: {len(heads)} tensor slots but "
                f"{len(indices)} indices in {term} — SymPy's tensor layout "
                f"has changed; this function needs updating")
        slots = {}
        for position, idx in enumerate(indices):
            slots.setdefault(idx.name, []).append(position)
        value = term.coeff
        for name, positions in slots.items():
            if len(positions) != 2:
                raise ValueError(
                    f"contract_to_dots: index {name!r} appears "
                    f"{len(positions)} times in {term}; expected a contracted "
                    f"pair (the expression must have no free indices)")
            value *= dot(heads[positions[0]], heads[positions[1]])
        total += value
    return total
