"""The ε (γ₅) trace algebra: Tier 2 of ``docs/manual/scattering_roadmap.md``.

:func:`~feynlag.pheno.lorentz.reduce_projectors` proves the ``Tr[Xγ₅]`` piece
of a chiral chain vanishes for a 1→2 decay and refuses to compute it
otherwise — SymPy's own tensor engine has no γ₅ or ε (Levi-Civita) object at
all (``sympy.physics.hep.gamma_matrices`` ships neither; see that module's
docstring). This module is where the refused term gets computed for real,
for the one topology :class:`~feynlag.pheno.diagrams.Amplitude` builds: two
chiral fermion chains meeting at a single s-channel propagator.

Two identities do all the work, and **both signs are derived here, never
quoted from a textbook** — following the precedent of
:func:`feynlag.dirac.majorana_symmetry_sign`, which derives its own sign from
the same explicit Dirac-basis representation (:func:`feynlag.dirac._dirac_rep`)
rather than asserting one:

1. ``Tr[γ^aγ^bγ^cγ^dγ₅] = κ·ε^{abcd}`` — :func:`gamma5_trace_coefficient`
   reads off ``κ`` by tracing the literal ``γ^0γ^1γ^2γ^3γ₅`` matrix and
   dividing by ``ε^{0123}`` (which is ``+1`` by the normalization
   :func:`levi_civita_array` states explicitly).
2. ``ε^{a…}ε^{b…} = s_det·det[g^{a_ib_j}]`` — :func:`epsilon_product_sign`
   reads off ``s_det`` from the same normalization against the ``(+,−,−,−)``
   metric.

:func:`epsilon_pair_tensor` expands identity 2 as 24 signed products of four
metrics with all free Lorentz indices left **open** (never contracted here);
each metric slot resolves by what occupies it — two momenta contract through
a fresh dummy, a momentum meeting a free index just evaluates the head there,
two free indices meet the ordinary ``LorentzIndex.metric``. This needs no
``Eps`` tensor object and no partial-contraction identity, and the
all-momentum case (both chains' slots given entirely as momenta) degenerates
directly to the roadmap's ``−det[p_i·p'_j]`` Gram determinant — verified by
construction, not asserted.

The one loose end identity 1+2 alone don't close: when only *one* chain in a
diagram carries a non-zero ε coefficient, ``|M|²`` still has a **single**-ε
cross term (the other chain's ordinary, γ₅-free trace times this chain's ε
piece). Such a term is a scalar times ``ε(four momenta drawn from the
diagram's own external legs)``, and a 2→2 diagram's four legs satisfy
``k₄ = k₁+k₂−k₃`` — rank 3, not 4 — so every such ε is provably zero.
Because ``ε(a,b,c,d)² = −det[·]`` (identity 2 again, at ``a=b``),
:func:`assert_epsilon_single_vanishes` proves this by computing the actual
Gram determinant from the diagram's own ``kin.dot`` table, rather than
assuming it from a momentum count — a computed guard, in the same
prove-or-refuse spirit as :func:`~feynlag.pheno.lorentz.reduce_projectors`,
but one that re-derives the fact from the kinematics in hand instead of a
proxy that would need re-tuning for a future topology.
"""

import itertools

import sympy as sp
from sympy.combinatorics import Permutation
from sympy.physics.hep.gamma_matrices import LorentzIndex

from ..dirac import _dirac_rep
from .lorentz import index

__all__ = [
    "assert_epsilon_single_vanishes", "epsilon_pair_tensor",
    "epsilon_product_sign", "gamma5_trace_coefficient", "gram_determinant",
    "levi_civita_array",
]

#: ``(+,−,−,−)`` — the one metric convention everything here is derived
#: against (matches ``dirac.py``'s ``_dirac_rep`` and
#: ``tests/test_scattering.py``'s explicit-matrix oracle).
_MET = sp.diag(1, -1, -1, -1)

_dummy_counter = itertools.count()


def levi_civita_array():
    """Rank-4 totally antisymmetric array with ``ε^{0123} = +1``.

    This is the *stated* normalization every sign in this module is derived
    against — nothing here assumes a sign independently of this array.
    """
    cache = levi_civita_array.__dict__.get("_cache")
    if cache is not None:
        return cache
    arr = sp.MutableDenseNDimArray.zeros(4, 4, 4, 4)
    for perm in itertools.permutations(range(4)):
        arr[perm] = Permutation(list(perm)).signature()
    levi_civita_array.__dict__["_cache"] = arr
    return arr


def gamma5_trace_coefficient():
    """``κ`` in ``Tr[γ^aγ^bγ^cγ^dγ₅] = κ·ε^{abcd}``.

    Derived, not quoted: traces the literal ``γ^0γ^1γ^2γ^3γ₅`` matrix from
    :func:`feynlag.dirac._dirac_rep` and divides by ``ε^{0123} = 1`` (so the
    division is really just a normalization check, not new information).
    Both sides are totally antisymmetric under exchange of any two of
    ``a,b,c,d`` when they are distinct, so this one component fixes ``κ`` for
    all of them.
    """
    cache = gamma5_trace_coefficient.__dict__.get("_cache")
    if cache is not None:
        return cache
    rep = _dirac_rep()
    g = [rep[("g", m)] for m in range(4)]
    trace = (g[0] * g[1] * g[2] * g[3] * rep["g5"]).trace()
    kappa = sp.nsimplify(trace) / levi_civita_array()[0, 1, 2, 3]
    gamma5_trace_coefficient.__dict__["_cache"] = kappa
    return kappa


def epsilon_product_sign():
    """``s_det`` in ``ε^{a…}ε^{b…} = s_det·det[g^{a_ib_j}]``.

    Derived, not quoted: evaluated at ``a_i=b_i=i`` (identity slots) so the
    left side is ``ε^{0123}ε^{0123} = 1`` (:func:`levi_civita_array`'s stated
    normalization) and the right side is ``s_det·det[g^{ij}]`` with
    ``g^{ij}`` the ``(+,−,−,−)`` metric (self-inverse, so upper and lower
    forms coincide numerically).
    """
    cache = epsilon_product_sign.__dict__.get("_cache")
    if cache is not None:
        return cache
    LC = levi_civita_array()
    lhs = LC[0, 1, 2, 3] * LC[0, 1, 2, 3]
    s_det = sp.nsimplify(lhs / _MET.det())
    epsilon_product_sign.__dict__["_cache"] = s_det
    return s_det


def _fresh_dummy():
    """A never-before-used Lorentz dummy index name.

    Two momentum–momentum slot pairings can occur within the *same* additive
    term of :func:`epsilon_pair_tensor`'s 24-term expansion (e.g. both chains'
    slots given entirely as momenta); reusing a fixed dummy name across them
    would make one index name appear four times instead of two contracted
    pairs, which :func:`~feynlag.pheno.lorentz.contract_to_dots` rejects.
    """
    return index(f"eps_dummy_{next(_dummy_counter)}")


def _resolve_slot_pair(slot_a, slot_b):
    """One metric factor of the 24-term expansion, resolved by slot content.

    ``slot_a``/``slot_b`` are ``('m', TensorHead)`` for a momentum or
    ``('i', TensorIndex)`` for a free Lorentz index (open, not contracted
    here). See the module docstring's slot-resolution table.
    """
    tag_a, val_a = slot_a
    tag_b, val_b = slot_b
    if tag_a == "m" and tag_b == "m":
        d = _fresh_dummy()
        return val_a(d) * val_b(-d)
    if tag_a == "m":
        return val_a(val_b)
    if tag_b == "m":
        return val_b(val_a)
    return LorentzIndex.metric(val_a, val_b)


def epsilon_pair_tensor(slots_a, slots_b):
    """``ε^{a…}ε^{b…}`` for two 4-slot index/momentum assignments, expanded.

    Implements ``ε^{a1a2a3a4}ε^{b1b2b3b4} = s_det·det[g^{a_ib_j}]`` as the
    signed sum over the 24 permutations pairing slot ``i`` of ``slots_a``
    with slot ``perm(i)`` of ``slots_b``, each pairing resolved by
    :func:`_resolve_slot_pair`. Any free Lorentz indices among the slots stay
    open in the result — reduction happens later, alongside the rest of the
    diagram, via ``.contract_metric``/
    :func:`~feynlag.pheno.lorentz.contract_to_dots`.

    Args:
        slots_a, slots_b: each a 4-tuple of ``('m', TensorHead)``/
            ``('i', TensorIndex)`` slots (see
            :meth:`~feynlag.pheno.diagrams.SpinorChain.epsilon_structure`).
    """
    total = sp.S.Zero
    for perm in itertools.permutations(range(4)):
        sign = Permutation(list(perm)).signature()
        term = sp.S.One
        for i in range(4):
            term = term * _resolve_slot_pair(slots_a[i], slots_b[perm[i]])
        total += sign * term
    return epsilon_product_sign() * total


def gram_determinant(heads, dot):
    """``det[dot(h_i, h_j)]`` for a sequence of momentum heads.

    Args:
        heads: a sequence of momentum ``TensorHead``\\ s.
        dot: callable ``(head_a, head_b) -> scalar``, e.g.
            :meth:`~feynlag.pheno.kinematics.TwoToTwoKinematics.dot`.
    """
    n = len(heads)
    return sp.Matrix(n, n, lambda i, j: dot(heads[i], heads[j])).det()


def assert_epsilon_single_vanishes(heads, dot):
    """Prove-or-refuse guard for the single-ε cross terms.

    A single-ε term is a scalar times ``ε(four momenta drawn from `heads`)``,
    and ``ε(a,b,c,d)² = epsilon_product_sign()⁻¹·(−1)·det[dot]`` up to the
    normalization in :func:`epsilon_product_sign` — so a vanishing Gram
    determinant over every 4-subset of ``heads`` *proves* every such ε is
    zero, term by term, not just plausibly small.

    Args:
        heads: every momentum ``TensorHead`` appearing in the diagram (see
            :func:`~feynlag.pheno.diagrams._momentum_heads`).
        dot: as in :func:`gram_determinant`.

    Raises:
        NotImplementedError: some 4-subset has a non-zero Gram determinant —
            a topology with genuinely non-vanishing single-ε terms, beyond
            Tier 2's single s-channel 2→2 diagram.
    """
    unique = list(dict.fromkeys(heads))
    if len(unique) < 4:
        return
    for subset in itertools.combinations(unique, 4):
        det = sp.simplify(gram_determinant(subset, dot))
        if det != 0:
            names = [h.name for h in subset]
            raise NotImplementedError(
                f"assert_epsilon_single_vanishes: the Gram determinant for "
                f"momenta {names} is {det} != 0 — a single-ε cross term does "
                f"not provably vanish here. This is beyond Tier 2's single "
                f"s-channel 2→2 diagram (which always has exactly 3 "
                f"independent external momenta); refusing to silently drop "
                f"it.")
