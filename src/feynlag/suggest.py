"""Invariant-term suggestion: enumerate the Lagrangian a model *can* have.

Given the declared fields and symmetries, this module enumerates every
gauge- and discrete-invariant term up to mass dimension 4 — turning the
model-building stage from "hand-derive every invariant, then hope
``check_invariance`` catches your mistakes" into "declare fields, review the
suggested operator basis, attach couplings".

The engine is feynlag's own, already-tested invariance machinery used as an
oracle (:mod:`feynlag.invariance`): candidate terms are *constructed* from
gauge-invariant building blocks, filtered for U(1) neutrality, projected
onto the discrete-invariant subspace by group averaging (the Reynolds
operator), reduced to a linearly independent basis, and then every emitted
term is re-verified against the full ``check_*`` battery.

Public surface:

- :class:`SuggestedTerm` — one invariant, coupling-free, with metadata.
- :func:`suggest_potential` — scalar-potential invariants.
- :func:`suggest_yukawa` — Yukawa / bare-mass invariants.
- :func:`suggest_kinetic` — kinetic/gauge scaffolding (no enumeration).
- :func:`build_lagrangian` — attach couplings and fill a :class:`Lagrangian`.
"""

import itertools
from dataclasses import dataclass, field as dc_field

import sympy as sp

from .fields import Fermion, Scalar, dag
from .groups.gauge import SU2
from .invariance import (
    _apply_field_map, _fermion_transform_discrete, check_gauge_invariance,
    check_discrete_invariance, check_hermiticity, check_mass_dimension,
)
from .operators import Dmu
from .parameters import ExternalParameter
from .vertices.bilinear import (
    Bilinear, MajoranaBilinear, expand_bilinear, fermion_gauge_current,
)
from .vertices.extract import extract_interaction_coefficients

__all__ = ["SuggestedTerm", "suggest_potential", "suggest_yukawa",
           "suggest_kinetic", "build_lagrangian"]

#: hard cap on discrete-group order, to fail loudly on pathological input
_MAX_GROUP_ORDER = 200


# --------------------------------------------------------------------------
# Finite-group enumeration + the Reynolds (group-averaging) projector
# --------------------------------------------------------------------------

def _group_elements(group):
    """Enumerate a discrete group as **words in its generators**.

    Returns a list of words (tuples of generator indices); the empty word is
    the identity.  Distinct group elements are told apart by their action in
    the faithful direct sum of all the group's irreps (the tuple of
    generator-matrix products) — so ``S3`` yields its 6 elements, ``Z_N``
    its ``N``.  Sequential substitution along a word *is* composition, which
    is why storing words (rather than composed substitution dicts) suffices.
    """
    irreps = group._irrep_generators
    n_gens = len(next(iter(irreps.values())))
    labels = list(irreps)

    def fingerprint(word):
        parts = []
        for lab in labels:
            M = sp.eye(irreps[lab][0].shape[0])
            for i in word:
                M = M * irreps[lab][i]
            parts.append(tuple(sp.nsimplify(x) for x in M))
        return tuple(parts)

    identity = ()
    seen = {fingerprint(identity): identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for word in frontier:
            for i in range(n_gens):
                cand = word + (i,)
                fp = fingerprint(cand)
                if fp not in seen:
                    seen[fp] = cand
                    nxt.append(cand)
                    if len(seen) > _MAX_GROUP_ORDER:
                        raise ValueError(
                            f"discrete group {group.name} exceeded "
                            f"{_MAX_GROUP_ORDER} elements during enumeration; "
                            f"refusing to continue")
        frontier = nxt
    return list(seen.values())


def _apply_generator(expr, group, gen_index, components, has_fermion):
    """Apply a single generator's transformation to ``expr``.

    Mirrors one iteration of
    :func:`feynlag.invariance.check_discrete_invariance`: the bosonic
    component substitution (index-preserving through ``PartialMu``) plus, if
    the term carries a :class:`~feynlag.vertices.bilinear.Bilinear`, the
    fermion-leg transform and ``expand_bilinear`` redistribution.
    """
    sub = group.generator_maps()[gen_index]
    out = _apply_field_map(expr, sub, components=components)
    if has_fermion:
        out = _fermion_transform_discrete(out, group, gen_index)
        out = expand_bilinear(out)
    return out


def _apply_element(expr, group, word, components, has_fermion):
    """Apply a whole group element (a word) letter by letter."""
    for i in word:
        expr = _apply_generator(expr, group, i, components, has_fermion)
    return expr


def reynolds_project(expr, discrete_groups):
    """Project ``expr`` onto the joint discrete-invariant subspace.

    ``P(m) = (1/|G|) Σ_{g∈G} g·m`` for each declared discrete group, applied
    in turn (the composition of the per-group Reynolds operators projects
    onto the intersection of invariant subspaces when the group actions
    commute, which holds for independently-declared symmetries; the caller's
    oracle check re-verifies invariance regardless).  Returns the (possibly
    zero) invariant combination.

    For a ``Z_N``-charged monomial this reduces automatically to the usual
    "total charge ≡ 0 (mod N)" filter — the average is either ``m`` itself
    (neutral) or ``0`` (charged).  For ``S3`` it produces the genuine
    invariant linear combinations that no single monomial realizes alone.
    """
    for group in discrete_groups:
        has_fermion = expr.has(Bilinear) or expr.has(MajoranaBilinear)
        components = group.components()
        elements = _group_elements(group)
        total = sp.S.Zero
        for word in elements:
            total += _apply_element(expr, group, word, components,
                                    has_fermion)
        expr = sp.expand(total / len(elements))
        if has_fermion:
            expr = expand_bilinear(expr)
    return sp.expand(expr)


# --------------------------------------------------------------------------
# Linear-independence reduction (shared by potential and Yukawa)
# --------------------------------------------------------------------------

def _normalize_atoms(expr, registry):
    """Replace ``conjugate(...)`` nodes and ``Bilinear`` atoms by stable
    Dummies so the result is a polynomial over plain commuting symbols.

    ``registry`` is shared across all candidate expressions so the *same*
    conjugate/bilinear maps to the *same* Dummy everywhere — the
    prerequisite for comparing coefficient vectors across candidates.  The
    ``conjugate(component)`` substitution is the same trick
    :func:`feynlag.vacuum.masses.charged_mass_matrix` uses (SymPy cannot
    treat ``conjugate(φ)`` as an independent polynomial variable directly).
    """
    expr = sp.expand(expr)
    for node in expr.atoms(sp.conjugate):
        registry.setdefault(node, sp.Dummy(f"conj_{len(registry)}"))
    for node in expr.atoms(Bilinear, MajoranaBilinear):
        registry.setdefault(node, sp.Dummy(f"bil_{len(registry)}"))
    return expr.xreplace(registry)


def _coeff_vector(expr, registry):
    """``{monomial-tuple: coefficient}`` of a normalized candidate."""
    norm = _normalize_atoms(expr, registry)
    symbols = sorted(norm.free_symbols, key=lambda s: s.sort_key())
    table = extract_interaction_coefficients(norm, symbols)
    vector = {}
    for terms in table.values():
        for field_tuple, coeff in terms.items():
            vector[field_tuple] = vector.get(field_tuple, sp.S.Zero) + coeff
    return vector


def _independent_subset(candidates):
    """Indices of a greedily-chosen linearly independent subset.

    Each candidate is expanded into component monomials (via
    :func:`_coeff_vector`) and kept only if its coefficient vector is
    linearly independent of those already kept — yielding a minimal
    operator basis matching literature counts (the 8-term 2HDM, etc.).

    Returns a list of *indices* into ``candidates`` (not the expressions
    themselves): SymPy caches structurally-equal expressions to one object,
    so two value-equal candidates share an ``id`` and cannot be told apart
    by identity — positions can.
    """
    registry = {}
    vectors = [_coeff_vector(c, registry) for c in candidates]
    keys = sorted({k for v in vectors for k in v}, key=str)
    if not keys:
        return []

    kept, rows = [], []
    for idx, vec in enumerate(vectors):
        row = [vec.get(k, sp.S.Zero) for k in keys]
        if all(x == 0 for x in row):
            continue
        trial = sp.Matrix(rows + [row])
        if trial.rank() > len(rows):
            rows.append(row)
            kept.append(idx)
    return kept


# --------------------------------------------------------------------------
# SuggestedTerm + build_lagrangian
# --------------------------------------------------------------------------

@dataclass
class SuggestedTerm:
    """One invariant Lagrangian term, coupling-free, with metadata.

    Attributes:
        expr: the invariant expression (hermitian-completed, no coupling).
        dim: mass dimension (2, 3 or 4).
        label: human-readable description.
        sector: ``"potential"``, ``"yukawa"`` or ``"kinetic"``.
        hc_added: whether ``+ h.c.`` completion was needed.
    """

    expr: sp.Expr
    dim: int
    label: str
    sector: str = "other"
    hc_added: bool = False
    meta: dict = dc_field(default_factory=dict)

    def __repr__(self):
        hc = " + h.c." if self.hc_added else ""
        return f"SuggestedTerm({self.label}{hc} [{self.sector}, dim {self.dim}])"

    def _repr_latex_(self):
        return f"$\\displaystyle {sp.latex(self.expr)}$"


def build_lagrangian(suggestions, coupling_prefix="c", start=1):
    """Attach a fresh coupling to each suggestion and fill a Lagrangian.

    Args:
        suggestions: iterable of :class:`SuggestedTerm`.
        coupling_prefix: base name for the generated couplings.
        start: first coupling index.

    Returns:
        ``(Lagrangian, [ExternalParameter, ...])`` — potential/Yukawa terms
        enter as ``−c·expr`` (feynlag stores ``L ⊃ −V``); kinetic-scaffold
        terms enter coupling-free.  The generated parameters carry
        ``unit_dim = 4 − dim``.
    """
    from .lagrangian import Lagrangian

    L = Lagrangian()
    params = []
    idx = start
    for term in suggestions:
        if term.sector == "kinetic" or (term.sector == "yukawa"
                                        and "current" in term.meta):
            L.add(term.expr, sector=term.sector, name=term.label)
            continue
        c = ExternalParameter(f"{coupling_prefix}{idx}", unit_dim=4 - term.dim)
        idx += 1
        params.append(c)
        L.add(-c.s * term.expr, sector=term.sector, name=term.label)
    return L, params


# --------------------------------------------------------------------------
# Gauge building blocks
# --------------------------------------------------------------------------

def _u1_groups(gauge_groups):
    return [g for g in gauge_groups if getattr(g, "abelian", False)]


def _nonabelian_signature(field, gauge_groups):
    """Rep dimension of ``field`` under each non-abelian group (1 = singlet)."""
    sig = []
    for g in gauge_groups:
        if getattr(g, "abelian", False):
            continue
        rep = field.reps.get(g)
        sig.append((g, g.rep_dim(rep) if rep is not None else 1))
    return tuple(sig)


def _charge_vector(field_or_expr_charges, u1s):
    return tuple(field_or_expr_charges.get(g, sp.S.Zero) for g in u1s)


def _is_su2_doublet(field, gauge_groups):
    """True iff ``field`` is a 2 of exactly one SU(2) and a singlet of every
    other non-abelian group."""
    su2s = [g for g in gauge_groups if isinstance(g, SU2)]
    doublet = [g for g in su2s if field.reps.get(g) == 2]
    if len(doublet) != 1:
        return False
    sig = dict(_nonabelian_signature(field, gauge_groups))
    return all(d == 1 for g, d in sig.items() if not isinstance(g, SU2))


@dataclass
class _Block:
    expr: sp.Expr
    dim: int
    charges: dict          # {U1 group: charge}
    label: str


def _scalar_blocks(scalars, gauge_groups):
    """Gauge-(non-abelian)-invariant building blocks with U(1) charge tags."""
    u1s = _u1_groups(gauge_groups)
    blocks = []

    # singlet scalars: the component and its conjugate (or just the real
    # component for a self-conjugate scalar)
    for s in scalars:
        sig = _nonabelian_signature(s, gauge_groups)
        if all(d == 1 for _, d in sig):
            c = s.components[0]
            q = {g: s.charge(g) for g in u1s}
            blocks.append(_Block(c, 1, dict(q), f"{c}"))
            if not s.self_conjugate:
                blocks.append(_Block(sp.conjugate(c), 1,
                                     {g: -v for g, v in q.items()},
                                     f"{c}†"))

    # matched non-trivial reps: A†B (contracts the whole rep) — dim 2
    for A, B in itertools.product(scalars, repeat=2):
        sigA = _nonabelian_signature(A, gauge_groups)
        sigB = _nonabelian_signature(B, gauge_groups)
        if sigA != sigB or all(d == 1 for _, d in sigA):
            continue
        expr = (dag(A) * B.mat)[0]
        q = {g: B.charge(g) - A.charge(g) for g in u1s}
        blocks.append(_Block(sp.expand(expr), 2, q, f"({A.name}†{B.name})"))

    # SU(2) ε-contraction A^T(iσ₂)B for doublet pairs — dim 2, charge sum
    for A, B in itertools.product(scalars, repeat=2):
        if not (_is_su2_doublet(A, gauge_groups)
                and _is_su2_doublet(B, gauge_groups)):
            continue
        a0, a1 = A.components[:2]
        b0, b1 = B.components[:2]
        expr = a0 * b1 - a1 * b0            # A^T (iσ₂) B
        q = {g: A.charge(g) + B.charge(g) for g in u1s}
        blocks.append(_Block(sp.expand(expr), 2, q,
                             f"({A.name}·{B.name})"))
    return blocks


def _charge_is_zero(charges, u1s):
    total = sum((charges.get(g, sp.S.Zero) for g in u1s), sp.S.Zero)
    return sp.simplify(total) == 0


# --------------------------------------------------------------------------
# suggest_potential
# --------------------------------------------------------------------------

def suggest_potential(scalars, gauge_groups, discrete_groups=(), max_dim=4,
                      min_dim=2, verify=True):
    """Enumerate scalar-potential invariants up to ``max_dim``.

    Args:
        scalars: the :class:`~feynlag.fields.Scalar` fields.
        gauge_groups: the gauge groups (U(1) charges filtered, non-abelian
            reps contracted into blocks).
        discrete_groups: discrete symmetries to impose (Reynolds-projected).
        max_dim / min_dim: mass-dimension window (default 2..4).
        verify: assert every emitted term against the invariance oracle.

    Returns:
        list of :class:`SuggestedTerm` (sector ``"potential"``), a minimal
        linearly-independent basis.
    """
    u1s = _u1_groups(gauge_groups)
    blocks = _scalar_blocks(scalars, gauge_groups)

    # enumerate charge-neutral, gauge-invariant monomials in the blocks
    candidates = []
    labels = []
    max_factors = max_dim  # each block has dim >= 1
    for n in range(1, max_factors + 1):
        for combo in itertools.combinations_with_replacement(blocks, n):
            dim = sum(b.dim for b in combo)
            if dim < min_dim or dim > max_dim:
                continue
            charges = {}
            for b in combo:
                for g, q in b.charges.items():
                    charges[g] = charges.get(g, sp.S.Zero) + q
            if not _charge_is_zero(charges, u1s):
                continue
            expr = sp.expand(sp.Mul(*[b.expr for b in combo]))
            if expr == 0:
                continue
            candidates.append((expr, dim, " ".join(b.label for b in combo)))

    # Reynolds-project onto the discrete-invariant subspace
    projected = []
    for expr, dim, label in candidates:
        p = reynolds_project(expr, discrete_groups) if discrete_groups else expr
        if p == 0:
            continue
        projected.append((sp.expand(p), dim, label))

    # hermitian completion
    completed = []
    for expr, dim, label in projected:
        if sp.expand(expr - sp.conjugate(expr)) == 0:
            completed.append(SuggestedTerm(expr, dim, label, "potential", False))
        else:
            completed.append(SuggestedTerm(sp.expand(expr + sp.conjugate(expr)),
                                           dim, label, "potential", True))

    # reduce to a linearly independent basis
    kept = _independent_subset([t.expr for t in completed])
    basis = [completed[i] for i in kept]

    if verify:
        _verify_terms(basis, scalars, gauge_groups, discrete_groups)
    return basis


# --------------------------------------------------------------------------
# suggest_yukawa
# --------------------------------------------------------------------------

def _nonabelian_reps(field, gauge_groups):
    """Ordered ``[(group, dim), ...]`` of the non-abelian reps a field
    actually carries (singlets omitted), in the field's own declaration
    order — the order :meth:`~feynlag.fields.Field.generators` uses to build
    the Kronecker product, hence the order of the flat component index."""
    out = []
    for g in field.reps:
        if getattr(g, "abelian", False):
            continue
        d = g.rep_dim(field.reps[g])
        if d > 1:
            out.append((g, d))
    return out


def _flat_index(field, local, gauge_groups):
    """Flat component index of ``field`` from per-group local indices.

    ``local`` maps a non-abelian group to its index within that group's rep;
    groups absent from ``local`` (singlets) contribute index 0.  Mixed-radix
    in the field's declaration order — matching the Kronecker product in
    :meth:`~feynlag.fields.Field.generators` / the flat ``components`` list.
    """
    idx = 0
    for g, d in _nonabelian_reps(field, gauge_groups):
        idx = idx * d + local.get(g, 0)
    return idx


def _spectator_iter(reps):
    """Cartesian product over a list of ``(group, dim)`` index ranges,
    yielding ``{group: local_index}`` dicts (one for the empty list)."""
    groups = [g for g, _ in reps]
    ranges = [range(d) for _, d in reps]
    for combo in itertools.product(*ranges):
        yield dict(zip(groups, combo))


def _scalar_leg_options(FL, FR, scalars, gauge_groups):
    """Yield ``(scalar_contraction_expr_builder, dim, charges, label)`` for
    every gauge-(non-abelian)-invariant way to sandwich ``F̄_L Γ F_R`` with a
    scalar leg (or none, for a bare mass).

    ``expr_builder(i, j)`` returns the coefficient-free Bilinear expression
    with flavor indices ``i`` (bar leg) and ``j`` (field leg).  Non-abelian
    indices other than the SU(2) one a doublet scalar saturates are summed
    diagonally (the ``δ``-contraction — e.g. the SU(3) color sum of a quark
    Yukawa, mirroring ``examples/sm_scalar_gauge.py``'s ``qcolor``).
    """
    from .dirac import diracPR

    u1s = _u1_groups(gauge_groups)
    sigL = _nonabelian_signature(FL, gauge_groups)
    sigR = _nonabelian_signature(FR, gauge_groups)

    def bare_charges(scalar_charges):
        return {g: -FL.charge(g) + FR.charge(g) + scalar_charges.get(g, 0)
                for g in u1s}

    # bare Dirac mass: matched non-abelian signatures, no scalar
    if sigL == sigR:
        def builder(i, j, FL=FL, FR=FR):
            return sum(Bilinear(FL.bar_components[k][i], diracPR,
                                FR.components[k][j])
                       for k in range(FL.dim))
        yield builder, 3, bare_charges({}), f"{FL.name}̄·{FR.name}"

    # doublet(F_L) × singlet-under-SU(2)(F_R): a doublet scalar saturates the
    # SU(2) index; every other non-abelian rep must match and is δ-summed.
    su2s = [g for g in gauge_groups if isinstance(g, SU2)]
    fl_su2 = [g for g in su2s if FL.reps.get(g) == 2]
    if len(fl_su2) == 1 and all(FR.reps.get(g) != 2 for g in su2s):
        su2 = fl_su2[0]
        # spectator reps = FL's non-abelian reps other than the saturated SU(2)
        spec = [(g, d) for g, d in _nonabelian_reps(FL, gauge_groups)
                if g is not su2]
        # they must equal FR's non-abelian reps (color etc.)
        if spec == _nonabelian_reps(FR, gauge_groups):
            for S in scalars:
                if not _is_su2_doublet(S, gauge_groups):
                    continue
                qS = {g: S.charge(g) for g in u1s}
                qSc = {g: -S.charge(g) for g in u1s}

                def builder_direct(i, j, FL=FL, FR=FR, S=S, su2=su2, spec=spec):
                    total = sp.S.Zero
                    for sp_idx in _spectator_iter(spec):
                        fr = FR.components[_flat_index(FR, sp_idx, gauge_groups)]
                        for a in (0, 1):
                            fl = FL.bar_components[_flat_index(
                                FL, {su2: a, **sp_idx}, gauge_groups)]
                            total += S.components[a] * Bilinear(
                                fl[i], diracPR, fr[j])
                    return total
                yield (builder_direct, 4, bare_charges(qS),
                       f"{FL.name}̄·{S.name} {FR.name}")

                def builder_tilde(i, j, FL=FL, FR=FR, S=S, su2=su2, spec=spec):
                    eps = {(0, 1): 1, (1, 0): -1}
                    total = sp.S.Zero
                    for sp_idx in _spectator_iter(spec):
                        fr = FR.components[_flat_index(FR, sp_idx, gauge_groups)]
                        for a, b in ((0, 1), (1, 0)):
                            fl = FL.bar_components[_flat_index(
                                FL, {su2: a, **sp_idx}, gauge_groups)]
                            total += (eps[(a, b)]
                                      * sp.conjugate(S.components[b])
                                      * Bilinear(fl[i], diracPR, fr[j]))
                    return total
                yield (builder_tilde, 4, bare_charges(qSc),
                       f"{FL.name}̄·{S.name}̃ {FR.name}")


def _majorana_leg_options(FL, scalars, gauge_groups):
    """Yield ``(builder, dim, charges, label)`` for the dim-5 Weinberg
    contraction of a same-chirality SU(2) doublet with two doublet scalars.

    The operator is ``(ε_ab F^a H^b)(ε_cd F^c H^d)`` with the Majorana
    ``C P_L`` between the two fermion legs — ``builder(i, j)`` returns the
    coefficient-free :class:`~feynlag.vertices.bilinear.MajoranaBilinear`
    expression.  Only doublet ``F`` and doublet scalars qualify (raises are
    avoided by the ``_is_su2_doublet`` guards); its U(1) charge is
    ``2 q_F + 2 q_S``.
    """
    from .dirac import diracC, diracPL

    u1s = _u1_groups(gauge_groups)
    if not _is_su2_doublet(FL, gauge_groups):
        return
    CPL = diracC * diracPL
    for S in scalars:
        if not _is_su2_doublet(S, gauge_groups):
            continue

        def builder(i, j, FL=FL, S=S):
            # (εX)_a = ε_ab X^b with ε_12 = 1, ε_21 = −1  →  (εX) = [X^2, −X^1]
            epsS = [S.components[1], -S.components[0]]
            total = sp.S.Zero
            for a in (0, 1):
                for c in (0, 1):
                    total += epsS[a] * epsS[c] * MajoranaBilinear(
                        FL.components[a][i], CPL, FL.components[c][j])
            return total

        charges = {g: 2 * FL.charge(g) + 2 * S.charge(g) for g in u1s}
        yield builder, 5, charges, f"({FL.name}·ε{S.name})² [Weinberg]"


def suggest_yukawa(fermions, scalars, gauge_groups, discrete_groups=(),
                   flavor_indices=None, max_dim=4, verify=True):
    """Enumerate Yukawa / bare-mass / Weinberg invariants over fermion pairs.

    Pairs run over ordered ``(F_L, F_R)`` of opposite chirality (two
    :class:`~feynlag.fields.WeylFermion`\\ s — vector-like fermions are two
    Weyls with identical reps, per ``CLAUDE.md``).  Each pair is offered
    every gauge-invariant scalar contraction (direct doublet, the H̃ pattern,
    or a bare mass); U(1) neutrality and discrete invariance filter, and the
    ``+ h.c.`` partner is generated (``Bilinear._eval_conjugate`` produces
    the correct P_L partner).

    With ``max_dim >= 5`` it also enumerates the **dim-5 Weinberg operator**
    ``(F ε H)(F ε H)`` — a *same-chirality* Majorana contraction of a doublet
    fermion with two doublet scalars, generating a Majorana neutrino mass after
    EWSB (:func:`~feynlag.vertices.bilinear.majorana_mass_matrix` +
    :func:`~feynlag.vacuum.diagonalize.diagonalize_takagi`).

    Args:
        flavor_indices: ``(i, j)`` symbols for the bar/field legs
            (default fresh ``i``, ``j``).
        max_dim: mass-dimension ceiling (default ``4``; ``5`` admits the
            Weinberg operator).

    Returns:
        list of :class:`SuggestedTerm` (sector ``"yukawa"``), a minimal basis.
    """
    if flavor_indices is None:
        flavor_indices = sp.symbols("i j", integer=True)
    i, j = flavor_indices

    left = [f for f in fermions if getattr(f, "chirality", None) == "L"]
    right = [f for f in fermions if getattr(f, "chirality", None) == "R"]
    u1s = _u1_groups(gauge_groups)

    # (F_L, F_R) options + (F, F) same-chirality Majorana options (dim ≥ 5)
    option_sets = [_scalar_leg_options(FL, FR, scalars, gauge_groups)
                   for FL, FR in itertools.product(left, right)]
    if max_dim >= 5:
        for F in left + right:
            option_sets.append(_majorana_leg_options(F, scalars, gauge_groups))

    candidates = []
    for options in option_sets:
        for builder, dim, charges, label in options:
            if dim > max_dim:
                continue
            if not _charge_is_zero(charges, u1s):
                continue
            expr = sp.expand(builder(i, j))
            if expr == 0:
                continue
            if discrete_groups:
                expr = reynolds_project(expr, discrete_groups)
                if expr == 0:
                    continue
            # h.c. completion (both bilinears know their conjugate)
            hc = sp.expand(sp.conjugate(expr))
            full = expand_bilinear(sp.expand(expr + hc))
            candidates.append(SuggestedTerm(full, dim, label, "yukawa", True))

    kept = _independent_subset([t.expr for t in candidates])
    basis = [candidates[i] for i in kept]

    if verify:
        _verify_terms(basis, list(fermions) + list(scalars), gauge_groups,
                      discrete_groups, max_dim=max_dim)
    return basis


# --------------------------------------------------------------------------
# suggest_kinetic (scaffolding)
# --------------------------------------------------------------------------

def suggest_kinetic(fields, flavor_index=None):
    """Kinetic/gauge scaffolding — no enumeration, just the canonical terms.

    Per :class:`~feynlag.fields.Scalar`: ``(D_μφ)†(D^μφ)`` (sector
    ``kinetic``).  Per :class:`~feynlag.fields.Fermion`: its gauge current
    ``fermion_gauge_current`` (sector ``yukawa``, matching how the examples
    add currents; the free kinetic ``iψ̄γ^μ∂_μψ`` carries no coupling and is
    handled at extraction time).  Gauge-boson self-couplings are built by
    ``cubic_couplings``/``quartic_couplings`` at extraction time, so no
    Lagrangian term is emitted for them here.

    Returns:
        list of :class:`SuggestedTerm`.
    """
    if flavor_index is None:
        flavor_index = sp.Symbol("i", integer=True)
    out = []
    for f in fields:
        if isinstance(f, Scalar):
            DH = Dmu(f)
            expr = sp.expand((dag(DH) * DH)[0])
            out.append(SuggestedTerm(expr, 4, f"(Dμ{f.name})†(Dμ{f.name})",
                                     "kinetic", False))
        elif isinstance(f, Fermion):
            if not f.reps:
                continue
            expr = fermion_gauge_current(f, flavor_index)
            if expr != 0:
                out.append(SuggestedTerm(expr, 4, f"{f.name} gauge current",
                                         "yukawa", False,
                                         meta={"current": True}))
    return out


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------

def _verify_terms(terms, fields, gauge_groups, discrete_groups, max_dim=4):
    """Assert every suggested term passes the full invariance battery."""
    for t in terms:
        for g in gauge_groups:
            ok, viol = check_gauge_invariance(t.expr, fields, g)
            assert ok, f"suggested term {t.label} fails gauge:{g.name}: {viol}"
        for g in discrete_groups:
            ok, viol = check_discrete_invariance(t.expr, g)
            assert ok, f"suggested term {t.label} fails discrete:{g.name}: {viol}"
        ok, worst = check_mass_dimension(t.expr, fields, max_dim=max_dim)
        assert ok, (f"suggested term {t.label} has dimension {worst} > "
                    f"{max_dim}")
        ok, _ = check_hermiticity(t.expr)
        assert ok, f"suggested term {t.label} is not hermitian"
