"""Symmetry and consistency checks for Lagrangian terms.

Strategy (see plan): **explicit component transformation**.

- Gauge groups: infinitesimal variation ``δφ = i α^a T^a φ`` with constant
  ``α^a``; the O(α) coefficient of every generator must vanish.  For terms
  written with covariant derivatives this global check is the correct
  tree-level invariance test.  Fermion legs (living inside ``Bilinear``
  atoms as ``Indexed`` components with a flavor index) get a parallel
  transform that preserves the flavor index — see :func:`_fermion_transform`.
- Discrete groups: substitute each finite generator map; the term must be
  unchanged after ``expand``.  Fermion legs get the same finite substitution
  as bosons, index-preserved via ``.replace()`` (no linearization needed —
  discrete transforms are finite, not infinitesimal) — see
  :func:`_fermion_transform_discrete`.
- Hermiticity: ``expand(L − L*) = 0`` (valid for bosonic sectors; fermion
  bilinears need Dirac-conjugation identities not yet implemented — see
  ``Model.check_invariance``, which skips sectors containing ``Bilinear``).
- Mass dimension: symbolic power counting (renormalizability: ≤ 4).  A
  fermion ``Bilinear`` contributes exactly 3 (two spin-½ legs).
"""

import sympy as sp

from .fields import Fermion
from .operators import PartialMu, expand_derivatives
from .vertices.bilinear import Bilinear

__all__ = ["gauge_variation", "check_gauge_invariance",
           "check_discrete_invariance", "check_hermiticity",
           "check_mass_dimension"]


def _transform_map(fields, group, alphas):
    """First-order gauge transformation of every component symbol, and of
    ``PartialMu(component)``.

    ``comp → comp + i α^a (T^a · comps)_i`` summed over generators.  Because
    the transformation uses a *global* (spacetime-constant) ``α``, ``∂_μ``
    commutes with it exactly: ``∂_μ(φ + iαTφ) = ∂_μφ + iαT(∂_μφ)``.  So
    ``PartialMu(comp)`` gets the identical linear substitution as its own
    key — this avoids ever having to differentiate through the ``PartialMu``
    wrapper (it has no ``fdiff`` rule, so SymPy's chain rule would otherwise
    leave an unevaluated, non-cancelling ``Subs(Derivative(...))`` residue).

    Fermion fields are skipped here (their components are ``IndexedBase``,
    never used as bare atoms — only as ``Indexed(base, flavor)`` inside
    ``Bilinear``\\ s); they go through :func:`_fermion_transform` instead.
    """
    sub = {}
    for field in fields:
        if isinstance(field, Fermion) or group not in field.reps:
            continue
        gens = field.generators(group)
        vec = field.mat
        dvec = sp.Matrix([[PartialMu(c)] for c in field.components])
        delta = sp.zeros(len(field.components), 1)
        ddelta = sp.zeros(len(field.components), 1)
        for alpha, T in zip(alphas, gens):
            delta += sp.I * alpha * (T * vec)
            ddelta += sp.I * alpha * (T * dvec)
        for i, comp in enumerate(field.components):
            sub[comp] = comp + delta[i]
            sub[PartialMu(comp)] = PartialMu(comp) + ddelta[i]
    return sub


def _apply_field_map(expr, sub, components=()):
    """Apply a component substitution, also inside ``PartialMu`` heads.

    ``expr`` is first Leibniz-expanded (:func:`~feynlag.operators.
    expand_derivatives`) so any compound ``PartialMu(product)`` is reduced to
    atomic ``PartialMu(component)`` terms matching ``sub``'s keys — ``Dmu``
    already only ever wraps atomic components, but this guards hand-written
    terms too.  A single ``xreplace`` then handles both plain components and
    their ``PartialMu`` counterparts (``sub`` carries both); it also
    traverses inside ``conjugate(...)`` so conjugated components transform
    consistently.
    """
    if not sub:
        return expr
    if components:
        expr = expand_derivatives(expr, components)
    return expr.xreplace(sub)


def _fermion_transform(expr, fields, group, alphas):
    """First-order gauge transformation of fermion legs inside ``Bilinear``.

    Fermion components are ``IndexedBase``-typed and only ever appear as
    ``Indexed(base, flavor_index)`` — a different substitution mechanism
    from the plain-``Symbol`` bosonic case, since the flavor index on the
    matched node must be preserved symbolically (a fixed ``xreplace`` dict
    can't do that; :meth:`~sympy.Basic.replace` with a predicate + builder
    function can, because it hands the matched node itself to the builder).

    Transformation (T = generator matrix, dimension = field.dim, i.e. the
    gauge index, NOT the flavor index):
    - ``δψ_r[i] = i α Σ_c T[r,c] ψ_c[i]``
    - ``δψ̄_r[i] = −i α Σ_c ψ̄_c[i] T[c,r]`` (dual representation)
    """
    replacements = {}
    for field in fields:
        if not isinstance(field, Fermion) or group not in field.reps:
            continue
        T_list = field.generators(group)
        for r, base in enumerate(field.components):
            replacements[base] = (field.components, r, T_list, 1)
        for r, base in enumerate(field.bar_components):
            replacements[base] = (field.bar_components, r, T_list, -1)
    if not replacements:
        return expr

    def _sub(node):
        comp_list, r, T_list, sign = replacements[node.base]
        idx = node.indices
        delta = sp.S.Zero
        for alpha, T in zip(alphas, T_list):
            if sign > 0:
                delta += sp.I * alpha * sum(
                    T[r, c] * comp_list[c][idx] for c in range(len(comp_list))
                    if T[r, c] != 0)
            else:
                delta += -sp.I * alpha * sum(
                    T[c, r] * comp_list[c][idx] for c in range(len(comp_list))
                    if T[c, r] != 0)
        return node + delta

    return expr.replace(
        lambda x: isinstance(x, sp.Indexed) and x.base in replacements, _sub)


def _split_indexed_term(term):
    """Split ``coeff * Indexed(...)`` into ``(coeff, Indexed(...))``.

    Unlike :meth:`~sympy.core.expr.Expr.as_coeff_Mul` (which, with its
    default ``rational=True``, only pulls out a *Rational* coefficient and
    leaves any Dummy/Symbol factors — such as the infinitesimal ``alpha``
    from :func:`_fermion_transform` — bundled into the "atom" side), this
    finds the single ``Indexed`` factor explicitly and treats everything
    else (numbers, ``I``, ``alpha``, generator matrix entries, …) as the
    coefficient.
    """
    if isinstance(term, sp.Indexed):
        return sp.S.One, term
    factors = term.as_ordered_factors()
    indexed = [f for f in factors if isinstance(f, sp.Indexed)]
    if len(indexed) != 1:
        raise ValueError(f"expected exactly one Indexed factor in {term!r}, "
                         f"found {len(indexed)}")
    atom = indexed[0]
    coeff = sp.Mul(*[f for f in factors if f is not atom])
    return coeff, atom


def _expand_bilinear(expr):
    """Distribute ``Bilinear(bar, gamma, field)`` over ``Add``-valued
    ``bar``/``field`` legs.

    ``Bilinear`` is linear in each of those two slots (not in ``gamma``) but
    is an opaque custom ``Function``, so ``sp.expand()`` alone won't
    distribute it — the same "teach SymPy about a custom operator's
    linearity" need as :func:`~feynlag.operators.D_linear` for
    ``PartialMu``, here applied to two slots instead of one.
    """
    def _one(bar, gamma, field):
        bar_terms = sp.Add.make_args(sp.expand(bar))
        field_terms = sp.Add.make_args(sp.expand(field))
        total = sp.S.Zero
        for bt in bar_terms:
            bc, ba = _split_indexed_term(bt)
            for ft in field_terms:
                fc, fa = _split_indexed_term(ft)
                total += bc * fc * Bilinear(ba, gamma, fa)
        return total

    return expr.replace(Bilinear, _one)


def _fermion_transform_discrete(expr, group, gen_index):
    """Finite discrete transformation of fermion legs inside ``Bilinear``.

    Discrete transformations are exact substitutions (no ``α``-linearization
    like the gauge case needs), so the builder returns the fully transformed
    value directly: ``Σ_k Mat[i,k] · comp_at_slot[k][idx]``, using
    ``group.fermion_generator_data()`` (see that method's docstring for the
    field-vs-bar ``M`` vs ``(M⁻¹)ᵀ`` derivation).
    """
    data = group.fermion_generator_data()[gen_index]
    if not data:
        return expr

    def _sub(node):
        comp_at_slot, i, Mat = data[node.base]
        idx = node.indices
        return sum(Mat[i, k] * comp_at_slot[k][idx]
                  for k in range(len(comp_at_slot)) if Mat[i, k] != 0)

    return expr.replace(
        lambda x: isinstance(x, sp.Indexed) and x.base in data, _sub)


def gauge_variation(term, fields, group):
    """O(α) variation coefficients of ``term`` under ``group``, per generator.

    Returns a list (length = number of generators) of expressions; all must
    vanish for invariance.
    """
    n = group.n_generators
    alphas = [sp.Dummy(f"alpha_{group.name}_{a}", real=True) for a in range(n)]
    sub = _transform_map(fields, group, alphas)
    has_fermion_content = term.has(Bilinear)
    if not sub and not has_fermion_content:
        return [sp.S.Zero] * n

    components = [c for f in fields for c in getattr(f, "components", ())
                 if not isinstance(f, Fermion)]
    transformed = (_apply_field_map(term, sub, components=components)
                  if sub else term)
    if has_fermion_content:
        transformed = _fermion_transform(transformed, fields, group, alphas)
        transformed = _expand_bilinear(transformed)

    delta = sp.expand(transformed - term)
    coeffs = []
    zero_all = {a: 0 for a in alphas}
    for alpha in alphas:
        coeffs.append(sp.expand(sp.diff(delta, alpha).subs(zero_all)))
    return coeffs


def check_gauge_invariance(term, fields, group):
    """Whether ``term`` is invariant under ``group`` at first order.

    Returns:
        ``(ok: bool, violations: list[(generator_index, coefficient)])``.
    """
    violations = []
    for a, coeff in enumerate(gauge_variation(term, fields, group)):
        if sp.expand(coeff) != 0:
            violations.append((a, coeff))
    return (not violations, violations)


def check_discrete_invariance(term, group):
    """Whether ``term`` is invariant under every generator of ``group``.

    Returns:
        ``(ok: bool, violations: list[(generator_index, residual)])``.
    """
    violations = []
    components = group.components()
    has_fermion_content = term.has(Bilinear)
    for gen_index, sub in enumerate(group.generator_maps()):
        transformed = _apply_field_map(term, sub, components=components)
        if has_fermion_content:
            transformed = _fermion_transform_discrete(transformed, group,
                                                       gen_index)
            transformed = _expand_bilinear(transformed)
        residual = sp.expand(transformed - term)
        if residual != 0:
            residual = sp.simplify(residual)  # ω-phases need simplification
        if residual != 0:
            violations.append((gen_index, residual))
    return (not violations, violations)


def check_hermiticity(expr):
    """Whether a bosonic Lagrangian (sector) is hermitian: ``L = L*``.

    Returns:
        ``(ok: bool, residual)``.
    """
    residual = sp.expand(expr - sp.conjugate(expr))
    if residual != 0:
        residual = sp.simplify(residual)
    return (residual == 0, residual)


def check_mass_dimension(term, fields, parameters=None, max_dim=4):
    """Whether every monomial of ``term`` has mass dimension ≤ ``max_dim``.

    Field dimensions from spin (scalar/vector: 1, fermion: 3/2); parameter
    dimensions from ``Parameter.unit_dim``.  ``PartialMu`` counts as +1;
    a fermion ``Bilinear`` counts as +3 (two spin-½ legs).

    Returns:
        ``(ok: bool, worst_dim)``.
    """
    dims = {}
    for field in fields:
        d = sp.Rational(3, 2) if getattr(field, "spin", 0) == sp.Rational(1, 2) else 1
        for comp in field.components:
            dims[comp] = d
        for comp, (vev, re, im) in getattr(field, "vev_expansions", {}).items():
            dims[vev] = 1
            dims[re] = 1
            if im is not None:
                dims[im] = 1
    if parameters is not None:
        for p in parameters:
            dims[p.symbol] = p.unit_dim

    u = sp.Symbol("__mass_unit__", positive=True)
    # ∂_μ adds one mass dimension
    term = term.replace(PartialMu, lambda arg: u * arg)
    # a fermion bilinear is two spin-1/2 legs (3/2 + 3/2 = 3); its internal
    # Dirac/flavor structure is irrelevant to mass dimension, so collapse
    # the whole opaque node to a bare power of u.
    term = term.replace(Bilinear, lambda bar, gamma, field: u ** 3)
    subs = {s: u ** d for s, d in dims.items()}
    powered = sp.expand(term.xreplace(subs))

    worst = 0
    for monomial in powered.as_ordered_terms():
        poly = sp.together(monomial)
        found = (sp.degree(sp.numer(poly), u) - sp.degree(sp.denom(poly), u)
                 if poly.has(u) else 0)
        worst = max(worst, found)
    return (worst <= max_dim, worst)
