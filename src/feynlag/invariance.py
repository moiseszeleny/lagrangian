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
from .vertices.bilinear import Bilinear, MajoranaBilinear, expand_bilinear

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

    Each generator's coefficient is computed from an *independent* single-α
    transformation (the α-vector is zero except for the generator under test).
    This is mathematically identical to expanding the full multi-α variation
    and picking out one α — the O(α²) cross-terms between generators vanish
    when the other α's are set to zero anyway — but it keeps every intermediate
    expression small: only one generator matrix enters at a time.  That matters
    for large / radical-valued representations (a general SU(N) irrep built in
    the Gelfand–Tsetlin basis), where the all-generators-at-once expansion of a
    covariant-derivative kinetic term is intractable.
    """
    n = group.n_generators
    has_fermion_content = term.has(Bilinear) or term.has(MajoranaBilinear)

    probe = [sp.Dummy() for _ in range(n)]
    if not _transform_map(fields, group, probe) and not has_fermion_content:
        return [sp.S.Zero] * n

    components = [c for f in fields for c in getattr(f, "components", ())
                 if not isinstance(f, Fermion)]
    # Leibniz-expand derivatives once; the per-generator xreplace reuses it.
    base = expand_derivatives(term, components) if components else term

    coeffs = []
    for a in range(n):
        alpha = sp.Dummy(f"alpha_{group.name}_{a}", real=True)
        alphas = [sp.S.Zero] * n
        alphas[a] = alpha
        sub = _transform_map(fields, group, alphas)
        transformed = base.xreplace(sub) if sub else base
        if has_fermion_content:
            transformed = _fermion_transform(transformed, fields, group, alphas)
            transformed = expand_bilinear(transformed)
        delta = transformed - base
        coeffs.append(sp.expand(sp.diff(delta, alpha).subs({alpha: 0})))
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
    has_fermion_content = term.has(Bilinear) or term.has(MajoranaBilinear)
    for gen_index, sub in enumerate(group.generator_maps()):
        transformed = _apply_field_map(term, sub, components=components)
        if has_fermion_content:
            transformed = _fermion_transform_discrete(transformed, group,
                                                       gen_index)
            transformed = expand_bilinear(transformed)
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
    subs = {s: u ** d for s, d in dims.items()}

    def _replace(t):
        # ∂_μ adds one mass dimension; a fermion bilinear is two spin-½ legs
        # (3/2 + 3/2 = 3), a Majorana bilinear ψᵀCΓψ likewise — the internal
        # Dirac/flavor structure carries no mass dimension, so collapse the
        # opaque node to a bare power of u.
        t = t.replace(PartialMu, lambda arg: u * arg)
        t = t.replace(Bilinear, lambda bar, gamma, field: u ** 3)
        t = t.replace(MajoranaBilinear, lambda f1, gamma, f2: u ** 3)
        return t.xreplace(subs)

    # Process each additive term *independently*: collapsing every bilinear to
    # the same ``u**3`` erases the distinct legs, so two different-leg terms
    # whose boson coefficients differ in sign (e.g. the (εH)(εH) Weinberg
    # structure, H₀²−2H₀G⁺+G⁺²) would spuriously cancel to 0 if expanded
    # together after substitution.  Per-term counting avoids that.
    worst = 0
    top_terms = sp.expand(term).as_ordered_terms()
    for top in top_terms:
        powered = sp.expand(_replace(top))
        for monomial in powered.as_ordered_terms():
            poly = sp.together(monomial)
            found = (sp.degree(sp.numer(poly), u) - sp.degree(sp.denom(poly), u)
                     if poly.has(u) else 0)
            worst = max(worst, found)
    return (worst <= max_dim, worst)
