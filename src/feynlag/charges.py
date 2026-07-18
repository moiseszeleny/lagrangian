"""Electric-charge registry and charge-based vertex checks.

Physical-basis fields (``h``, ``Z``, ``Wp``, …) are ad-hoc SymPy symbols the
user creates during diagonalization; they carry no quantum numbers.  This
module supplies the two things the remaining Phase-B checks need:

- a :class:`ChargeRegistry` — the user's declared ``{physical symbol: electric
  charge}`` map, auto-completed with conjugate/bar partners — which
  :func:`check_charge_conservation` and :func:`check_hermiticity_pairing`
  consult for every vertex leg;
- an independent **derivation** of the electric-charge operator straight from
  the vacuum (:func:`derive_charge_operator`): electric charge is the
  combination of diagonal (Cartan) generators and U(1) charges that annihilates
  every VEV, so it falls out of a null-space solve — no hard-coded ``Q=T3+Y``.
  :func:`check_charge_consistency` cross-checks the declared map against it.

The declared map is the authority for the vertex checks; the vacuum derivation
is the cross-check that catches a charge assignment inconsistent with the group
theory (and confirms every physical field is a genuine charge eigenstate).
"""

import math
from collections import Counter

import sympy as sp

from .fields import Fermion, _BAR_PARTNER
from .operators import momentum

__all__ = [
    "ChargeRegistry",
    "derive_charge_operator", "physical_charges",
    "check_charge_consistency", "ChargeConsistencyReport",
    "check_charge_conservation", "ChargeConservationReport",
    "check_hermiticity_pairing", "HermiticityPairingReport",
]


# --------------------------------------------------------------- registry

class ChargeRegistry:
    """Declared ``{leg: electric charge}`` map, auto-completed with partners.

    ``declared`` keys are physical field symbols (``Wp``, ``h``, …) or fermion
    component ``IndexedBase``\\ s (``eL``, ``nuL``, …).  Every fermion field's
    Dirac-adjoint partner is registered with the opposite charge, and any
    ``conjugate_map`` partner (``Gm`` for ``Gp``) gets the negated charge.
    """

    def __init__(self, declared, conjugate_map=None):
        self.q = {}
        for leg, value in declared.items():
            key = self._key(leg)
            self.q[key] = sp.sympify(value)
            # a fermion component's bar carries the opposite charge
            partner = _BAR_PARTNER.get(key)
            if partner is not None:
                self.q[partner] = -self.q[key]
        if conjugate_map:
            for conj_node, partner in conjugate_map.items():
                base = conj_node.args[0] if conj_node.args else conj_node
                bkey = self._key(base)
                if bkey in self.q:
                    self.q[self._key(partner)] = -self.q[bkey]

    @staticmethod
    def _key(leg):
        if isinstance(leg, sp.Indexed):
            return leg.base
        return leg

    def known(self, leg):
        return self._key(leg) in self.q

    def charge_of(self, leg):
        key = self._key(leg)
        if key not in self.q:
            raise KeyError(
                f"no electric charge registered for {leg} (base {key}); "
                f"declare it in the charge map passed to ChargeRegistry")
        return self.q[key]


# --------------------------------------------------- charge-operator derivation

def _fundamental_label(group):
    """Rep label of the defining (fundamental) rep of an SU(N) group.

    feynlag labels reps by dimension (SU(2): 1/2/3, SU(3): 1/3/8), and
    ``n_generators = N²−1``, so the fundamental dimension ``N = √(n+1)`` is also
    its label.
    """
    return math.isqrt(group.n_generators + 1)


def _cartan_indices(group):
    """Indices of the diagonal (Cartan) generators in the fundamental rep."""
    gens = group.generators(_fundamental_label(group))
    return [i for i, M in enumerate(gens) if sp.Matrix(M).is_diagonal()]


def _vevd_components(model):
    """``[(field, component_index)]`` for every VEV'd scalar component."""
    out = []
    for f in model.fields:
        for comp in getattr(f, "vev_expansions", {}):
            out.append((f, f.components.index(comp)))
    return out


def _operator_on(group, cartan_idx, field):
    """Matrix of a candidate diagonal operator on ``field``'s component space."""
    if group not in field.reps:
        return sp.zeros(field.dim, field.dim)
    if cartan_idx is None:                       # a U(1) charge
        return field.charge(group) * sp.eye(field.dim)
    return sp.Matrix(field.generators(group)[cartan_idx])


def derive_charge_operator(model):
    """Electric-charge operator as the unbroken vacuum-annihilating combination.

    Returns ``{(group, cartan_idx_or_None): coefficient}`` (unnormalised — the
    null space fixes the operator only up to an overall scale).

    Raises:
        ValueError: no VEVs registered; the vacuum breaks every diagonal
            generator (no unbroken charge); or more than one independent
            unbroken U(1) remains (electric charge genuinely ambiguous).
    """
    vevd = _vevd_components(model)
    if not vevd:
        raise ValueError(
            "no VEVs registered; cannot derive the electric-charge operator "
            "(call expand_vev on the symmetry-breaking scalar first)")

    # Candidate diagonal operators, excluding any group under which no VEV'd
    # scalar transforms (unbroken by construction — colour, spectator U(1)s —
    # and so not part of electric charge by convention).
    candidates = []
    for g in model.gauge_groups:
        if not any(g in f.reps for f, _ in vevd):
            continue
        if g.abelian:
            candidates.append((g, None))
        else:
            candidates.extend((g, idx) for idx in _cartan_indices(g))
    if not candidates:
        raise ValueError("no diagonal generator acts on any VEV; "
                         "cannot define electric charge")

    # Constraint rows: Q·e_k = 0 (column k of Q must vanish) for each VEV.
    rows = []
    for f, k in vevd:
        cols = [_operator_on(g, idx, f)[:, k] for g, idx in candidates]
        for r in range(f.dim):
            rows.append([col[r] for col in cols])

    null = sp.Matrix(rows).nullspace()
    if len(null) == 0:
        raise ValueError(
            "the vacuum breaks every diagonal generator: no unbroken electric "
            "charge exists for this VEV configuration")
    if len(null) > 1:
        raise ValueError(
            f"electric charge is ambiguous: {len(null)} independent unbroken "
            f"U(1)s remain. Reduce the spectator gauge content or break the "
            f"tie with an explicit charge operator")
    vec = null[0]
    return {cand: vec[a] for a, cand in enumerate(candidates)}


def _field_charge_matrix(field, coeffs):
    Q = sp.zeros(field.dim, field.dim)
    for (g, idx), c in coeffs.items():
        if c == 0:
            continue
        Q += c * _operator_on(g, idx, field)
    return sp.Matrix(Q)


def _eigen_charge(Q, k):
    """Charge of component ``k`` if it is a charge eigenstate, else ``None``."""
    n = Q.shape[0]
    for j in range(n):
        if j != k and (sp.simplify(Q[k, j]) != 0 or sp.simplify(Q[j, k]) != 0):
            return None
    return sp.simplify(Q[k, k])


def physical_charges(model, coeffs):
    """Charge of every weak component and physical (rotated) field.

    Charges are **unnormalised** (the same overall scale as ``coeffs``); the
    consistency check fixes the scale against the declared map.
    """
    charges = {}
    field_Q = {}
    comp_owner = {}

    for f in model.fields:
        Q = _field_charge_matrix(f, coeffs)
        field_Q[id(f)] = Q
        for k, comp in enumerate(f.components):
            comp_owner[comp] = (f, k)
            q = _eigen_charge(Q, k)
            if q is not None:
                charges[comp] = q
        # VEV fluctuation symbols are neutral by definition of the vacuum.
        for comp, (vev, re, im) in getattr(f, "vev_expansions", {}).items():
            k = f.components.index(comp)
            qk = _eigen_charge(Q, k)
            if qk is not None and qk != 0:
                raise ValueError(
                    f"VEV in charged component {comp} (Q={qk}): the vacuum "
                    f"breaks electric charge")
            charges[re] = sp.S.Zero
            if im is not None:
                charges[im] = sp.S.Zero

    def q_entry(oa, ob):
        if oa in comp_owner and ob in comp_owner:
            fa, ka = comp_owner[oa]
            fb, kb = comp_owner[ob]
            return field_Q[id(fa)][ka, kb] if fa is fb else sp.S.Zero
        if oa is ob and oa in charges:
            return charges[oa]
        return sp.S.Zero

    for rot in model.rotations:
        n = len(rot.old_fields)
        Qb = sp.Matrix(n, n, lambda a, b: q_entry(rot.old_fields[a],
                                                  rot.old_fields[b]))
        Qnew = sp.Matrix(rot.matrix) * Qb * sp.Matrix(rot.inverse)
        for i, nf in enumerate(rot.new_fields):
            offdiag = any(sp.simplify(Qnew[i, j]) != 0
                          for j in range(n) if j != i)
            if offdiag:
                raise ValueError(
                    f"physical field {nf} is not an electric-charge eigenstate "
                    f"(rotation mixes states of different charge)")
            charges[nf] = sp.simplify(Qnew[i, i])
    return charges


# --------------------------------------------------------- consistency report

class ChargeConsistencyReport:
    """Result of :func:`check_charge_consistency`."""

    def __init__(self, coeffs, scale, mismatches):
        self.coeffs = coeffs
        self.scale = scale
        #: list of ``(field, declared, derived)``
        self.mismatches = mismatches

    @property
    def ok(self):
        return not self.mismatches

    def raise_on_failure(self):
        if not self.ok:
            lines = "\n".join(f"  {s}: declared {d}, derived {v}"
                              for s, d, v in self.mismatches)
            raise ValueError(
                f"declared charges disagree with the vacuum-derived operator:\n"
                f"{lines}")
        return self

    def __repr__(self):
        status = "consistent" if self.ok else f"{len(self.mismatches)} mismatch"
        return f"ChargeConsistencyReport({status})"


def check_charge_consistency(model, registry):
    """Cross-check the declared charges against the vacuum-derived operator."""
    coeffs = derive_charge_operator(model)
    derived = physical_charges(model, coeffs)

    # Charges are defined only up to the overall null-space scale.  Fix it from
    # the declared map by the *most common* declared/derived ratio, so a single
    # mistuned declaration is flagged rather than shifting the scale and
    # blaming every other field.
    pairs = []
    ratios = []
    for sym, dq in derived.items():
        key = registry._key(sym)
        if key in registry.q:
            declared = registry.q[key]
            pairs.append((sym, declared, dq))
            if dq != 0:
                ratios.append(sp.nsimplify(declared / dq))
    if ratios:
        scale = Counter(ratios).most_common(1)[0][0]
    else:
        scale = sp.S.One

    mismatches = []
    for sym, declared, dq in pairs:
        if sp.simplify(declared - scale * dq) != 0:
            mismatches.append((sym, declared, sp.simplify(scale * dq)))
    return ChargeConsistencyReport(coeffs, scale, mismatches)


# -------------------------------------------------------- conservation report

class ChargeConservationReport:
    """Result of :func:`check_charge_conservation`."""

    def __init__(self, n_checked, failures):
        self.n_checked = n_checked
        #: list of ``(legs, total_charge)``
        self.failures = failures

    @property
    def ok(self):
        return not self.failures

    def raise_on_failure(self):
        if not self.ok:
            lines = "\n".join(f"  {legs}: ΣQ = {q}" for legs, q in self.failures)
            raise ValueError(f"charge is not conserved at:\n{lines}")
        return self

    def __repr__(self):
        status = "ok" if self.ok else f"{len(self.failures)} violation"
        return f"ChargeConservationReport({self.n_checked} vertices, {status})"


def check_charge_conservation(registry, bosonic_vertices=(), fermion_table=None):
    """Every vertex must have vanishing total electric charge.

    Args:
        registry: a :class:`ChargeRegistry`.
        bosonic_vertices: iterable of
            :class:`~feynlag.vertices.vertex.Vertex`.
        fermion_table: optional ``extract_fermion_vertices`` output
            (``{(bar, gamma, field): {n: {boson-tuple: coeff}}}``).
    """
    failures = []
    n = 0
    for v in bosonic_vertices:
        n += 1
        total = sum((registry.charge_of(p) for p in v.particles), sp.S.Zero)
        if sp.simplify(total) != 0:
            failures.append((tuple(v.particles), sp.simplify(total)))

    if fermion_table:
        for key, by_n in fermion_table.items():
            if isinstance(key[0], tuple):        # FFFF: two-bilinear key
                fermion_legs = tuple(leg for (bar, gamma, field) in key
                                     for leg in (bar, field))
            else:                                 # FFS/FFV: one-bilinear key
                bar, gamma, field = key
                fermion_legs = (bar, field)
            base = sum((registry.charge_of(f) for f in fermion_legs), sp.S.Zero)
            for boson_dict in by_n.values():
                for boson_tuple in boson_dict:
                    n += 1
                    total = base + sum((registry.charge_of(b)
                                        for b in boson_tuple), sp.S.Zero)
                    if sp.simplify(total) != 0:
                        legs = fermion_legs + tuple(boson_tuple)
                        failures.append((legs, sp.simplify(total)))
    return ChargeConservationReport(n, failures)


# --------------------------------------------------- hermiticity-pairing report

class HermiticityPairingReport:
    """Result of :func:`check_hermiticity_pairing`."""

    def __init__(self, n_checked, failures, skipped=()):
        self.n_checked = n_checked
        #: list of ``(legs, reason)``
        self.failures = list(failures)
        #: list of ``(legs, reason)`` for vertices a check could not evaluate
        self.skipped = list(skipped)

    @property
    def ok(self):
        return not self.failures

    def raise_on_failure(self):
        if not self.ok:
            lines = "\n".join(f"  {legs}: {why}" for legs, why in self.failures)
            raise ValueError(f"hermiticity pairing failed:\n{lines}")
        return self

    def __repr__(self):
        status = "ok" if self.ok else f"{len(self.failures)} failure"
        skip = f", {len(self.skipped)} skipped" if self.skipped else ""
        return f"HermiticityPairingReport({self.n_checked} vertices, {status}{skip})"


def _momentum_degree(coupling):
    """Highest power of momentum tags in any term (the derivative count)."""
    expr = sp.expand(coupling)
    terms = expr.as_ordered_terms() if expr.is_Add else [expr]
    deg = 0
    for t in terms:
        d = 0
        for f in sp.Mul.make_args(t):
            if isinstance(f, sp.Pow) and f.base.func is momentum:
                d += int(f.exp)
            elif f.func is momentum:
                d += 1
        deg = max(deg, d)
    return deg


def _expected_partner_coupling(coupling, conj_leg):
    """Coupling the hermitian-conjugate vertex must carry.

    For ``L ⊃ c·O + c*·O†`` with ``O`` carrying ``d`` derivatives, the
    momentum-space rules (``i × coefficient``, ``∂→ip``) give
    ``rule(O†) = (−1)^{1+d} · [rule(O)]*`` with every momentum tag ``p(φ)``
    relabelled to its conjugate leg ``p(φ̄)`` (``p`` itself is real).
    """
    d = _momentum_degree(coupling)
    c = sp.conjugate(coupling)
    # p is real: strip the conjugate wrapper off momentum tags (no relabel yet),
    c = c.replace(
        lambda e: e.func is sp.conjugate and e.args and e.args[0].func is momentum,
        lambda e: e.args[0])
    # then relabel every momentum tag to its conjugate leg exactly once.
    c = c.replace(lambda e: e.func is momentum,
                  lambda e: momentum(conj_leg(e.args[0])))
    return sp.expand((-1) ** (1 + d) * c)


def check_hermiticity_pairing(bosonic_vertices=(), conjugates=None,
                              fermion_table=None):
    """Every vertex must pair with its hermitian conjugate.

    A term and its ``+ h.c.`` produce two vertices whose legs are mutual
    conjugates and whose couplings satisfy the momentum-space hermiticity
    relation (:func:`_expected_partner_coupling`).  A missing partner or a
    coupling that is not the conjugate is a hermiticity violation — the vertex
    analogue of the Lagrangian-level check, catching a dropped ``h.c.``.

    Args:
        bosonic_vertices: iterable of :class:`~feynlag.vertices.vertex.Vertex`.
        conjugates: full antiparticle pairing ``{field: partner}`` for every
            non-self-conjugate physical field (both directions, e.g.
            ``{Gp: Gm, Gm: Gp, Wp: Wm, Wm: Wp}``); fields absent from it are
            treated as self-conjugate. This is the same information as
            ``UFOParticle.antisymbol`` and is required because charged physical
            fields (``W±``) come from rotations, not ``conjugate_pair``.
        fermion_table: optional ``extract_fermion_vertices`` output.
    """
    from .dirac import dirac_conjugate

    conjugates = conjugates or {}

    def conj_leg(sym):
        return conjugates.get(sym, sym)

    failures = []
    skipped = []
    n = 0

    # --- bosonic ---
    table = {tuple(v.particles): sp.expand(v.coupling) for v in bosonic_vertices}
    seen = set()
    for v in bosonic_vertices:
        legs = tuple(v.particles)
        if legs in seen:
            continue
        n += 1
        conj_legs = tuple(sorted((conj_leg(p) for p in legs),
                                 key=sp.default_sort_key))
        expected = _expected_partner_coupling(v.coupling, conj_leg)
        if conj_legs not in table:
            failures.append((legs, f"no conjugate partner {conj_legs}"))
            seen.add(legs)
            continue
        seen.add(legs)
        seen.add(conj_legs)
        if sp.simplify(sp.expand(table[conj_legs]) - expected) != 0:
            failures.append((legs, f"coupling is not the conjugate of {conj_legs}"))

    # --- fermionic ---
    if fermion_table:
        keys = set(fermion_table)
        seen_f = set()

        def conj_subkey(bar, gamma, field):
            """``(ψ̄₁Γψ₂)† = ψ̄₂Γ̄ψ₁`` at the level of one bilinear subkey."""
            gbar = dirac_conjugate(gamma)      # may raise NotImplementedError
            return (_BAR_PARTNER[field.base][field.indices], gbar,
                    _BAR_PARTNER[bar.base][bar.indices])

        for key, by_n in fermion_table.items():
            if key in seen_f:
                continue
            n += 1
            try:
                if isinstance(key[0], tuple):    # FFFF: two-bilinear key
                    subpartners = [conj_subkey(*sub) for sub in key]
                    subpartners.sort(
                        key=lambda t: sp.default_sort_key(sp.Tuple(*t)))
                    partner = tuple(subpartners)
                else:                             # FFS/FFV: one-bilinear key
                    partner = conj_subkey(*key)
            except NotImplementedError as exc:
                # exotic Γ we can't conjugate — report as skipped, not a crash
                skipped.append((key, str(exc)))
                continue
            seen_f.add(key)
            if partner not in keys:
                failures.append((key, "no conjugate bilinear partner"))
                continue
            seen_f.add(partner)

    return HermiticityPairingReport(n, failures, skipped)
