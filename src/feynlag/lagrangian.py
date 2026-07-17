"""Lagrangian assembly and the top-level Model object.

The :class:`Model` is a **lazy pipeline**: nothing is solved or diagonalized
at construction; every stage is a method/cached property invoked on demand
(this fixes the compute-at-import flaw of the DLRSM1 reference model).

Phase 1 provides declaration + invariance checking; the EWSB/tadpole/mass
/vertex stages are added by the vacuum and vertices packages.
"""

from dataclasses import dataclass, field as dc_field

import sympy as sp

from .fields import Scalar
from .groups.base import GaugeGroup
from .groups.discrete import DiscreteSymmetry
from .invariance import (
    check_discrete_invariance, check_gauge_invariance, check_hermiticity,
    check_mass_dimension,
)
from .operators import PartialMu, to_momentum_space
from .parameters import ParameterSet
from .vacuum.ewsb import Vacuum
from .vacuum.masses import (
    charged_mass_matrix, gauge_mass_matrix, scalar_mass_matrix,
)
from .vacuum.tadpoles import extract_tadpoles, solve_tadpoles
from .vertices.extract import extract_interaction_coefficients, feynman_rule
from .vertices.vertex import Vertex

__all__ = ["LagrangianTerm", "Lagrangian", "Model", "InvarianceReport"]

SECTORS = ("kinetic", "gauge", "potential", "yukawa", "other")


@dataclass
class LagrangianTerm:
    """One term of the Lagrangian, tagged with its sector."""

    expr: sp.Expr
    sector: str = "other"
    name: str = ""

    def __post_init__(self):
        if self.sector not in SECTORS:
            raise ValueError(f"sector must be one of {SECTORS}, "
                             f"got {self.sector!r}")
        self.expr = sp.sympify(self.expr)


class Lagrangian:
    """Sector-tagged collection of Lagrangian terms."""

    def __init__(self):
        self.terms = []

    def add(self, expr, sector="other", name=""):
        """Add a term (a SymPy expression in field components)."""
        self.terms.append(LagrangianTerm(expr, sector=sector, name=name))
        return self

    def sector(self, name):
        """Sum of all terms in one sector."""
        return sp.Add(*[t.expr for t in self.terms if t.sector == name])

    @property
    def total(self):
        return sp.Add(*[t.expr for t in self.terms])

    def _repr_latex_(self):
        return f"$\\displaystyle \\mathcal{{L}} = {sp.latex(self.total)}$"

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)


@dataclass
class InvarianceReport:
    """Result of :meth:`Model.check_invariance`."""

    #: list of (term, check_name, details)
    failures: list = dc_field(default_factory=list)
    checked: int = 0

    @property
    def ok(self):
        return not self.failures

    def raise_on_failure(self):
        if not self.ok:
            lines = [f"  [{name}] term {term.name or term.expr}: {details}"
                     for term, name, details in self.failures]
            raise ValueError("invariance check failed:\n" + "\n".join(lines))
        return self

    def __repr__(self):
        status = "OK" if self.ok else f"{len(self.failures)} FAILURES"
        return f"InvarianceReport({self.checked} checks, {status})"


class ValidationReport:
    """Aggregate result of :meth:`Model.validate`.

    Holds the named sub-reports of every consistency check that ran (each an
    object with an ``.ok`` property and a ``raise_on_failure()`` method); a
    ``None`` entry marks a check that was not applicable and was skipped.
    """

    def __init__(self, checks):
        #: ``{name: sub-report or None}`` in run order
        self.checks = checks

    @property
    def ok(self):
        return all(r.ok for r in self.checks.values() if r is not None)

    def raise_on_failure(self):
        failed = [n for n, r in self.checks.items()
                  if r is not None and not r.ok]
        if failed:
            raise ValueError(
                "model validation failed: " + ", ".join(failed))
        return self

    def summary(self):
        """Multi-line human-readable rundown of every check."""
        lines = [f"ValidationReport [{'PASS' if self.ok else 'FAIL'}]"]
        for name, r in self.checks.items():
            if r is None:
                lines.append(f"  {name}: skipped")
            else:
                mark = "ok" if r.ok else "FAILED"
                lines.append(f"  {name}: {mark} — {r!r}")
        return "\n".join(lines)

    def __repr__(self):
        run = [r for r in self.checks.values() if r is not None]
        n_ok = sum(1 for r in run if r.ok)
        status = "PASS" if self.ok else "FAIL"
        return f"ValidationReport({n_ok}/{len(run)} checks passed, {status})"


class Model:
    """A BSM model: symmetries + fields + parameters + Lagrangian.

    All pipeline stages are **lazy** — nothing is solved at construction.

    Pipeline surface:

    - :meth:`check_invariance` — gauge/discrete invariance of every term,
      hermiticity per sector, mass-dimension power counting;
    - :attr:`potential`, :attr:`vacuum` — EWSB setup (``L ⊃ −V``);
    - :meth:`tadpoles`, :meth:`solve_tadpoles` — vacuum conditions;
    - :meth:`mass_matrix` — real or charged scalar blocks at the vacuum;
    - :meth:`rotate` — register weak → physical Rotations;
    - :meth:`physical_lagrangian` — shifted, tadpole-substituted, rotated L;
    - :meth:`interactions`, :meth:`feynman_rules` — vertex extraction.
    """

    def __init__(self, name, gauge_groups=(), discrete_groups=(), fields=(),
                 parameters=None, lagrangian=None):
        self.name = name
        self.gauge_groups = list(gauge_groups)
        self.discrete_groups = list(discrete_groups)
        self.fields = list(fields)
        if parameters is None:
            parameters = ParameterSet()
        elif not isinstance(parameters, ParameterSet):
            parameters = ParameterSet(*parameters)
        self.parameters = parameters
        self.lagrangian = lagrangian if lagrangian is not None else Lagrangian()

        for g in self.gauge_groups:
            if not isinstance(g, GaugeGroup):
                raise TypeError(f"{g!r} is not a GaugeGroup")
        for g in self.discrete_groups:
            if not isinstance(g, DiscreteSymmetry):
                raise TypeError(f"{g!r} is not a DiscreteSymmetry")

        #: registered weak → physical Rotations, in application order
        self.rotations = []
        self._tadpole_solutions = {}
        self._cache = {}

    def _invalidate(self):
        """Drop cached pipeline results after a state mutation."""
        self._cache.clear()

    # ------------------------------------------------------------------ EWSB

    @property
    def scalars(self):
        return [f for f in self.fields if isinstance(f, Scalar)]

    @property
    def potential(self):
        """The scalar potential ``V`` (the Lagrangian stores ``−V``)."""
        return -self.lagrangian.sector("potential")

    @property
    def vacuum(self):
        if "vacuum" not in self._cache:
            self._cache["vacuum"] = Vacuum(self.scalars)
        return self._cache["vacuum"]

    def tadpoles(self):
        """Tadpole conditions ``{vev: ∂V/∂vev |_vacuum}``."""
        if "tadpoles" not in self._cache:
            self._cache["tadpoles"] = extract_tadpoles(self.potential,
                                                       self.vacuum)
        return self._cache["tadpoles"]

    def solve_tadpoles(self, for_params):
        """Solve tadpoles for ``for_params``; solutions are remembered and
        applied by :meth:`mass_matrix` / :meth:`physical_lagrangian`, and
        any :class:`InternalParameter` among them gets defined."""
        solution = solve_tadpoles(self.potential, self.vacuum, for_params)
        self._tadpole_solutions.update(solution)
        self._invalidate()
        return solution

    def mass_matrix(self, fields, charged=False):
        """Scalar mass matrix at the vacuum for a block of fields.

        Args:
            fields: real fluctuation symbols (CP-even/odd block) or complex
                weak components with ``charged=True``.
            charged: use the ``∂²V/∂φ̄∂φ`` complex-field derivative.
        """
        builder = charged_mass_matrix if charged else scalar_mass_matrix
        return builder(self.potential, self.vacuum, list(fields),
                       tadpole_subs=self._tadpole_solutions)

    def gauge_mass_matrix(self, gauge_components):
        """Gauge boson mass matrix from the (vacuum-evaluated) kinetic
        sector: ``M²_ab = ∂²L_kin,vac/∂A^a∂A^b``."""
        return gauge_mass_matrix(self.lagrangian.sector("kinetic"),
                                 self.vacuum, list(gauge_components),
                                 tadpole_subs=self._tadpole_solutions)

    # ------------------------------------------------------- physical basis

    def rotate(self, rotation):
        """Register a weak → physical :class:`~feynlag.vacuum.Rotation`."""
        self.rotations.append(rotation)
        self._invalidate()
        return rotation

    def physical_lagrangian(self, sector=None):
        """The Lagrangian in the physical basis: vacuum-shifted, tadpole
        solutions substituted, all registered rotations applied, expanded."""
        key = ("physical_lagrangian", sector)
        if key not in self._cache:
            L = (self.lagrangian.total if sector is None
                 else self.lagrangian.sector(sector))
            L = self.vacuum.shift(L)
            if self._tadpole_solutions:
                L = L.subs(self._tadpole_solutions)
            for rot in self.rotations:
                L = L.xreplace(rot.substitution())
            self._cache[key] = sp.expand(L)
        return self._cache[key]

    # -------------------------------------------------------------- vertices

    def interactions(self, fields, sector=None, conjugate_map=None,
                     min_legs=3):
        """Extract interaction coefficients from the physical Lagrangian.

        Args:
            fields: physical field symbols to extract vertices for.
            sector: restrict to one Lagrangian sector (default: all).
            conjugate_map: optional ``{conjugate(φ): φ̄_symbol}`` dict
                normalizing conjugated complex fields into plain symbols
                (e.g. ``conjugate(Gp) → Gm``) before extraction.
            min_legs: drop monomials with fewer field legs (default 3 —
                vacuum/tadpole/mass terms are not interactions).

        Returns:
            ``{n_fields: {sorted-field-tuple: coefficient}}``.
        """
        L = self.physical_lagrangian(sector=sector)
        if L.has(PartialMu):
            # derivative couplings: Leibniz-expand and go to momentum space;
            # conjugated complex fields are dynamical too
            dynamical = list(fields)
            if conjugate_map:
                dynamical += list(conjugate_map.keys())
            L = to_momentum_space(L, dynamical)
        if conjugate_map:
            L = L.xreplace(conjugate_map)
        table = extract_interaction_coefficients(L, list(fields))
        return {n: terms for n, terms in table.items() if n >= min_legs}

    # ---------------------------------------------------------------- spins

    def spin_map(self, conjugate_map=None):
        """``{symbol: spin}`` for every known component, fluctuation and
        rotated physical field (rotations propagate block spin; conjugate
        partners inherit the spin of the field they conjugate)."""
        spins = {}
        for f in self.fields:
            spin = getattr(f, "spin", None)
            for c in f.components:
                spins[c] = spin
            for comp, (vev, re, im) in getattr(f, "vev_expansions",
                                               {}).items():
                spins[re] = 0
                if im is not None:
                    spins[im] = 0
        for rot in self.rotations:
            block = {spins.get(o) for o in rot.old_fields}
            if len(block) == 1:
                spin = block.pop()
                for nf in rot.new_fields:
                    spins[nf] = spin
        if conjugate_map:
            for conj_node, partner in conjugate_map.items():
                base = conj_node.args[0] if conj_node.args else conj_node
                if base in spins:
                    spins[partner] = spins[base]
        return spins

    def vertices(self, fields, sector=None, conjugate_map=None, min_legs=3,
                 simplifier=None):
        """Extract :class:`~feynlag.vertices.vertex.Vertex` objects (typed by
        the closed Lorentz catalog) from the physical Lagrangian."""
        table = self.interactions(fields, sector=sector,
                                  conjugate_map=conjugate_map,
                                  min_legs=min_legs)
        spins = self.spin_map(conjugate_map=conjugate_map)
        out = []
        for terms in table.values():
            for field_tuple, coeff in terms.items():
                if simplifier is not None:
                    coeff = simplifier(coeff)
                if coeff == 0:
                    continue
                out.append(Vertex.from_coefficient(field_tuple, coeff,
                                                   spin_map=spins))
        return out

    def feynman_rules(self, fields, sector=None, conjugate_map=None,
                      min_legs=3, simplifier=None):
        """Feynman rules ``i × coefficient × ∏(multiplicity)!`` per vertex.

        Returns:
            flat dict ``{sorted-field-tuple: rule}``.
        """
        table = self.interactions(fields, sector=sector,
                                  conjugate_map=conjugate_map,
                                  min_legs=min_legs)
        rules = {}
        for terms in table.values():
            for field_tuple, coeff in terms.items():
                rule = feynman_rule(coeff, field_tuple)
                if simplifier is not None:
                    rule = simplifier(rule)
                if rule != 0:
                    rules[field_tuple] = rule
        return rules

    # ------------------------------------------------------------ invariance

    def check_invariance(self, hermiticity=True, dimension=True,
                         raise_on_failure=False):
        """Check every Lagrangian term against every declared symmetry.

        Args:
            hermiticity: also check ``L = L*`` per sector.
            dimension: also check mass dimension ≤ 4 per term.
            raise_on_failure: raise ``ValueError`` instead of returning a
                failing report.

        Returns:
            :class:`InvarianceReport`.
        """
        report = InvarianceReport()

        for term in self.lagrangian:
            for group in self.gauge_groups:
                ok, violations = check_gauge_invariance(
                    term.expr, self.fields, group)
                report.checked += 1
                if not ok:
                    report.failures.append(
                        (term, f"gauge:{group.name}", violations))
            for group in self.discrete_groups:
                ok, violations = check_discrete_invariance(term.expr, group)
                report.checked += 1
                if not ok:
                    report.failures.append(
                        (term, f"discrete:{group.name}", violations))
            if dimension:
                ok, worst = check_mass_dimension(
                    term.expr, self.fields, self.parameters)
                report.checked += 1
                if not ok:
                    report.failures.append(
                        (term, "mass-dimension", f"dimension {worst} > 4"))

        if hermiticity:
            for sector in SECTORS:
                expr = self.lagrangian.sector(sector)
                if expr == 0:
                    continue
                ok, residual = check_hermiticity(expr)
                report.checked += 1
                if not ok:
                    fake_term = LagrangianTerm(expr, sector=sector,
                                               name=f"<sector {sector}>")
                    report.failures.append(
                        (fake_term, "hermiticity", residual))

        if raise_on_failure:
            report.raise_on_failure()
        return report

    def check_anomalies(self, raise_on_failure=False):
        """Check that gauge anomalies cancel for the declared fermion content.

        Returns an :class:`~feynlag.anomalies.AnomalyReport`; see
        :mod:`feynlag.anomalies`.
        """
        from .anomalies import check_anomaly_free
        report = check_anomaly_free(self)
        if raise_on_failure:
            report.raise_on_failure()
        return report

    def validate(self, invariance=True, hermiticity=True, dimension=True,
                 anomalies=True, ufo_path=None, external_values=None,
                 charges=None, fields=None, conjugate_map=None,
                 conjugates=None, fermion_table=None, raise_on_failure=False):
        """Run every applicable consistency check and aggregate the results.

        The umbrella entry point: it runs the symmetry/hermiticity/dimension
        checks ({meth}`check_invariance`), the gauge-anomaly check
        ({meth}`check_anomalies`, skipped when the model has no charged
        fermions), and — if ``ufo_path`` names a written UFO directory — the
        numeric round-trip of the exported model
        ({func}`~feynlag.verify.verify_ufo_numeric`).

        When a declared ``charges`` map and the physical ``fields`` list are
        supplied it also runs the charge-based checks
        ({mod}`feynlag.charges`): per-vertex charge conservation, the
        declared-vs-vacuum-derived charge-consistency cross-check (only when
        the model has a vacuum), and vertex-level hermiticity pairing.

        Args:
            invariance / hermiticity / dimension: toggle and configure the
                :meth:`check_invariance` pass.
            anomalies: run the gauge-anomaly check when applicable.
            ufo_path: directory written by ``write_ufo`` to round-trip; skip
                if ``None``.
            external_values: optional external-parameter overrides forwarded to
                the UFO round-trip.
            charges: declared ``{physical field: electric charge}`` map; enables
                the charge-based checks (needs ``fields`` too).
            fields: physical boson symbols to extract vertices from (as in
                :meth:`feynman_rules`).
            conjugate_map: ``{conjugate(sym): partner}`` for vertex extraction.
            conjugates: full antiparticle pairing ``{field: partner}`` (both
                directions) for hermiticity pairing; defaults to the pairs in
                ``conjugate_map``.
            fermion_table: optional ``extract_fermion_vertices`` output to add
                the fermionic vertices to the charge/hermiticity checks.
            raise_on_failure: raise ``ValueError`` if any check fails.

        Returns:
            :class:`ValidationReport`.
        """
        from .fields import Fermion

        checks = {}
        if invariance:
            checks["invariance"] = self.check_invariance(
                hermiticity=hermiticity, dimension=dimension)
        if anomalies:
            has_charged_fermions = bool(self.gauge_groups) and any(
                isinstance(f, Fermion) and f.reps for f in self.fields)
            checks["anomalies"] = (self.check_anomalies()
                                   if has_charged_fermions else None)
        if ufo_path is not None:
            from .verify import verify_ufo_numeric
            checks["ufo_roundtrip"] = verify_ufo_numeric(
                ufo_path, external_values=external_values)
        if charges is not None and fields is not None:
            from .charges import (
                ChargeRegistry, check_charge_conservation,
                check_charge_consistency, check_hermiticity_pairing)
            registry = ChargeRegistry(charges, conjugate_map=conjugate_map)
            verts = self.vertices(fields, conjugate_map=conjugate_map,
                                  simplifier=sp.simplify)
            if conjugates is None:
                conjugates = {}
                for conj_node, partner in (conjugate_map or {}).items():
                    base = conj_node.args[0] if conj_node.args else conj_node
                    conjugates[base] = partner
                    conjugates[partner] = base
            checks["charge_conservation"] = check_charge_conservation(
                registry, bosonic_vertices=verts, fermion_table=fermion_table)
            has_vacuum = any(getattr(f, "vev_expansions", None)
                             for f in self.fields)
            checks["charge_consistency"] = (
                check_charge_consistency(self, registry)
                if has_vacuum and self.rotations else None)
            checks["hermiticity_pairing"] = check_hermiticity_pairing(
                bosonic_vertices=verts, conjugates=conjugates,
                fermion_table=fermion_table)

        report = ValidationReport(checks)
        if raise_on_failure:
            report.raise_on_failure()
        return report
