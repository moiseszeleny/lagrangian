"""Model.validate() umbrella: aggregation, skip logic, failure propagation.

The individual checks (invariance, anomalies, UFO round-trip) have their own
dedicated test files; here we pin only the umbrella behaviour — that
``validate`` runs the applicable checks, skips the inapplicable ones, and
reports ``ok`` iff every check that ran passed.
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, Model, Scalar, SU2, SU3, U1, WeylFermion,
    ValidationReport,
)


def _groups():
    gs = ExternalParameter("g_s", 1.2, positive=True)
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    return SU3("SU3c", coupling=gs), SU2("SU2L", coupling=gw), \
        U1("U1Y", coupling=g1)


def _sm_fermions(SU3c, SU2L, U1Y):
    R = sp.Rational

    def reps(su3, su2, y):
        d = {U1Y: y}
        if su3 != 1:
            d[SU3c] = su3
        if su2 != 1:
            d[SU2L] = su2
        return d

    return [
        WeylFermion("QL", reps=reps(3, 2, R(1, 6)), chirality="L", nflavors=3),
        WeylFermion("uR", reps=reps(3, 1, R(2, 3)), chirality="R", nflavors=3),
        WeylFermion("dR", reps=reps(3, 1, R(-1, 3)), chirality="R", nflavors=3),
        WeylFermion("LL", reps=reps(1, 2, R(-1, 2)), chirality="L", nflavors=3),
        WeylFermion("eR", reps=reps(1, 1, -1), chirality="R", nflavors=3),
    ]


def test_validate_aggregates_passing_checks():
    SU3c, SU2L, U1Y = _groups()
    model = Model("SM", gauge_groups=[SU3c, SU2L, U1Y],
                  fields=_sm_fermions(SU3c, SU2L, U1Y))
    report = model.validate()
    assert isinstance(report, ValidationReport)
    assert report.ok, report.summary()
    assert "invariance" in report.checks
    assert report.checks["anomalies"] is not None      # ran, did not skip
    assert report.checks["anomalies"].ok


def test_validate_flags_anomaly_failure():
    """Drop eR: hypercharge anomalies no longer cancel -> umbrella fails."""
    SU3c, SU2L, U1Y = _groups()
    fields = _sm_fermions(SU3c, SU2L, U1Y)[:-1]        # no eR
    model = Model("SM_noER", gauge_groups=[SU3c, SU2L, U1Y], fields=fields)
    report = model.validate()
    assert not report.ok
    assert not report.checks["anomalies"].ok
    with pytest.raises(ValueError, match="anomalies"):
        report.raise_on_failure()


def test_validate_skips_anomalies_without_fermions():
    SU3c, SU2L, U1Y = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    model = Model("scalars", gauge_groups=[SU3c, SU2L, U1Y], fields=[H])
    report = model.validate()
    assert report.checks["anomalies"] is None          # skipped
    assert report.ok
    assert "skipped" in report.summary()


def test_validate_can_disable_checks():
    SU3c, SU2L, U1Y = _groups()
    model = Model("SM", gauge_groups=[SU3c, SU2L, U1Y],
                  fields=_sm_fermions(SU3c, SU2L, U1Y))
    report = model.validate(invariance=False, anomalies=True)
    assert "invariance" not in report.checks
    assert "anomalies" in report.checks


def test_validate_omits_charge_checks_without_charges():
    """Backward compatible: no charge/hermiticity entries unless charges given."""
    SU3c, SU2L, U1Y = _groups()
    model = Model("SM", gauge_groups=[SU3c, SU2L, U1Y],
                  fields=_sm_fermions(SU3c, SU2L, U1Y))
    report = model.validate()
    assert "charge_conservation" not in report.checks
    assert "hermiticity_pairing" not in report.checks
