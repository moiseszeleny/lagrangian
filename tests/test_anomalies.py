"""Gauge-anomaly cancellation of the declared fermion content.

Pins the textbook results: the Standard Model (with three right-handed
neutrinos) is anomaly-free under SU(3)_c × SU(2)_L × U(1)_Y × U(1)_{B-L};
dropping a field or mistuning a hypercharge makes a coefficient non-zero; and
with a symbolic U(1)_X charge the coefficients become the polynomial
constraints whose solution is the anomaly-free assignment.
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, Model, SU2, SU3, U1, WeylFermion,
    anomaly_coefficients, check_anomaly_free,
)


def _sm_groups():
    gs = ExternalParameter("g_s", 1.2, positive=True)
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    gx = ExternalParameter("gx", 0.3, positive=True)
    SU3c = SU3("SU3c", coupling=gs)
    SU2L = SU2("SU2L", coupling=gw)
    U1Y = U1("U1Y", coupling=g1)
    U1BL = U1("U1BL", coupling=gx)
    return SU3c, SU2L, U1Y, U1BL


def _sm_fermions(SU3c, SU2L, U1Y, U1BL=None, extra=None, nu=True):
    """One-parameter family of SM fermions; ``extra`` overrides B-L charges."""
    R = sp.Rational

    def reps(su3, su2, y, bl):
        d = {U1Y: y}
        if su3 != 1:
            d[SU3c] = su3
        if su2 != 1:
            d[SU2L] = su2
        if U1BL is not None:
            d[U1BL] = bl
        return d

    fields = [
        WeylFermion("QL", reps=reps(3, 2, R(1, 6), R(1, 3)), chirality="L",
                    nflavors=3),
        WeylFermion("uR", reps=reps(3, 1, R(2, 3), R(1, 3)), chirality="R",
                    nflavors=3),
        WeylFermion("dR", reps=reps(3, 1, R(-1, 3), R(1, 3)), chirality="R",
                    nflavors=3),
        WeylFermion("LL", reps=reps(1, 2, R(-1, 2), -1), chirality="L",
                    nflavors=3),
        WeylFermion("eR", reps=reps(1, 1, -1, -1), chirality="R", nflavors=3),
    ]
    if nu:
        fields.append(
            WeylFermion("nuR", reps=reps(1, 1, 0, -1), chirality="R",
                        nflavors=3))
    return fields


def test_standard_model_is_anomaly_free():
    SU3c, SU2L, U1Y, U1BL = _sm_groups()
    fields = _sm_fermions(SU3c, SU2L, U1Y, U1BL, nu=True)
    model = Model("SM_BL", gauge_groups=[SU3c, SU2L, U1Y, U1BL], fields=fields)
    report = check_anomaly_free(model)
    assert report.ok, report.nonzero


def test_key_coefficients_vanish_exactly():
    SU3c, SU2L, U1Y, U1BL = _sm_groups()
    fields = _sm_fermions(SU3c, SU2L, U1Y, U1BL, nu=True)
    model = Model("SM_BL", gauge_groups=[SU3c, SU2L, U1Y, U1BL], fields=fields)
    c = anomaly_coefficients(model)
    assert c["[SU3c]^2-U1Y"] == 0
    assert c["[SU2L]^2-U1Y"] == 0
    assert c["[U1Y][U1Y][U1Y]"] == 0
    assert c["grav^2-U1Y"] == 0
    assert c["[SU3c]^3"] == 0          # colour is vector-like
    assert c["[U1BL][U1BL][U1BL]"] == 0
    # SU(2) Witten global anomaly: doublet count must be even
    assert int(c["Witten-SU2L"]) % 2 == 0


def test_dropping_neutrino_breaks_B_minus_L():
    SU3c, SU2L, U1Y, U1BL = _sm_groups()
    fields = _sm_fermions(SU3c, SU2L, U1Y, U1BL, nu=False)  # no nuR
    model = Model("SM_noNu", gauge_groups=[SU3c, SU2L, U1Y, U1BL],
                  fields=fields)
    report = check_anomaly_free(model)
    assert not report.ok
    assert "[U1BL][U1BL][U1BL]" in report.nonzero


def test_mistuned_hypercharge_breaks_cancellation():
    SU3c, SU2L, U1Y, _ = _sm_groups()
    fields = _sm_fermions(SU3c, SU2L, U1Y, U1BL=None, nu=False)
    # break eR hypercharge from -1 to -2
    fields[-1] = WeylFermion("eR", reps={U1Y: sp.Integer(-2)}, chirality="R",
                             nflavors=3)
    model = Model("SM_bad", gauge_groups=[SU3c, SU2L, U1Y], fields=fields)
    report = check_anomaly_free(model)
    assert not report.ok


def test_symbolic_charge_gives_polynomial_constraint():
    """A gauged U(1)_X with one symbolic charge: the cubic anomaly becomes a
    constraint polynomial in that charge, not a number."""
    SU3c, SU2L, U1Y, _ = _sm_groups()
    x = sp.Symbol("x", real=True)
    # give nuR a symbolic B-L-like charge x under a new U(1)_X, everything else
    # neutral under X -> [U1X]^3 = (number of nuR) * (-x)^3 for RH, must be a
    # cube in x (non-trivially symbolic).
    U1X = U1("U1X", coupling=ExternalParameter("gX", 0.2, positive=True))
    fields = [
        WeylFermion("nuR", reps={U1Y: sp.Integer(0), U1X: x}, chirality="R",
                    nflavors=1),
    ]
    model = Model("Xtest", gauge_groups=[SU3c, SU2L, U1Y, U1X], fields=fields)
    c = anomaly_coefficients(model)
    cubic = sp.expand(c["[U1X][U1X][U1X]"])
    assert cubic == sp.expand(-x**3)  # single RH field -> (-x)^3
