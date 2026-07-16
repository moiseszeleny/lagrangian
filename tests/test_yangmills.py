"""Ground-truth verification of the Yang-Mills quartic (VVVV) self-coupling
assembly — the "Phase 5" step yangmills.py's ``quartic_couplings`` docstring
deferred and that was never built or pinned for ANY gauge group before this
(confirmed by reading tests/test_gauge_sector_sm.py::test_quartic_gauge_couplings,
which only asserts the SU(2) WWWW coupling is nonzero and proportional to
g^2 — no exact value has ever been pinned anywhere in this codebase).

The cross-check here is independent of ``export.ufo.vvvv.assemble_vvvv``'s
own implementation: it differentiates ``-1/4 F^a_{mu nu} F^{a mu nu}``
directly (F built straight from ``structure_constants``, no reliance on
``quartic_couplings`` at all) and reads off the three independent
Lorentz-structure coefficients via diagonal-metric-component probing
(picking (mu1,mu2,mu3,mu4) values that isolate exactly one of
Metric(1,2)Metric(3,4) / Metric(1,3)Metric(2,4) / Metric(1,4)Metric(2,3) at
a time, since eta is diagonal). This is checked against SU(3) (gluon,
4 distinct adjoint indices, including a case where all three structures are
simultaneously nonzero) and SU(2) (W self-coupling, necessarily repeated
adjoint indices since there are only 3 generators for 4 legs).
"""

import sympy as sp
import pytest

from feynlag import SU2, SU3, quartic_couplings, structure_constants
from feynlag.export.ufo.vvvv import assemble_vvvv


def _direct_vvvv(group, legs_idx):
    """Differentiate -1/4 F^a F^a directly; legs_idx = (i,j,k,l) adjoint
    indices (repeats allowed). Returns {VVVV1,VVVV2,VVVV3: coeff}, matching
    lorentz_map.py's definitions, with zero entries dropped."""
    n = group.n_generators
    f = structure_constants(group)
    g = group.g
    eta = sp.diag(1, -1, -1, -1)
    A = [[sp.Symbol(f"A{a}_{mu}") for mu in range(4)] for a in range(n)]

    def Fint(a, mu, nu):
        s = sp.S.Zero
        for b in range(n):
            for c in range(n):
                val = f.get((a, b, c), sp.S.Zero)
                if val != 0:
                    s += g * val * A[b][mu] * A[c][nu]
        return s

    L4 = sp.S.Zero
    for a in range(n):
        for mu in range(4):
            for nu in range(4):
                L4 += (sp.Rational(-1, 4) * eta[mu, mu] * eta[nu, nu]
                       * Fint(a, mu, nu) * Fint(a, mu, nu))
    L4 = sp.expand(L4)

    def vertex(legs):
        expr = L4
        for (a, mu) in legs:
            expr = sp.diff(expr, A[a][mu])
        return sp.expand(expr)

    i, j, k, l = legs_idx
    # diagonal-component probing: at these specific (mu1..mu4) choices, the
    # OTHER two metric-pair structures vanish identically (off-diagonal eta),
    # isolating exactly one coefficient per probe.
    c12 = -vertex([(i, 0), (j, 0), (k, 1), (l, 1)])   # Metric(1,2)Metric(3,4)
    c13 = -vertex([(i, 0), (j, 1), (k, 0), (l, 1)])   # Metric(1,3)Metric(2,4)
    c14 = -vertex([(i, 0), (j, 1), (k, 1), (l, 0)])   # Metric(1,4)Metric(2,3)
    raw = {"VVVV1": c14 - c13, "VVVV2": c14 - c12, "VVVV3": c13 - c12}
    return {name: v for name, v in raw.items() if v != 0}


class TestVVVVAssemblyGroundTruth:
    def test_su3_gluon_quartic_four_distinct_colors(self):
        """4 distinct adjoint indices; also exercises a case where all
        three VVVV structures are simultaneously nonzero (not just the
        common all-but-one-zero pattern)."""
        g = sp.Symbol("g")
        SU3c = SU3("SU3c", coupling=g)
        comps = list(SU3c.bosons().components)
        qc = quartic_couplings(SU3c)
        for (i, j, k, l) in [(0, 1, 3, 4), (0, 3, 5, 7)]:
            quad = (comps[i], comps[j], comps[k], comps[l])
            got = assemble_vvvv(qc, quad)
            expected = _direct_vvvv(SU3c, (i, j, k, l))
            assert got.keys() == expected.keys(), (i, j, k, l, got, expected)
            for name in expected:
                assert sp.simplify(got[name] - expected[name]) == 0, name

    def test_su2_w_self_coupling_repeated_indices(self):
        """Only 3 generators for 4 legs — necessarily repeated adjoint
        indices, the physically relevant WWWW-type case."""
        g = sp.Symbol("g")
        SU2L = SU2("SU2L", coupling=g)
        comps = list(SU2L.bosons().components)
        qc = quartic_couplings(SU2L)
        quad = (comps[0], comps[1], comps[0], comps[1])
        got = assemble_vvvv(qc, quad)
        expected = _direct_vvvv(SU2L, (0, 1, 0, 1))
        assert got.keys() == expected.keys()
        for name in expected:
            assert sp.simplify(got[name] - expected[name]) == 0, name
