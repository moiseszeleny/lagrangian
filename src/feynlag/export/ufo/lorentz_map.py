"""UFO Lorentz structures for the closed v1 vertex catalog.

Structure strings follow the FeynRules/UFO index conventions:
``P(l, n)`` = momentum of leg *n* with Lorentz index *l* (indices are leg
numbers); ``Metric(m, n)``; ``Gamma(l, s1, s2)``; ``ProjM``/``ProjP`` = P_∓.
UFO spins are ``2s+1``: scalar 1, fermion 2, vector 3.
"""

__all__ = ["UFO_LORENTZ", "structures_for"]

#: name → (spins, structure)
UFO_LORENTZ = {
    "SSS1":  ([1, 1, 1], "1"),
    "SSSS1": ([1, 1, 1, 1], "1"),
    "VSS1":  ([3, 1, 1], "P(1,2) - P(1,3)"),
    "VVS1":  ([3, 3, 1], "Metric(1,2)"),
    "VVSS1": ([3, 3, 1, 1], "Metric(1,2)"),
    "VVV1":  ([3, 3, 3],
              "P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) "
              "+ P(2,3)*Metric(1,3) - P(2,1)*Metric(1,3) "
              "+ P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)"),
    "VVVV1": ([3, 3, 3, 3],
              "Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)"),
    "VVVV2": ([3, 3, 3, 3],
              "Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)"),
    "VVVV3": ([3, 3, 3, 3],
              "Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)"),
    "FFSL":  ([2, 2, 1], "ProjM(2,1)"),
    "FFSR":  ([2, 2, 1], "ProjP(2,1)"),
    "FFVL":  ([2, 2, 3], "Gamma(3,2,-1)*ProjM(-1,1)"),
    "FFVR":  ([2, 2, 3], "Gamma(3,2,-1)*ProjP(-1,1)"),
}


def structures_for(vertex_type):
    """UFO lorentz names available for a feynlag catalog vertex type."""
    mapping = {
        "SSS": ["SSS1"], "SSSS": ["SSSS1"], "VSS": ["VSS1"],
        "VVS": ["VVS1"], "VVSS": ["VVSS1"], "VVV": ["VVV1"],
        "VVVV": ["VVVV1", "VVVV2", "VVVV3"],
        "FFS": ["FFSL", "FFSR"], "FFV": ["FFVL", "FFVR"],
    }
    return mapping[vertex_type]
