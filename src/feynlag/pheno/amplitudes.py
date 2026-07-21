"""Amplitude squared evaluators for different vertex catalog types."""

import sympy as sp
from .trace import dirac_trace
from feynlag.dirac import DiracGamma, dirac_conjugate, minkowski_metric

def build_slashed(p_symbol, mu_index):
    """Construct p_slash = gamma^mu p_mu."""
    return DiracGamma(mu_index) * p_symbol(mu_index)

def squared_amplitude(vertex, parent_momentum, child_momenta, masses):
    """
    Constructs and evaluates |M|^2 symbolically.
    
    Args:
        vertex: Vertex object.
        parent_momentum: SymPy Function or Symbol representing parent momentum.
        child_momenta: List of SymPy Functions representing children momenta.
        masses: Dict mapping field symbols to their mass variables.
        
    Returns:
        SymPy expression for |M|^2.
    """
    vtype = vertex.vertex_type
    
    # Remove the overall 'i' from coupling to get the matrix element M
    # vertex.coupling is i * M_vertex
    # We want M, so we factor out i. Wait, actually M = -i V usually?
    # feynlag rule is i * coeff. So M = -i * (i * coeff) = coeff.
    M_expr = vertex.coupling / sp.I
    
    if vtype == "SSS":
        return sp.expand(M_expr * sp.conjugate(M_expr))
        
    elif vtype == "FFS":
        # S(p) -> F1(p1) F2(p2)
        # Assuming M_expr is just a matrix in Dirac space.
        # |M|^2 = Tr[ M (p1_slash + m1) M^dagger (p2_slash - m2) ]
        # This requires carefully substituting indices and tracing.
        # We will implement this rigorously in subsequent iterations.
        raise NotImplementedError("FFS |M|^2 not fully implemented.")

    raise NotImplementedError(f"Amplitude squared for {vtype} not implemented yet.")
