"""Symbolic Dirac trace evaluator."""

import sympy as sp
from feynlag.dirac import (
    DiracGamma, DiracGammaLower, PL, PR, DiracIdentity, DiracZero,
    minkowski_metric
)

def dirac_trace(expr):
    """Evaluate the trace of a Dirac matrix expression.
    
    Applies Tr[...].
    - Tr[odd number of gammas] = 0
    - Tr[gamma_mu gamma_nu] = 4 g_{mu nu}
    - Tr[gamma_mu gamma_nu gamma_alpha gamma_beta] = 4(g_{mu nu}g_{alpha beta} - ...)
    - Tr[PL] = Tr[PR] = 2
    
    Note: For 2-body unpolarized decays, the gamma_5 trace vanishes because 
    the Levi-Civita tensor contracts with symmetric momentum tensors.
    """
    expr = sp.expand(expr)
    
    if isinstance(expr, DiracZero) or expr == 0:
        return sp.S.Zero
        
    if isinstance(expr, DiracIdentity):
        return sp.S(4)
        
    if isinstance(expr, sp.Add):
        return sp.Add(*[dirac_trace(arg) for arg in expr.args])
        
    if isinstance(expr, sp.Mul):
        # Separate commuting coefficients from Dirac objects
        coeff = sp.S.One
        dirac_parts = []
        for arg in expr.args:
            if isinstance(arg, (DiracGamma, DiracGammaLower, PL, PR, DiracIdentity, DiracZero)):
                dirac_parts.append(arg)
            else:
                coeff *= arg
                
        if not dirac_parts:
            return coeff * 4
            
        dirac_parts = [p for p in dirac_parts if not isinstance(p, DiracIdentity)]
        if not dirac_parts:
            return coeff * 4
            
        gammas = [g for g in dirac_parts if isinstance(g, (DiracGamma, DiracGammaLower))]
        projectors = [p for p in dirac_parts if isinstance(p, (PL, PR))]
        
        if len(gammas) % 2 != 0:
            return sp.S.Zero
            
        if not gammas and len(projectors) == 1:
            return coeff * 2
            
        if len(gammas) == 2:
            idx1 = gammas[0].index
            idx2 = gammas[1].index
            trace_val = 4 * minkowski_metric(idx1, idx2)
            if projectors:
                trace_val = 2 * minkowski_metric(idx1, idx2)
            return coeff * trace_val
            
        if len(gammas) == 4:
            i1 = gammas[0].index
            i2 = gammas[1].index
            i3 = gammas[2].index
            i4 = gammas[3].index
            g = minkowski_metric
            trace_val = 4 * (g(i1, i2)*g(i3, i4) - g(i1, i3)*g(i2, i4) + g(i1, i4)*g(i2, i3))
            if projectors:
                trace_val = trace_val / 2
            return coeff * trace_val
            
        raise NotImplementedError(f"dirac_trace not implemented for >4 gammas: {dirac_parts}")
        
    return expr * 4
