"""Direct small-case Dirac trace evaluator.

This is **not** the decay pipeline's engine — that is
:func:`feynlag.pheno.lorentz.dirac_trace`, which handles arbitrary chain
lengths through SymPy's Clifford algebra.  What lives here is a short,
self-contained implementation of the standard identities for chains of at most
four gamma matrices, written directly against feynlag's own
:class:`~feynlag.dirac.DiracGamma` objects.

Its value is **independence**: it shares no code with the covariant engine, so
the two agreeing on the overlapping identities is a genuine cross-check rather
than a tautology (``tests/test_pheno.py``).  Note that
:func:`~feynlag.dirac.minkowski_metric` returns a ``KroneckerDelta`` expression
for *symbolic* indices, so comparisons against the covariant engine are made at
concrete integer indices, where it returns ±1.
"""

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

    # A repeated index collapses to a Pow -- SymPy stores gamma(0)*gamma(0) as
    # gamma(0)**2, which never reaches the Mul branch below.  Clifford gives
    # (gamma^mu)^2 = g^{mu mu} I, so the trace is 4 g^{mu mu}.
    if (isinstance(expr, sp.Pow) and expr.exp == 2
            and isinstance(expr.base, (DiracGamma, DiracGammaLower))):
        return 4 * minkowski_metric(expr.base.index, expr.base.index)

    if isinstance(expr, sp.Mul):
        # Separate commuting coefficients from Dirac objects
        coeff = sp.S.One
        dirac_parts = []
        for arg in expr.args:
            if isinstance(arg, (DiracGamma, DiracGammaLower, PL, PR, DiracIdentity, DiracZero)):
                dirac_parts.append(arg)
            elif (isinstance(arg, sp.Pow) and arg.exp == 2
                    and isinstance(arg.base, (DiracGamma, DiracGammaLower))):
                # same collapse, but as one factor of a larger product
                coeff *= minkowski_metric(arg.base.index, arg.base.index)
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
