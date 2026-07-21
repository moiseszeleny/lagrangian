"""Phase space kinematics."""

import sympy as sp

def kallen(x, y, z):
    """The Källén triangle function."""
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*z*x

def two_body_phase_space(M_parent, m1, m2):
    """Two-body phase space integration factor.
    
    Args:
        M_parent: Mass of the decaying particle.
        m1: Mass of first daughter.
        m2: Mass of second daughter.
        
    Returns:
        The kinematic factor integrated over phase space.
    """
    return sp.sqrt(kallen(M_parent**2, m1**2, m2**2)) / (16 * sp.pi * M_parent**3)
