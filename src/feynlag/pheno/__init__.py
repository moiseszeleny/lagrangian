"""Phenomenology: tree-level two-body decay widths and branching ratios.

Turns the Feynman rules feynlag extracts into partial widths, total widths and
branching ratios for 1→2 decays.

The computation path is **covariant**: Dirac traces are evaluated with SymPy's
own Clifford engine (:mod:`sympy.physics.hep.gamma_matrices` — ships with
SymPy, so the library stays pure-SymPy), contracted with spin and polarization
sums, and reduced to on-shell invariants ``p_i·p_j`` only at the end.  Explicit
4×4 Dirac matrices are deliberately kept off this path and used solely as the
independent test oracle in ``tests/test_pheno.py``.

Typical use::

    calc = DecayCalculator(model, masses=..., boson_fields=[h, Z, Wp, Wm],
                           fermion_sectors=("gauge", "yukawa"))
    calc.partial_widths(h)        # {(f, fbar): Γ, ...}
    calc.branching_ratios(h)
"""

from .amplitudes import (
    amplitude_squared, ffs_squared, ffv_squared, polarization_sum, spin_sum,
    sss_squared, vvs_squared,
)
from .calculator import DecayCalculator, DecayChannel, partial_width
from .kinematics import (
    TwoBodyKinematics, is_allowed, kallen, two_body_momentum,
    two_body_phase_space,
)
from .lorentz import contract_to_dots, dirac_trace, reduce_projectors
from .particles import DiracParticle, expand_particles
from .vertices import DecayVertex, classify_gamma, collect_decay_vertices

__all__ = [
    "DecayCalculator", "DecayChannel", "DecayVertex", "DiracParticle",
    "TwoBodyKinematics", "amplitude_squared", "classify_gamma",
    "collect_decay_vertices", "contract_to_dots", "dirac_trace",
    "expand_particles", "ffs_squared", "ffv_squared", "is_allowed", "kallen",
    "partial_width", "polarization_sum", "reduce_projectors", "spin_sum",
    "sss_squared", "two_body_momentum", "two_body_phase_space", "vvs_squared",
]
