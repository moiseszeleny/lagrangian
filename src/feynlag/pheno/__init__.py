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
from .diagrams import (
    Amplitude, BosonPropagator, ChainVertex, Diagram, Leg, SpinorChain,
)
from .kinematics import (
    TwoBodyKinematics, is_allowed, kallen, two_body_momentum,
    two_body_phase_space,
)
from .kinematics import ThreeBodyKinematics, TwoToTwoKinematics
from .epsilon import (
    assert_epsilon_single_vanishes, epsilon_pair_tensor, gamma5_trace_coefficient,
)
from .lorentz import contract_to_dots, dirac_trace, reduce_projectors
from .particles import DiracParticle, ExternalState, expand_particles
from .propagator import (
    breit_wigner, propagator_denominator, scalar_propagator,
    vector_propagator, vector_propagator_numerator,
)
from .integrate import dalitz_integral, have_scipy
from .loop import (
    A_half, A_one, higgs_gammagamma_width, higgs_gg_width, higgs_zgamma_width,
)
from .offshell import (
    offshell_scalar_vv_width, scalar_offshell_vv_width, scalar_vv_s12_integral,
    scalar_vv_squared,
)
from .scattering import (
    average_factor, cross_section, differential_cross_section,
    ffs_s_channel_squared, ffv_s_channel_squared, forward_backward_asymmetry,
)
from .vertices import DecayVertex, classify_gamma, collect_decay_vertices

__all__ = [
    "Amplitude", "BosonPropagator", "ChainVertex", "DecayCalculator",
    "DecayChannel", "DecayVertex", "Diagram", "DiracParticle",
    "ExternalState", "Leg", "SpinorChain",
    "TwoBodyKinematics", "ThreeBodyKinematics", "TwoToTwoKinematics",
    "amplitude_squared", "assert_epsilon_single_vanishes", "average_factor",
    "breit_wigner", "classify_gamma", "collect_decay_vertices",
    "cross_section", "differential_cross_section",
    "A_half", "A_one", "contract_to_dots", "dalitz_integral", "dirac_trace",
    "epsilon_pair_tensor", "expand_particles", "ffs_squared",
    "ffs_s_channel_squared", "ffv_squared", "ffv_s_channel_squared",
    "forward_backward_asymmetry", "gamma5_trace_coefficient",
    "higgs_gammagamma_width", "higgs_gg_width", "higgs_zgamma_width",
    "have_scipy", "is_allowed", "kallen",
    "offshell_scalar_vv_width", "partial_width", "polarization_sum",
    "propagator_denominator",
    "reduce_projectors", "scalar_offshell_vv_width", "scalar_propagator",
    "scalar_vv_s12_integral", "scalar_vv_squared",
    "spin_sum", "sss_squared", "two_body_momentum", "two_body_phase_space",
    "vector_propagator", "vector_propagator_numerator", "vvs_squared",
]
