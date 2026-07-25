"""2→2 differential and total cross sections: averaging + Tier-1 assemblers.

The spin/colour **averaging** factor lives here — never inside a squared-
amplitude function.  :func:`~feynlag.pheno.amplitudes.ffv_squared`'s
hardcoded ``/3`` (a decay parent's polarization average baked into the
Lorentz algebra) is exactly the anti-pattern this module avoids: it made that
function unusable the moment the vector is *internal* rather than the
parent.  Here, :func:`~feynlag.pheno.diagrams.Amplitude.squared` always
returns the fully spin/colour-**summed** ``|M|²``; :func:`average_factor`
computes the averaging denominator from declared
:class:`~feynlag.pheno.particles.ExternalState` degrees of freedom, and
:func:`cross_section`/:func:`differential_cross_section` apply it exactly
once.
"""

import sympy as sp

from .diagrams import (
    Amplitude, BosonPropagator, ChainVertex, Diagram, Leg, SpinorChain,
    _chain_indices,
)

__all__ = [
    "average_factor", "cross_section", "differential_cross_section",
    "ffs_s_channel_squared", "ffv_s_channel_squared",
]


def average_factor(initial):
    """``1/∏ dof`` over the two initial :class:`~feynlag.pheno.particles.ExternalState`\\ s.

    Args:
        initial: a 2-tuple/list of :class:`~feynlag.pheno.particles.ExternalState`.
    """
    a, b = initial
    return sp.Rational(1, a.dof() * b.dof())


def differential_cross_section(m2_summed, kin, initial, variable="t",
                               symmetry_factor=1):
    """``dσ/dt`` (default) or ``dσ/dcosθ`` from the spin/colour-**summed**
    ``|M|²``.

    Args:
        m2_summed: ``Σ|M|²`` (e.g. from
            :func:`~feynlag.pheno.diagrams.Amplitude.squared`).
        kin: a :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`.
        initial: the ``(state_1, state_2)`` pair passed to
            :func:`average_factor`.
        variable: ``"t"`` or ``"cos"``.
        symmetry_factor: ``1/2`` for identical final-state particles,
            ``1`` otherwise — applied here, never inside ``m2_summed``
            (mirrors :class:`~feynlag.pheno.calculator.DecayChannel`'s
            ``symmetry_factor``).
    """
    avg = average_factor(initial)
    if variable == "t":
        factor = kin.dsigma_dt_factor()
    elif variable == "cos":
        factor = kin.dsigma_dcos_factor()
    else:
        raise ValueError(f"differential_cross_section: variable must be "
                         f"'t' or 'cos', got {variable!r}")
    return sp.expand(symmetry_factor * avg * m2_summed * factor)


def cross_section(m2_summed, kin, initial, symmetry_factor=1):
    """``σ = ∫ dσ/dt dt`` over :meth:`~feynlag.pheno.kinematics.TwoToTwoKinematics.t_bounds`
    (symbolic)."""
    dsdt = differential_cross_section(m2_summed, kin, initial, variable="t",
                                      symmetry_factor=symmetry_factor)
    t_min, t_max = kin.t_bounds()
    return sp.integrate(dsdt, (kin.t, t_min, t_max))


def _s_channel_amplitude(g_in_left, g_in_right, g_out_left, g_out_right, kin,
                         mediator_spin, mediator_mass, mediator_width=0):
    """Shared assembler: ``1(k₁) 2(k₂) → mediator(s-channel) → 3(k₃) 4(k₄)``.

    Leg roles follow the standard external-fermion-line table: the incoming
    pair's bar/field slots are ``(k2, k1)`` (incoming antiparticle/particle),
    the outgoing pair's are ``(k3, k4)`` (outgoing particle/antiparticle) —
    see :class:`~feynlag.pheno.diagrams.Leg`.
    """
    if mediator_spin == 1:
        structure = "V"
        in_indices, in_bar_indices = _chain_indices("in")
        out_indices, out_bar_indices = _chain_indices("out")
        in_indices, in_bar_indices = (in_indices,), (in_bar_indices,)
        out_indices, out_bar_indices = (out_indices,), (out_bar_indices,)
    elif mediator_spin == 0:
        structure = "S"
        in_indices = in_bar_indices = out_indices = out_bar_indices = ()
    else:
        raise NotImplementedError(
            f"_s_channel_amplitude: unsupported mediator_spin {mediator_spin}")

    chain_in = SpinorChain(
        out=Leg(kin.k2, kin.m2, anti=True),
        inn=Leg(kin.k1, kin.m1, anti=False),
        vertices=(ChainVertex(structure, g_in_left, g_in_right),),
        indices=in_indices, bar_indices=in_bar_indices)
    chain_out = SpinorChain(
        out=Leg(kin.k3, kin.m3, anti=False),
        inn=Leg(kin.k4, kin.m4, anti=True),
        vertices=(ChainVertex(structure, g_out_left, g_out_right),),
        indices=out_indices, bar_indices=out_bar_indices)
    prop = BosonPropagator(
        spin=mediator_spin, q2=kin.s, mass=mediator_mass, width=mediator_width,
        momentum=(lambda i: kin.k1(i) + kin.k2(i)) if mediator_spin == 1 else None)
    return Amplitude(diagrams=(
        Diagram(chains=(chain_in, chain_out), propagators=(prop,)),
    ))


def ffv_s_channel_squared(g_in_left, g_in_right, g_out_left, g_out_right,
                          kin, mediator_mass, mediator_width=0):
    """``Σ|M|²`` for ``f₁ f̄₁ → f₂ f̄₂`` through one s-channel vector.

    The Tier-1 worked assembler, in the low-level, coupling-in/number-out
    style of :func:`~feynlag.pheno.offshell.scalar_offshell_vv_width`.  Raises
    (via :meth:`~feynlag.pheno.diagrams.Amplitude.squared`) if the chiral
    couplings give a non-zero ε contribution — e.g. a purely chiral current
    like a bare ``Z`` coupling; a pure-vector (QED) coupling
    (``g_left == g_right``) never trips it.
    """
    amp = _s_channel_amplitude(g_in_left, g_in_right, g_out_left, g_out_right,
                               kin, mediator_spin=1, mediator_mass=mediator_mass,
                               mediator_width=mediator_width)
    return amp.squared(kin)


def ffs_s_channel_squared(g_in_left, g_in_right, g_out_left, g_out_right,
                          kin, mediator_mass, mediator_width=0):
    """``Σ|M|²`` for ``f₁ f̄₁ → f₂ f̄₂`` through one s-channel scalar.

    Always ε-free (a scalar chain has at most two γ's per line — see
    :meth:`~feynlag.pheno.diagrams.ChainVertex.epsilon_coefficient`).
    """
    amp = _s_channel_amplitude(g_in_left, g_in_right, g_out_left, g_out_right,
                               kin, mediator_spin=0, mediator_mass=mediator_mass,
                               mediator_width=mediator_width)
    return amp.squared(kin)
