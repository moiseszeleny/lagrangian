"""Two- and three-body phase space kinematics."""

from dataclasses import dataclass

import sympy as sp

from .lorentz import momentum

__all__ = [
    "ThreeBodyKinematics", "TwoBodyKinematics", "is_allowed", "kallen",
    "two_body_momentum", "two_body_phase_space",
]


def kallen(x, y, z):
    """The Källén triangle function ``λ(x,y,z) = x²+y²+z²−2xy−2yz−2zx``."""
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*z*x


def two_body_momentum(M_parent, m1, m2):
    """Magnitude of either daughter's three-momentum in the parent rest frame.

    ``|k| = √λ(M², m₁², m₂²) / 2M``.
    """
    return sp.sqrt(kallen(M_parent**2, m1**2, m2**2)) / (2 * M_parent)


def two_body_phase_space(M_parent, m1, m2):
    """Two-body phase space integration factor.

    ``Γ = ⟨|M|²⟩ × √λ(M², m₁², m₂²)/(16 π M³)`` for a spin-averaged
    ``⟨|M|²⟩`` — i.e. everything in ``Γ`` except the amplitude.

    Args:
        M_parent: Mass of the decaying particle.
        m1: Mass of first daughter.
        m2: Mass of second daughter.

    Returns:
        The kinematic factor integrated over phase space.
    """
    return sp.sqrt(kallen(M_parent**2, m1**2, m2**2)) / (16 * sp.pi * M_parent**3)


def is_allowed(M_parent, m1, m2):
    """Whether ``M_parent ≥ m1 + m2``, when that is decidable.

    Returns ``True``/``False`` for a decidable comparison and ``None`` when the
    masses are symbolic and the ordering is unknown — callers treat ``None`` as
    "keep the channel, numbers get substituted later".
    """
    gap = sp.simplify(sp.sympify(M_parent) - sp.sympify(m1) - sp.sympify(m2))
    if gap.is_number:
        return bool(gap >= 0)
    if gap.is_nonnegative:
        return True
    if gap.is_negative:
        return False
    return None


@dataclass(frozen=True)
class TwoBodyKinematics:
    """On-shell kinematics of ``parent(P) → 1(p₁) + 2(p₂)``.

    Holds the momentum tensor heads the trace engine contracts over, and the
    on-shell dot products they reduce to::

        p₁·p₁ = m₁²,  p₂·p₂ = m₂²,  p₁·p₂ = (M² − m₁² − m₂²)/2

    the last from ``M² = (p₁+p₂)² = m₁² + m₂² + 2 p₁·p₂``.
    """

    M: sp.Expr
    m1: sp.Expr
    m2: sp.Expr
    name1: str = "p1"
    name2: str = "p2"

    @property
    def p1(self):
        return momentum(self.name1)

    @property
    def p2(self):
        return momentum(self.name2)

    @property
    def beta(self):
        """``β = √λ/M²`` — the two-body velocity factor."""
        return sp.sqrt(kallen(self.M**2, self.m1**2, self.m2**2)) / self.M**2

    def dot(self, head_a, head_b):
        """``p_a·p_b`` for two momentum heads.

        Signature required by
        :func:`~feynlag.pheno.lorentz.contract_to_dots`.
        """
        pair = tuple(sorted((head_a.name, head_b.name)))
        table = {
            (self.name1, self.name1): self.m1**2,
            (self.name2, self.name2): self.m2**2,
            tuple(sorted((self.name1, self.name2))):
                (self.M**2 - self.m1**2 - self.m2**2) / 2,
        }
        if pair not in table:
            raise KeyError(f"no on-shell dot product for momenta {pair}")
        return table[pair]

    def phase_space(self):
        """The ``√λ/(16πM³)`` factor for this configuration."""
        return two_body_phase_space(self.M, self.m1, self.m2)

    def allowed(self):
        """``M ≥ m1 + m2`` (``None`` if undecidable symbolically)."""
        return is_allowed(self.M, self.m1, self.m2)


@dataclass(frozen=True)
class ThreeBodyKinematics:
    """Dalitz kinematics of ``parent(M) → 1(p₁) + 2(p₂) + 3(p₃)``.

    Parametrised by the two Dalitz invariants ``s₁₂ = (p₁+p₂)²`` and
    ``s₂₃ = (p₂+p₃)²`` (the third, ``s₁₃``, is fixed by
    ``s₁₂+s₁₃+s₂₃ = M²+m₁²+m₂²+m₃²``).  ``s₂₃`` doubles as the invariant mass²
    of the ``(2,3)`` subsystem — the off-shell ``q²`` of a resonance decaying to
    particles 2 and 3.

    The width follows the PDG form
    ``Γ = 1/((2π)³ 32 M³) ∬ |M|² ds₁₂ ds₂₃`` over the physical Dalitz region.
    """

    M: sp.Expr
    m1: sp.Expr
    m2: sp.Expr
    m3: sp.Expr
    names: tuple = ("p1", "p2", "p3")

    @property
    def p1(self):
        return momentum(self.names[0])

    @property
    def p2(self):
        return momentum(self.names[1])

    @property
    def p3(self):
        return momentum(self.names[2])

    @property
    def s12(self):
        return sp.Symbol("s12", positive=True)

    @property
    def s23(self):
        return sp.Symbol("s23", positive=True)

    def dot(self, head_a, head_b):
        """``p_a·p_b`` as a function of the Dalitz invariants ``s₁₂, s₂₃``.

        Signature required by
        :func:`~feynlag.pheno.lorentz.contract_to_dots`.  Uses
        ``pᵢ·pⱼ = (sᵢⱼ − mᵢ² − mⱼ²)/2`` with ``s₁₃`` from the invariant sum
        rule.
        """
        n1, n2, n3 = self.names
        m1, m2, m3, M = self.m1, self.m2, self.m3, self.M
        s12, s23 = self.s12, self.s23
        s13 = M**2 + m1**2 + m2**2 + m3**2 - s12 - s23
        table = {
            (n1, n1): m1**2, (n2, n2): m2**2, (n3, n3): m3**2,
            tuple(sorted((n1, n2))): (s12 - m1**2 - m2**2) / 2,
            tuple(sorted((n2, n3))): (s23 - m2**2 - m3**2) / 2,
            tuple(sorted((n1, n3))): (s13 - m1**2 - m3**2) / 2,
        }
        pair = tuple(sorted((head_a.name, head_b.name)))
        if pair not in table:
            raise KeyError(f"no Dalitz dot product for momenta {pair}")
        return table[pair]

    def s23_range(self):
        """The outer ``s₂₃`` bounds: ``[(m₂+m₃)², (M−m₁)²]``."""
        return ((self.m2 + self.m3)**2, (self.M - self.m1)**2)

    def s12_bounds(self, s23_val):
        """``(s₁₂⁻, s₁₂⁺)`` at a fixed numeric ``s₂₃`` (PDG Eq. 47.23).

        Evaluated in the ``(2,3)`` rest frame: particles 2 and 3 have energies
        ``E₂*, E₃*`` and particle 1 energy ``E₁*``, and ``s₁₂ = (p₁+p₂)²``
        ranges as the 1–2 opening angle sweeps ``[-1, 1]``.
        """
        s23 = sp.sympify(s23_val)
        M, m1, m2, m3 = self.M, self.m1, self.m2, self.m3
        rs = sp.sqrt(s23)
        E2 = (s23 - m3**2 + m2**2) / (2 * rs)
        E1 = (M**2 - s23 - m1**2) / (2 * rs)
        p2 = sp.sqrt(E2**2 - m2**2)
        p1 = sp.sqrt(E1**2 - m1**2)
        base = (E1 + E2)**2
        return (base - (p1 + p2)**2, base - (p1 - p2)**2)

    def phase_space_constant(self):
        """The ``1/((2π)³ 32 M³)`` prefactor of the Dalitz integral."""
        return 1 / ((2 * sp.pi)**3 * 32 * self.M**3)
