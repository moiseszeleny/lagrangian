"""Two-body phase space kinematics."""

from dataclasses import dataclass

import sympy as sp

from .lorentz import momentum

__all__ = [
    "TwoBodyKinematics", "is_allowed", "kallen", "two_body_momentum",
    "two_body_phase_space",
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
