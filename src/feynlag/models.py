"""Reusable Standard Model building blocks.

The electroweak scaffold — the SU(2)×U(1) gauge groups, the Higgs doublet with
its potential and covariant kinetic term, and the physical-basis rotations
(Weinberg angle → Z/γ, then the W± unitary combination) — is identical across
every SM-based example in this repository.  This module factors it out so a
worked model can say *what it is about* (a seesaw, a vector-like lepton, a Z′)
without re-deriving the electroweak boilerplate.

Two levels of API:

- **Composable primitives** — :func:`electroweak_gauge`, :func:`higgs_doublet`,
  :func:`weinberg_rotation`, :func:`charged_current_rotation` — for models that
  reuse *some* of the scaffold but diverge (e.g. a U(1)_X extension whose
  Weinberg step produces an intermediate ``Z0`` that then mixes with a Z′).
- **Bundles** — :func:`electroweak_scaffold` returns an
  :class:`ElectroweakScaffold` (groups + Higgs + parameters), and
  :func:`to_physical_basis` applies the standard two rotations and returns a
  :class:`PhysicalBasis`.  :func:`standard_model` composes everything for the
  plain EW+leptons case.

This mirrors the precedent set by :func:`~feynlag.flavor.standard_ckm`: the
library ships the *standard* construction, and non-standard models compose the
pieces.
"""

from dataclasses import dataclass, field

import sympy as sp

from .fields import Scalar, WeylFermion, conjugate_pair
from .groups import SU2, U1
from .lagrangian import Lagrangian, Model
from .operators import Dmu
from .parameters import ExternalParameter, InternalParameter
from .vacuum import Rotation, rotation_2x2
from .vertices.bilinear import fermion_gauge_current

__all__ = [
    "ElectroweakScaffold", "PhysicalBasis", "StandardModel",
    "charged_current_rotation", "electroweak_gauge", "electroweak_scaffold",
    "higgs_doublet", "standard_model", "to_physical_basis",
    "weinberg_rotation",
]

# PDG-ish default parameter point (also used by the examples/tutorials).
GW_DEFAULT, G1_DEFAULT = 0.6535, 0.3580
VEV_DEFAULT, MH_DEFAULT = 246.0, 125.25


# --------------------------------------------------------------- primitives

def electroweak_gauge(gw=GW_DEFAULT, g1=G1_DEFAULT, names=("SU2L", "U1Y")):
    """The electroweak gauge groups and their coupling parameters.

    Returns:
        ``(SU2L, U1Y, gw_param, g1_param)`` — the two :class:`GaugeGroup`\\ s
        and their ``ExternalParameter`` couplings.
    """
    gw_p = ExternalParameter("gw", gw, positive=True)
    g1_p = ExternalParameter("g1", g1, positive=True)
    SU2L = SU2(names[0], coupling=gw_p)
    U1Y = U1(names[1], coupling=g1_p)
    return SU2L, U1Y, gw_p, g1_p


def higgs_doublet(SU2L, U1Y, v=VEV_DEFAULT, mh=MH_DEFAULT, lam=None,
                  extra_reps=None, name="H"):
    """The SM Higgs doublet with its VEV expanded and potential parameters.

    Args:
        SU2L, U1Y: the electroweak gauge groups.
        v: the VEV (numeric default for ``ExternalParameter`` ``v``).
        mh: Higgs mass, used to fix ``lam = mh²/(2v²)`` when ``lam`` is not
            given.
        lam: an explicit quartic value; overrides the ``mh``-derived default
            (some models pin ``λ`` directly rather than through ``m_h``).
        extra_reps: additional ``{group: rep}`` charges to merge into the
            doublet (e.g. a U(1)_X charge for a Z′ model).
        name: field name.

    Returns:
        ``(H, v_param, lam_param, mu2_param)``.  ``mu2`` is an
        :class:`InternalParameter` (fixed later by the tadpole condition).
    """
    reps = {SU2L: 2, U1Y: sp.Rational(1, 2)}
    if extra_reps:
        reps.update(extra_reps)
    v_p = ExternalParameter("v", v, positive=True, unit_dim=1)
    lam_val = lam if lam is not None else mh**2 / (2 * v**2)
    lam_p = ExternalParameter("lam", lam_val)
    mu2_p = InternalParameter("mu2", unit_dim=2)
    H = Scalar(name, reps=reps, component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v_p})
    return H, v_p, lam_p, mu2_p


def higgs_lagrangian(H, lam, mu2):
    """The Higgs kinetic + potential terms, as ``{sector: expr}``.

    ``L ⊃ (D_μH)†(D^μH) − V``, ``V = −μ² H†H + λ (H†H)²``.  The library stores
    ``L ⊃ −V`` in the ``potential`` sector.
    """
    from .fields import dag
    HdH = (dag(H) * H.mat)[0]
    DH = Dmu(H)
    return {
        "kinetic": (dag(DH) * DH)[0],
        "potential": -(-mu2.s * HdH + lam.s * HdH**2),
    }


def weinberg_rotation(model, SU2L, U1Y, z="Z", a="A"):
    """Register the ``W³,B → Z,γ`` Weinberg rotation on ``model``.

    Args:
        z, a: output symbol names (``z`` is overridable — a chained model, e.g.
            U(1)_X, names the intermediate neutral state ``"Z0"`` because it
            still mixes with a Z′ in a following rotation).

    Returns:
        ``(Z, A)`` symbols.
    """
    W3 = SU2L.bosons().components[2]
    B = U1Y.bosons().components[0]
    Z, A = sp.symbols(f"{z} {a}", real=True)
    theta_w = sp.atan(U1Y.coupling.s / SU2L.coupling.s)
    model.rotate(Rotation([W3, B], [Z, A], rotation_2x2(-theta_w)))
    return Z, A


def charged_current_rotation(model, SU2L, wp="Wp", wm="Wm"):
    """Register the ``W¹,W² → W⁺,W⁻`` unitary rotation on ``model``.

    ``W± = (W¹ ∓ i W²)/√2``.

    Returns:
        ``(Wp, Wm)`` symbols.
    """
    W1, W2 = SU2L.bosons().components[0], SU2L.bosons().components[1]
    Wp, Wm = sp.symbols(f"{wp} {wm}")
    U = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
    model.rotate(Rotation([W1, W2], [Wp, Wm], U, kind="unitary"))
    return Wp, Wm


# ------------------------------------------------------------------- bundles

@dataclass
class ElectroweakScaffold:
    """The SM electroweak scaffold: gauge groups, Higgs, and core parameters.

    Attributes:
        SU2L, U1Y: the gauge groups.
        gw, g1: their coupling ``ExternalParameter``\\ s.
        H: the Higgs doublet (VEV already expanded).
        v, lam, mu2: the Higgs-sector parameters.
        W, B: the SU(2) / U(1) gauge-boson fields.
    """

    SU2L: object
    U1Y: object
    gw: object
    g1: object
    H: object
    v: object
    lam: object
    mu2: object
    W: object
    B: object

    @property
    def gauge_groups(self):
        return [self.SU2L, self.U1Y]

    @property
    def parameters(self):
        """The five core EW parameters, in dependency-safe order."""
        return [self.gw, self.g1, self.v, self.lam, self.mu2]

    @property
    def fields(self):
        """Higgs + the two gauge-boson fields."""
        return [self.H, self.W, self.B]

    def add_higgs(self, L):
        """Add the Higgs kinetic + potential terms to a :class:`Lagrangian`."""
        for sector, expr in higgs_lagrangian(self.H, self.lam, self.mu2).items():
            L.add(expr, sector=sector)
        return L


def electroweak_scaffold(gw=GW_DEFAULT, g1=G1_DEFAULT, v=VEV_DEFAULT,
                         mh=MH_DEFAULT, lam=None, extra_higgs_reps=None):
    """Build an :class:`ElectroweakScaffold` at the given parameter point."""
    SU2L, U1Y, gw_p, g1_p = electroweak_gauge(gw, g1)
    H, v_p, lam_p, mu2_p = higgs_doublet(SU2L, U1Y, v=v, mh=mh, lam=lam,
                                         extra_reps=extra_higgs_reps)
    W, B = SU2L.bosons("W"), U1Y.bosons("B")
    return ElectroweakScaffold(SU2L, U1Y, gw_p, g1_p, H, v_p, lam_p, mu2_p, W, B)


@dataclass
class PhysicalBasis:
    """The electroweak fields after the standard two rotations.

    Attributes:
        Z, A, Wp, Wm: the physical gauge bosons.
        h, G0: the physical CP-even Higgs and the neutral Goldstone (the real /
            imaginary parts of ``H⁰``).
        Gp, Gm: the charged Goldstone and its conjugate symbol.
        cmap: the ``{conjugate(Gp): Gm}`` map for the extractor.
        bosons: a convenience list ``[h, G0, Gp, Gm, Z, A, Wp, Wm]``.
    """

    Z: object
    A: object
    Wp: object
    Wm: object
    h: object
    G0: object
    Gp: object
    Gm: object
    cmap: dict
    bosons: list = field(default_factory=list)


def to_physical_basis(model, scaffold, gm_name="Gm"):
    """Apply the standard EW rotations and wire up the Goldstone bookkeeping.

    Registers the Weinberg (``W³,B → Z,γ``) and charged-current
    (``W¹,W² → W⁺,W⁻``) rotations, then reads the physical Higgs ``h`` and
    neutral Goldstone ``G0`` off the doublet's VEV expansion and builds the
    charged-Goldstone conjugate map.

    Returns:
        :class:`PhysicalBasis`.
    """
    Z, A = weinberg_rotation(model, scaffold.SU2L, scaffold.U1Y)
    Wp, Wm = charged_current_rotation(model, scaffold.SU2L)

    H = scaffold.H
    _, h, G0 = H.vev_expansions[H.components[1]]     # (vev, re, im)
    Gp = H.components[0]
    Gm, cmap = conjugate_pair(Gp, gm_name)
    bosons = [h, G0, Gp, Gm, Z, A, Wp, Wm]
    return PhysicalBasis(Z, A, Wp, Wm, h, G0, Gp, Gm, cmap, bosons)


# ------------------------------------------------------------- convenience

@dataclass
class StandardModel:
    """A fully-built Standard Model and its handles.

    Attributes:
        model: the :class:`~feynlag.lagrangian.Model`.
        scaffold: the :class:`ElectroweakScaffold`.
        physical: the :class:`PhysicalBasis` (``None`` if not rotated).
        leptons: ``[(L_doublet, right_singlet), …]`` per generation.
    """

    model: object
    scaffold: object
    physical: object
    leptons: list = field(default_factory=list)


def standard_model(generations=1, physical_basis=True, name="StandardModel"):
    """The plain SM electroweak + lepton sector, ready to use.

    Builds the electroweak scaffold plus ``generations`` lepton doublets and
    charged right-singlets, each contributing a gauge current, and (by default)
    rotates to the physical basis.  Yukawa couplings are intentionally *not*
    added — a model that needs charged-lepton masses adds its own (see
    ``examples/sm_decays.py``), keeping this builder free of a ``pheno``
    dependency and unopinionated about flavour.

    Returns:
        :class:`StandardModel`.
    """
    scaffold = electroweak_scaffold()
    L = Lagrangian()
    scaffold.add_higgs(L)

    i = sp.Symbol("i", integer=True)
    leptons, lepton_fields, params = [], [], list(scaffold.parameters)
    for g in range(generations):
        suffix = "" if generations == 1 else f"{g + 1}"
        Ll = WeylFermion(f"Ll{suffix}",
                         reps={scaffold.SU2L: 2, scaffold.U1Y: -sp.Rational(1, 2)},
                         chirality="L", nflavors=1,
                         component_names=[f"nuL{suffix}", f"eL{suffix}"])
        eR = WeylFermion(f"eR{suffix}", reps={scaffold.U1Y: -1}, chirality="R",
                         nflavors=1, component_names=[f"eR{suffix}"])
        L.add(fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i),
              sector="gauge")
        leptons.append((Ll, eR))
        lepton_fields += [Ll, eR]

    model = Model(name, gauge_groups=scaffold.gauge_groups,
                  fields=scaffold.fields + lepton_fields,
                  parameters=params, lagrangian=L)
    model.solve_tadpoles([scaffold.mu2])

    physical = to_physical_basis(model, scaffold) if physical_basis else None
    return StandardModel(model, scaffold, physical, leptons)
