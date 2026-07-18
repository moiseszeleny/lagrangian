"""Muon-decay width: feynlag's Fermi four-fermion UFO vs the analytic formula.

The end-to-end correctness check for the four-fermion (FFFF) track — the acid
test that the exported ``FFFF*`` Lorentz strings and the fermion-flow leg
assignment are physically right, which the unit tests (symbolic extraction +
coupling round-trip) cannot see.  We export a self-contained Fermi UFO for

    L ⊃ −(4 G_F/√2) (ν̄_μ γ^μ P_L μ)(ē γ_μ P_L ν_e) + h.c.

and ask MadGraph for the muon partial width Γ(μ⁻ → e⁻ ν̄_e ν_μ), then compare to
the textbook result (m_e, m_ν → 0)

    Γ = G_F² m_μ⁵ / (192 π³)   ≈ 3.0 × 10⁻¹⁹ GeV   (⇔ τ_μ ≈ 2.2 µs).

**Both the operator vertex and its Hermitian conjugate must be exported.** A
Hermitian Lagrangian's ``op + h.c.`` produces two four-fermion vertices (two
distinct bilinear keys — see ``vertices/bilinear.py``), and MadGraph needs both
to establish a consistent fermion-number flow through the contact interaction:
a model carrying only one of the pair fails diagram generation
(``NoDiagramException``).  The exported ``FFFF*`` Lorentz string itself is the
same one MadGraph's shipped ``taudecay_UFO`` uses for τ decay, verbatim.

Not run in CI (each launch compiles Fortran, ~minutes).  Run locally:

    python scripts/madgraph_fermi.py

Reuses the MadGraph plumbing (download/locate, command runner) from
``madgraph_roundtrip.py``.
"""

import math
import re
import sys
import tempfile
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from madgraph_roundtrip import ensure_mg5, run_mg5   # noqa: E402

from feynlag import ExternalParameter, ParameterSet   # noqa: E402
from feynlag.export.ufo import UFOParticle, write_ufo  # noqa: E402

GF_VALUE = 1.1663787e-5     # GeV^-2
MMU_VALUE = 0.1056584       # GeV


def analytic_width():
    """Γ = G_F² m_μ⁵ / (192 π³) in the massless-daughter limit."""
    return GF_VALUE**2 * MMU_VALUE**5 / (192 * math.pi**3)


def export_fermi_ufo(path):
    """Write the Fermi-theory UFO (muon massive, e/ν massless).

    Exports **both** the operator vertex and its h.c. conjugate — MadGraph
    needs the pair for fermion-flow through the contact interaction.
    """
    GF = ExternalParameter("GF", GF_VALUE, positive=True, unit_dim=-2)
    MMU = ExternalParameter("MMU", MMU_VALUE, positive=True, unit_dim=1)
    WMU = ExternalParameter("WMU", 3.0e-19, positive=True, unit_dim=1)  # seed

    numu, numub, mm, mp, em, ep, nue, nueb = sp.symbols(
        "numu numub mm mp em ep nue nueb")
    particles = [
        UFOParticle(numu, 14, "vm", antiname="vm~", antisymbol=numub, spin=2),
        UFOParticle(mm, 13, "mu-", antiname="mu+", antisymbol=mp, spin=2,
                    mass="MMU", width="WMU", charge=-1),
        UFOParticle(em, 11, "e-", antiname="e+", antisymbol=ep, spin=2,
                    charge=-1),
        UFOParticle(nue, 12, "ve", antiname="ve~", antisymbol=nueb, spin=2),
    ]
    params = ParameterSet(GF, MMU, WMU)
    c = -4 * GF.s / sp.sqrt(2)
    # bar = antiparticle of the ψ̄ field, field = the ψ (validated SM
    # charged-current convention).  Vertex + its Hermitian conjugate:
    four_fermion = [
        # (ν̄_μ γ^μ P_L μ)(ē γ_μ P_L ν_e)
        {"bar1": numub, "field1": mm, "bar2": ep, "field2": nue,
         "couplings": {("VL", "VL"): c}},
        # h.c.: (μ̄ γ^μ P_L ν_μ)(ν̄_e γ_μ P_L e)
        {"bar1": mp, "field1": numu, "bar2": nueb, "field2": em,
         "couplings": {("VL", "VL"): c}},
    ]
    return write_ufo(path, "Fermi", params, particles,
                     four_fermion_vertices=four_fermion)


_WIDTH = re.compile(r"Width\s*:\s*([\d.eE+-]+)")


def compute_width(exe, ufo_dir, work):
    """Run MadGraph on μ⁻ → e⁻ ν̄_e ν_μ and return the width (GeV)."""
    out = work / "mudecay"
    commands = (
        f"import model {ufo_dir}\n"
        f"generate mu- > e- ve~ vm\n"
        f"output {out}\n"
        f"launch\n"
        f"0\n"
    )
    log = run_mg5(exe, commands)
    (work / "mg.log").write_text(log)
    matches = _WIDTH.findall(log)
    if not matches:
        return None, log
    return float(matches[-1]), log


def main():
    exe = ensure_mg5()
    print(f"MadGraph: {exe}")

    work = Path(tempfile.mkdtemp(prefix="mg_fermi_"))
    ufo_dir = work / "Fermi_UFO"
    export_fermi_ufo(ufo_dir)
    print(f"exported Fermi UFO → {ufo_dir}")

    analytic = analytic_width()
    print(f"analytic  Γ = G_F² m_μ⁵/192π³ = {analytic:.6e} GeV "
          f"(τ = {6.582119e-25 / analytic * 1e6:.3f} µs)")

    width, log = compute_width(exe, str(ufo_dir), work)
    if width is None:
        print("MadGraph did not report a width — see", work / "mg.log")
        sys.exit(1)

    rel = abs(width - analytic) / analytic
    # ~1% residual is MadEvent's default generation setup, not a physics error
    verdict = "MATCH" if rel < 0.03 else "MISMATCH"
    print(f"MadGraph  Γ = {width:.6e} GeV "
          f"(τ = {6.582119e-25 / width * 1e6:.3f} µs)")
    print(f"ratio MG/analytic = {width / analytic:.4f}  "
          f"(Δ = {rel * 100:.2f}%) → {verdict}")
    sys.exit(0 if verdict == "MATCH" else 1)


if __name__ == "__main__":
    main()
