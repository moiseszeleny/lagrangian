"""Extract the stored figures of 01_scalar_parameter_space.ipynb to figures/*.png.

The notebooks are tracked with ``nbstripout --keep-output``, so their figures are
real committed bytes rather than something a re-run has to reproduce.  This pulls
them out so ``report_scalar_sector.tex`` can \\includegraphics them without anyone
re-executing a 2 M-point scan.

The mapping below is by *cell index*, which is brittle if the notebook is
re-organised — so each entry also carries a marker string that must appear in the
producing cell's source.  A shifted notebook fails loudly here instead of
silently putting the wrong plot in the report.

Run from research/thdm_s3/:  python extract_figures.py
"""

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01_scalar_parameter_space.ipynb"
OUTDIR = HERE / "figures"

#: name -> (cell index, marker that must occur in that cell's source)
FIGURES = {
    "bfb_ft_curve": (59, "Eq. (4g) checks one point of a curve"),
    "cut_flow": (72, "Cut flow: S$_3$-3HDM scalar parameter space"),
    "mass_ranges": (76, "Allowed mass ranges after theory constraints"),
    "correlations": (77, "lightest CP-even scalar"),
}


def main():
    nb = json.loads(NOTEBOOK.read_text())
    cells = nb["cells"]
    OUTDIR.mkdir(exist_ok=True)

    for name, (idx, marker) in FIGURES.items():
        cell = cells[idx]
        src = "".join(cell["source"])
        if marker not in src:
            raise SystemExit(
                f"cell {idx} does not contain {marker!r} — the notebook has been "
                f"re-organised and FIGURES needs updating (refusing to write "
                f"{name}.png from the wrong cell)")
        pngs = [o["data"]["image/png"] for o in cell.get("outputs", [])
                if o["output_type"] == "display_data" and "image/png" in o.get("data", {})]
        if len(pngs) != 1:
            raise SystemExit(f"cell {idx} has {len(pngs)} PNG outputs, expected 1")
        raw = base64.b64decode(pngs[0])
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"cell {idx} output is not a PNG")
        (OUTDIR / f"{name}.png").write_bytes(raw)
        print("  %-16s <- cell %-3d  %6.1f KB" % (name + ".png", idx, len(raw) / 1024))

    print("wrote %d figures to %s/" % (len(FIGURES), OUTDIR.name))


if __name__ == "__main__":
    main()
