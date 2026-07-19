"""Cross-section round-trip: feynlag's SM UFO vs MadGraph's stock ``sm``.

The flagship correctness check — export the feynlag SM UFO, import it into
MadGraph, compute real cross sections, and confirm they match the standard
model shipped with MadGraph at the same parameter point. Two processes:

- ``e+ e- > mu+ mu-``  — the A/Z FFV couplings and the Z propagator;
- ``e+ e- > w+ w-``    — the gauge-cancellation acid test: the ν t-channel and
  γ/Z s-channel diagrams must cancel, checking relative signs across FFV and
  the WWγ/WWZ triple-gauge vertices.

Not run in CI (each launch compiles Fortran). Run locally:

    python scripts/madgraph_roundtrip.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

MG_HOME = Path(os.environ.get("MG5_HOME", Path.home() / ".local" / "mg5dl"))
MG_URLS = [
    ("MG5_aMC_v3_7_2", "https://launchpad.net/mg5amcnlo/3.0/3.7.x/+download/"
                       "MG5_aMC_v3.7.2.tar.gz"),
    ("MG5_aMC_v3_5_16", "https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/"
                        "MG5_aMC_v3.5.16.tar.gz"),
]

BEAM = 100.0  # GeV per beam → √s = 200 GeV
NEVENTS = 8000

PROCESSES = {
    "ee_mumu": "e+ e- > mu+ mu-",
    "ee_ww": "e+ e- > w+ w-",
}


def ensure_mg5():
    """Return the path to a working ``bin/mg5_aMC`` (downloading if needed)."""
    for name, _ in MG_URLS:
        exe = MG_HOME / name / "bin" / "mg5_aMC"
        if exe.exists():
            return exe
    MG_HOME.mkdir(parents=True, exist_ok=True)
    for name, url in MG_URLS:
        tarball = MG_HOME / f"{name}.tar.gz"
        print(f"downloading {url} …")
        try:
            urllib.request.urlretrieve(url, tarball)
            subprocess.run(["tar", "xzf", str(tarball)], cwd=MG_HOME,
                           check=True)
            exe = next(MG_HOME.glob("MG5_aMC_v*/bin/mg5_aMC"))
            r = subprocess.run([sys.executable, str(exe), "--help"],
                               capture_output=True, timeout=120)
            if r.returncode == 0 or b"mg5_aMC" in r.stdout + r.stderr:
                return exe
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} failed: {exc}")
    raise RuntimeError(
        "could not obtain a working MadGraph. Try a Python 3.11 env:\n"
        "  conda create -n mg5 python=3.11 && conda activate mg5")


def run_mg5(exe, commands):
    """Run an mg5 command script, returning combined stdout/stderr."""
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as f:
        f.write(commands)
        cmd_file = f.name
    try:
        r = subprocess.run([sys.executable, str(exe), "-f", cmd_file],
                           cwd=exe.parent.parent, capture_output=True,
                           text=True, timeout=1800)
        return r.stdout + r.stderr
    finally:
        os.unlink(cmd_file)


_XSEC = re.compile(r"Cross-section\s*:\s*([\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)")


def parse_xsec(run_dir):
    """(value, error) in pb from a completed madevent run."""
    # newest run's results
    candidates = sorted(Path(run_dir).glob("Events/run_*/*.txt")) + \
        sorted(Path(run_dir).glob("HTML/run_*/results.html"))
    text = ""
    banner = sorted(Path(run_dir).glob("Events/run_*/*banner*.txt"))
    for p in banner + candidates:
        text += p.read_text(errors="ignore")
    crossx = Path(run_dir) / "crossx.html"
    if crossx.exists():
        text += crossx.read_text(errors="ignore")
    m = _XSEC.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # fallback: parse the run's results.dat
    for res in Path(run_dir).glob("SubProcesses/**/results.dat"):
        parts = res.read_text().split()
        if parts:
            return float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0
    return None, None


def launch_block(model, process, out_dir):
    return (
        f"import model {model}\n"
        f"generate {process}\n"
        f"output {out_dir}\n"
        f"launch\n"
        f"set lpp1 0\nset lpp2 0\n"
        f"set ebeam1 {BEAM}\nset ebeam2 {BEAM}\n"
        f"set nevents {NEVENTS}\n"
        f"set use_syst False\n"
        f"0\n"
    )


def main():
    exe = ensure_mg5()
    print(f"MadGraph: {exe}")

    # export the feynlag SM UFO
    sys.path.insert(0, str(REPO / "scripts"))
    import export_sm_ufo
    ufo_dir = str(REPO / "scripts" / "FEYNLAG_SM_UFO")
    export_sm_ufo.export(ufo_dir)
    print(f"exported feynlag UFO → {ufo_dir}")

    models = {"feynlag": ufo_dir, "stock_sm": "sm"}
    work = Path(tempfile.mkdtemp(prefix="mg_roundtrip_"))
    results = {}
    for pkey, proc in PROCESSES.items():
        for mkey, model in models.items():
            out = work / f"{pkey}_{mkey}"
            print(f"running {mkey}: {proc} …")
            log = run_mg5(exe, launch_block(model, proc, str(out)))
            val, err = parse_xsec(out)
            results.setdefault(pkey, {})[mkey] = dict(xsec=val, err=err)
            if val is None:
                print(f"  {mkey}: NO CROSS SECTION PARSED")
                (work / f"{pkey}_{mkey}.log").write_text(log)
            else:
                print(f"  {mkey}: {val:.6g} +- {err:.2g} pb")

    # compare
    print("\n=== cross-section comparison ===")
    ok = True
    for pkey, bymodel in results.items():
        fl, st = bymodel.get("feynlag"), bymodel.get("stock_sm")
        if not (fl and st and fl["xsec"] and st["xsec"]):
            print(f"{pkey}: incomplete — see logs in {work}")
            ok = False
            continue
        diff = abs(fl["xsec"] - st["xsec"])
        sigma = (fl["err"]**2 + st["err"]**2) ** 0.5 or 1e-12
        rel = diff / st["xsec"]
        verdict = "MATCH" if diff <= 3 * sigma or rel < 0.01 else "MISMATCH"
        ok = ok and verdict == "MATCH"
        print(f"{pkey}: feynlag {fl['xsec']:.5g} vs stock {st['xsec']:.5g} pb "
              f"| Δ={rel*100:.2f}% ({diff/sigma:.1f}σ) → {verdict}")

    out_json = HERE / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
