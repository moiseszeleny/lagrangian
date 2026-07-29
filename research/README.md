# `research/` — open physics questions

This is the staging area for exploratory work: parameter scans, benchmark hunts,
model extensions whose conclusions are **not known in advance**. It is the fourth
home in the repo, and it exists because the other three all have rules that
exploratory work cannot satisfy.

| home | contents | rules |
|---|---|---|
| `src/feynlag/` | the library | must be general, tested, documented |
| `tests/` | pinned physics values | must be fast and deterministic |
| `examples/` | runnable models + pedagogical notebooks | must always run; conclusions known in advance; symlinked into `docs/tutorials/` |
| `scripts/` | external-tool runs (MadGraph) | not in CI; results land in `docs/*.md` + a committed JSON |
| **`research/`** | **open questions, scans, benchmark hunts** | **may be slow; may be wrong and get revised** |

## Rules

1. **Research notebooks are not documentation.** Do not symlink them into
   `docs/tutorials/` and do not add them to the Sphinx toctree. `docs/conf.py`
   sets `nb_execution_mode = "off"`, so Sphinx serves *stored* outputs — a stray
   symlink would silently publish a half-finished scan as if it were a lesson.

2. **Notebook hygiene is automatic.** `.gitattributes` applies the `nbstripout`
   filter to `*.ipynb` repo-wide with `--keep-output`, so research notebooks keep
   their plots and printed results in git while `execution_count` and volatile
   metadata are normalized away. Nothing to configure per project.

3. **Results graduate.** When a research result stabilizes it leaves `research/`:
   the physics gets pinned in `tests/`, the narrative gets a `docs/manual/`
   chapter, and the reproducible driver moves to `examples/` (if it teaches) or
   `scripts/` (if it needs an external tool). `research/` is the workbench, not
   the archive.

4. **`tests/` never imports `research/`.** Research code is expected to churn;
   a test that depended on it would make the suite hostage to an unfinished
   scan. Tests that pin research-discovered physics build their own model, the
   way `tests/test_thdm_s3.py` already does.

5. **Cite what you take from the literature.** The repo-wide rule applies here
   too: verify the source, then add an inline `[Tag]` plus a `## References`
   list. Each project's `NOTES.md` carries its reference list.

## Layout of a project

```
research/<project>/
  NOTES.md          # running log: open questions, findings, literature targets
  model.py          # thin, importable build — algebra only, no scanning/plotting
  NN_<topic>.ipynb  # the actual research, numbered in the order it was done
  results/          # small committed JSON artifacts (< ~100 kB)
  cache/            # heavy intermediates — gitignored, never committed
```

`research/` is deliberately **not** added to `[tool.setuptools.packages.find]` —
it is not library code and is not installed. Notebooks sit beside their
`model.py` and import it directly (`from model import build_model`), which works
because Jupyter sets the working directory to the notebook's own folder. From
the repo root, use `sys.path.insert(0, "research/<project>")`.

## Projects

- **`thdm_s3/`** — the S₃-symmetric three-Higgs-doublet model: scalar parameter
  space under theory constraints, the S₃ fermion sector, and scalar decays.
  See [`thdm_s3/NOTES.md`](thdm_s3/NOTES.md).
