"""Numeric round-trip verification of an exported UFO model.

The single thing no Lagrangian→Feynman-rule tool has traditionally been able
to do is *prove* that the model it just wrote is correct: FeynRules-style
generators emit thousands of vertices and validation is left to the author
(the well-known irregular-validation problem). feynlag already dual-verifies
every result symbolically and numerically *inside* the pipeline; this module
carries that guarantee across the export boundary by loading the written UFO
directory back in and evaluating it numerically.

:func:`verify_ufo_numeric` re-imports the generated UFO package, resolves the
whole parameter chain (externals from their declared values, internals by
evaluating their `value` expressions in dependency order), and evaluates every
coupling's `value` string. A model that exports cleanly but references an
undefined symbol, emits malformed `pycode`, or divides by zero at the chosen
point is caught here rather than downstream in MadGraph.
"""

import cmath
import importlib
import sys

__all__ = ["UFORoundTripReport", "verify_ufo_numeric"]


class UFORoundTripReport:
    """Result of :func:`verify_ufo_numeric`.

    Attributes:
        parameters: ``{name: complex value}`` for every resolved parameter.
        couplings: ``{name: complex value}`` for every evaluated coupling.
        failures: ``{name: error string}`` for anything that failed to
            resolve or evaluate to a finite number.
    """

    def __init__(self, parameters, couplings, failures):
        self.parameters = parameters
        self.couplings = couplings
        self.failures = failures

    @property
    def ok(self):
        return not self.failures

    def raise_on_failure(self):
        if not self.ok:
            lines = "\n".join(f"  {n}: {e}" for n, e in self.failures.items())
            raise ValueError(
                f"UFO round-trip evaluation failed:\n{lines}")
        return self

    def __repr__(self):
        status = "ok" if self.ok else f"{len(self.failures)} failed"
        return (f"UFORoundTripReport({len(self.parameters)} parameters, "
                f"{len(self.couplings)} couplings, {status})")


def _finite(value):
    z = complex(value)
    if cmath.isnan(z.real) or cmath.isnan(z.imag) \
            or cmath.isinf(z.real) or cmath.isinf(z.imag):
        raise ValueError(f"non-finite value {z}")
    return z


class _UFONamespace:
    """The parameter/coupling registries of a loaded UFO model."""

    def __init__(self, all_parameters, all_couplings):
        self.all_parameters = all_parameters
        self.all_couplings = all_couplings


def _import_ufo(path):
    """Load a UFO directory the way MadGraph does — the directory itself on
    ``sys.path`` and its submodules imported by bare name (UFO modules use
    absolute imports, e.g. ``from object_library import …``).
    """
    path = str(path)
    sys.path.insert(0, path)
    # drop any cached UFO submodules from a previous load (names are generic)
    for mod in ("object_library", "function_library", "parameters",
                "couplings", "particles", "lorentz", "coupling_orders",
                "vertices"):
        sys.modules.pop(mod, None)
    try:
        import object_library  # noqa: F401 — populated by the imports below
        import function_library  # noqa: F401
        import parameters  # noqa: F401 — constructing these fills registries
        import couplings  # noqa: F401
        return _UFONamespace(object_library.all_parameters,
                             object_library.all_couplings)
    finally:
        sys.path.pop(0)


def verify_ufo_numeric(path, external_values=None):
    """Load a written UFO model and evaluate every parameter and coupling.

    Args:
        path: directory written by
            :func:`~feynlag.export.ufo.writer.write_ufo`.
        external_values: optional ``{name: number}`` overriding the external
            parameters' declared numeric values (e.g. to probe a second point).

    Returns:
        :class:`UFORoundTripReport`. ``report.ok`` is ``True`` when every
        parameter resolved and every coupling evaluated to a finite complex
        number.
    """
    external_values = external_values or {}
    ufo = _import_ufo(path)

    namespace = {
        "cmath": cmath,
        "complex": complex,
        "abs": abs,
        "complexconjugate": lambda z: complex(z).conjugate(),
    }
    parameters = {}
    couplings = {}
    failures = {}

    # Resolve parameters in declared order (internals reference earlier ones).
    for p in ufo.all_parameters:
        try:
            if p.nature == "external":
                raw = external_values.get(p.name, p.value)
                val = _finite(raw)
            else:
                val = _finite(eval(p.value, namespace, dict(parameters)))
        except Exception as exc:  # noqa: BLE001 — report, don't crash
            failures[f"parameter {p.name}"] = f"{type(exc).__name__}: {exc}"
            continue
        parameters[p.name] = val
        namespace[p.name] = val

    # Evaluate every coupling value string.
    for c in ufo.all_couplings:
        try:
            couplings[c.name] = _finite(eval(c.value, namespace,
                                             dict(parameters)))
        except Exception as exc:  # noqa: BLE001
            failures[f"coupling {c.name}"] = f"{type(exc).__name__}: {exc}"

    return UFORoundTripReport(parameters, couplings, failures)
