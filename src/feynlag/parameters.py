"""Model parameters: external (experiment-fixed) vs internal (derived).

- :class:`ExternalParameter`: fixed by experiment (``v``, ``m_h``, ``g``…),
  carries a numeric value for benchmarks / UFO param cards.
- :class:`InternalParameter`: derived from other parameters via a SymPy
  expression (tadpole solutions, mixing angles, inverted quartics).  The
  expression may be assigned *later* by the pipeline.
- :class:`ParameterSet`: dependency DAG over internal parameters with
  topological ordering, full symbolic resolution and numeric evaluation —
  exactly what a UFO ``parameters.py`` needs.
"""

import sympy as sp

__all__ = ["Parameter", "ExternalParameter", "InternalParameter",
           "ParameterSet"]


class Parameter:
    """Wraps a SymPy Symbol with physics metadata.

    Args:
        name: symbol name.
        tex: LaTeX string (defaults to ``name``).
        real: SymPy assumption (default True — most Lagrangian parameters).
        positive: SymPy assumption (use for VEVs, masses; enables ``sqrt``
            simplifications per CONVENTIONS.md).
        unit_dim: mass dimension, for dimensional checks.
    """

    def __init__(self, name, tex=None, real=True, positive=False, unit_dim=0):
        self.name = name
        self.tex = tex if tex is not None else name
        assumptions = {"positive": True} if positive else {"real": True} if real else {}
        self.symbol = sp.Symbol(name, **assumptions)
        self.unit_dim = unit_dim

    @property
    def s(self):
        """The underlying SymPy symbol (shorthand for expressions)."""
        return self.symbol

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"

    # Allow parameters to be used directly in SymPy arithmetic.
    def _sympy_(self):
        return self.symbol

    def __add__(self, other):
        return self.symbol + sp.sympify(other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.symbol - sp.sympify(other)

    def __rsub__(self, other):
        return sp.sympify(other) - self.symbol

    def __mul__(self, other):
        return self.symbol * sp.sympify(other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.symbol / sp.sympify(other)

    def __rtruediv__(self, other):
        return sp.sympify(other) / self.symbol

    def __pow__(self, other):
        return self.symbol ** sp.sympify(other)

    def __neg__(self):
        return -self.symbol


class ExternalParameter(Parameter):
    """Parameter fixed by experiment; input of the model.

    ``value`` may be a float or a SymPy expression in *other external*
    parameters' numeric values (evaluated at benchmark time).
    """

    nature = "external"

    def __init__(self, name, value=None, tex=None, real=True, positive=False,
                 unit_dim=0):
        super().__init__(name, tex=tex, real=real, positive=positive,
                         unit_dim=unit_dim)
        self.value = value


class InternalParameter(Parameter):
    """Parameter derived from others via ``expr`` (assignable later).

    Examples: ``μ₁₁²`` from a tadpole condition, ``λ₁`` inverted from
    ``m_h`` and ``v``, a mixing angle from a ``tan 2θ`` condition.
    """

    nature = "internal"

    def __init__(self, name, expr=None, tex=None, real=True, positive=False,
                 unit_dim=0):
        super().__init__(name, tex=tex, real=real, positive=positive,
                         unit_dim=unit_dim)
        self.expr = sp.sympify(expr) if expr is not None else None

    def define(self, expr):
        """Assign (or reassign) the defining expression."""
        self.expr = sp.sympify(expr)
        return self


class ParameterSet:
    """Collection of parameters with dependency resolution.

    Internal parameters may depend on other internal parameters; the
    dependency graph must be acyclic.  :meth:`dependency_order` returns
    internals in evaluation order (what UFO's ``parameters.py`` requires).
    """

    def __init__(self, *params):
        self._by_name = {}
        self._by_symbol = {}
        self.add(*params)

    def add(self, *params):
        for p in params:
            if p.name in self._by_name and self._by_name[p.name] is not p:
                raise ValueError(f"duplicate parameter name: {p.name}")
            self._by_name[p.name] = p
            self._by_symbol[p.symbol] = p
        return self

    def __getitem__(self, key):
        if isinstance(key, sp.Symbol):
            return self._by_symbol[key]
        return self._by_name[key]

    def __contains__(self, key):
        return key in self._by_name or key in self._by_symbol

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self):
        return len(self._by_name)

    @property
    def externals(self):
        return [p for p in self if isinstance(p, ExternalParameter)]

    @property
    def internals(self):
        return [p for p in self if isinstance(p, InternalParameter)]

    def dependency_order(self):
        """Topologically sort internal parameters by their dependencies.

        Returns the internals in an order where every parameter appears
        after all internal parameters occurring in its ``expr``.

        Raises:
            ValueError: on an undefined internal (``expr is None``), a
                dependency cycle, or a dependency on an unknown symbol.
        """
        internals = self.internals
        internal_symbols = {p.symbol for p in internals}
        external_symbols = {p.symbol for p in self.externals}

        deps = {}
        for p in internals:
            if p.expr is None:
                raise ValueError(f"internal parameter {p.name} has no "
                                 f"defining expression")
            free = p.expr.free_symbols
            unknown = free - internal_symbols - external_symbols
            if unknown:
                raise ValueError(
                    f"internal parameter {p.name} depends on symbols not in "
                    f"the ParameterSet: {sorted(unknown, key=str)}")
            deps[p.symbol] = free & internal_symbols

        ordered, resolved = [], set()
        remaining = dict(deps)
        while remaining:
            ready = [s for s, d in remaining.items() if d <= resolved]
            if not ready:
                cycle = sorted(remaining, key=str)
                raise ValueError(f"dependency cycle among internal "
                                 f"parameters: {cycle}")
            for s in sorted(ready, key=str):
                ordered.append(self._by_symbol[s])
                resolved.add(s)
                del remaining[s]
        return ordered

    def resolve(self):
        """Map every internal symbol to an expression in externals only."""
        resolution = {}
        for p in self.dependency_order():
            resolution[p.symbol] = p.expr.subs(resolution)
        return resolution

    def numeric(self):
        """Numeric value of every parameter (externals must have values).

        Returns:
            dict ``{symbol: float/complex}``.
        """
        values = {}
        for p in self.externals:
            if p.value is None:
                raise ValueError(f"external parameter {p.name} has no value")
            values[p.symbol] = sp.sympify(p.value)
        # substitute progressively so external expressions of externals work
        for s in list(values):
            values[s] = values[s].subs(values)
        for p in self.dependency_order():
            values[p.symbol] = p.expr.subs(values)
        return {s: complex(v) if v.is_complex and not v.is_real else float(v)
                for s, v in values.items()}
