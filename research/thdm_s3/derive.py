"""A vocabulary for deriving by hand, not just verifying by machine.

The 3HDM-S₃ notebooks were written in a **state → assert** idiom: the markdown
announces a result, the code confirms it, and the object you would actually have
written on paper is computed and discarded.  That establishes correctness and
teaches nothing — a passing ``assert`` only says the machine agrees with a result
you were handed.

This module supplies the moves a physicist makes instead:

1. **set up**   — :func:`show` the raw object,
2. **collect**  — :func:`coefficients_of` so the structure *emerges* rather than
   being typed in from the answer and asserted,
3. **recognise** — (markdown; naming what appeared and why it had to),
4. **check**    — :func:`check_homogeneous`, :func:`check_limit`,
   :func:`check_invariant`, :func:`hand_slice`: checks drawn from *physics*, not
   from re-running the same algebra,

plus :func:`falsify` (a check that cannot fail is not a check).  The *predict*
move is a markdown blockquote in the notebook, not a function — prose typesets
and a print statement does not.

:class:`Ledger` records what each step claimed, how it was obtained, which
independent checks it survived and what would falsify it — and renders that both
as an end-of-notebook table and as a printable LaTeX appendix, so the algebra can
be checked line by line away from the screen.

Research code: `tests/` must never import this (see `research/README.md`).  If
the vocabulary proves itself it is a candidate to graduate into `feynlag.verify`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from pathlib import Path

import sympy as sp

__all__ = [
    "show", "rows", "lit", "coefficients_of", "check_homogeneous", "check_limit",
    "check_invariant", "hand_slice", "falsify", "Ledger", "ok",
]


def ok(msg):
    """The notebooks' existing pass marker, re-exported so cells read uniformly."""
    print("✓", msg)


def _in_notebook():
    """True only under an IPython kernel, where `display` renders MathJax.

    Everything in this module degrades to plain text outside one, so the
    vocabulary stays usable from a script or a bare interpreter.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _display(obj):
    """IPython's `display` when in a notebook, `sp.pprint` otherwise."""
    if _in_notebook():
        from IPython.display import display as _d
        _d(obj)
    else:
        sp.pprint(obj)


def _math(latex_body):
    """Render a LaTeX fragment as typeset maths (notebook only)."""
    from IPython.display import Math, display
    display(Math(latex_body))


def _render_rows(rows, sep=r"\;\text{multiplies}\;", fallback_sep="multiplies"):
    """A two-column table of (symbol, expression) as typeset maths.

    The whole point of `coefficients_of` is that you *look* at the coefficients,
    so they must not arrive as ``(r_1**2 + r_2**2)**2``.  Falls back to the
    aligned text table outside a notebook.
    """
    if _in_notebook():
        body = r"\\".join(
            r"%s &%s& %s" % (sp.latex(a), sep, sp.latex(b)) for a, b in rows)
        _math(r"\begin{array}{rcl}%s\end{array}" % body)
        return
    width = max((len(str(a)) for a, _ in rows), default=8)
    for a, b in rows:
        print(f"  {str(a):<{width}}  {fallback_sep}  {b}")


def lit(name):
    """A symbol whose name is passed to LaTeX **verbatim**.

    ``sp.Symbol("R_S^T\\hat n")`` prints as ``R^{T\\hat n}_{S}`` — sympy re-parses
    the ``_``/``^`` in a symbol name and reassembles them. Wrapping the whole
    name in braces suppresses that, so labels for `show`/`rows`/`Ledger` come
    out as written.
    """
    return sp.Symbol("{%s}" % name)


def rows(pairs, sep=r"\;=\;", fallback_sep="="):
    """Display ``[(lhs, rhs), ...]`` as a typeset two-column table.

    The public face of the layout `coefficients_of` uses, for the derivations
    that build their own two-column table (a dictionary read off a matrix, a
    residual evaluated in two forms, an eigen-decomposition).
    """
    _render_rows(list(pairs), sep=sep, fallback_sep=fallback_sep)


# --------------------------------------------------------------------------
# 1. set up — look at the thing
# --------------------------------------------------------------------------

def show(label, expr, collect=None, count_terms=True):
    """Display ``expr`` under a label — the object as you would write it.

    ``collect`` optionally groups it first (a symbol or list of symbols passed
    to ``sp.collect``).  ``count_terms`` reports how many additive terms there
    are, which is the first thing you notice on paper and a useful sanity cue
    when a "simplification" silently drops something.
    """
    n = len(sp.Add.make_args(sp.expand(expr))) if count_terms else None
    head = label if n is None else f"{label}   [{n} terms]"
    print(head)
    _display(sp.collect(sp.expand(expr), collect) if collect is not None else expr)
    return expr


# --------------------------------------------------------------------------
# 2. collect — let the structure emerge
# --------------------------------------------------------------------------

def coefficients_of(expr, params, show_table=True, simplifier=sp.factor):
    """``{param: coefficient}`` — the "what multiplies λ₄?" move.

    This is the step that turns *asserting* a grouped form into *deriving* one:
    read off what each parameter multiplies and the shared structures announce
    themselves (two parameters carrying the same coefficient can only ever
    appear in that combination).

    Returns the dict; renders it as a typeset table when ``show_table``.
    """
    expr = sp.expand(expr)
    out = {}
    for p in params:
        c = expr.coeff(p)
        out[p] = simplifier(c) if simplifier is not None else c
    if show_table:
        _render_rows(list(out.items()))
    return out


# --------------------------------------------------------------------------
# 3. check — from physics, not from the same algebra
# --------------------------------------------------------------------------

def _fail(message, residual):
    """Raise, but *show* the offending expression first.

    A failed check is precisely when you want to look at the object rather than
    read ``str()`` of it wedged into an exception message.
    """
    print("✗", message)
    _display(residual)
    raise AssertionError(f"{message}: {residual}")

def check_homogeneous(expr, variables, degree, label=None):
    """``expr(t·vars) == t^degree · expr`` — the dimension/scaling reflex.

    For a quartic potential this is what licenses "bounded from below iff
    non-negative on the unit sphere": the radial direction factors out.
    """
    t = sp.Symbol("_t", positive=True)
    scaled = expr.subs({v: t * v for v in variables}, simultaneous=True)
    residual = sp.simplify(sp.expand(scaled - t**degree * expr))
    if residual != 0:
        _fail(f"not homogeneous of degree {degree}", residual)
    ok(label or f"homogeneous of degree {degree} in {len(variables)} variables")
    return True


def check_limit(expr, sub, expect, label, simplifier=sp.simplify):
    """``expr`` under ``sub`` must equal ``expect`` — "does it reduce to what I know?".

    The most informative single check in physics: switch off a coupling and see
    whether a familiar theory comes back.  A limit that *fails* to reproduce the
    known case is the loudest possible signal.
    """
    got = simplifier(sp.expand(expr.subs(sub)) - sp.expand(expect))
    if got != 0:
        _fail(f"limit check '{label}' failed; the difference is", got)
    ok(label)
    return True


def check_invariant(expr, transform, label, simplifier=sp.simplify):
    """``expr`` unchanged under ``transform`` — a symmetry it must respect.

    ``transform`` is a substitution dict (or callable).  Independent of how the
    expression was derived, which is exactly what makes it worth doing.
    """
    if not callable(transform):
        if not transform:
            raise ValueError(
                "check_invariant with an empty transform is vacuous: the residual "
                "is identically zero whatever `expr` is, so the check would pass on "
                "anything.  Use a plain assert if you meant 'this expression is 0'.")
        if not any(expr.has(k) for k in transform):
            raise ValueError(
                "check_invariant: the expression contains none of %s, so the "
                "substitution is a no-op and the check is vacuous."
                % sorted(map(str, transform)))
    moved = transform(expr) if callable(transform) else expr.subs(
        transform, simultaneous=True)
    residual = simplifier(sp.expand(moved - expr))
    if residual != 0:
        _fail(f"not invariant ({label}); residual", residual)
    ok(label)
    return True


def hand_slice(expr, sub, label, collect=None):
    """Collapse to a case small enough to finish on paper, and show it.

    The point is not the machine's answer but that *you* can now check it:
    two variables, a few terms, minimisable in a couple of minutes by hand.
    """
    small = sp.expand(expr.subs(sub, simultaneous=True))
    show(f"{label}:", small, collect=collect)
    return small


# --------------------------------------------------------------------------
# falsify — a check that cannot fail is not a check
# --------------------------------------------------------------------------

def falsify(label, broken):
    """Assert that ``broken()`` *fails*, proving the check has teeth.

    ``broken`` is a zero-argument callable that runs the same check on a
    deliberately damaged input.  It should raise ``AssertionError`` or return
    a falsy value; anything else means the check would have passed on a wrong
    input and is therefore worthless.
    """
    try:
        result = broken()
    except AssertionError:
        ok(f"{label} — the check fires on a broken input, so it has teeth")
        return True
    if result:
        raise AssertionError(
            f"'{label}': the check PASSED on a deliberately broken input — "
            f"it is not testing what it claims to test")
    ok(f"{label} — the check fires on a broken input, so it has teeth")
    return True


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

@dataclass
class Step:
    claim: str
    obtained: str                  # 'derived' | 'imported' | 'assumed'
    checks: tuple = ()
    falsified_by: str = ""
    expr: object = None
    section: str = ""


@dataclass
class Ledger:
    """The research-judgment record: what was claimed, how, and on what evidence."""

    title: str
    steps: list = _field(default_factory=list)

    _KINDS = ("derived", "imported", "assumed")

    def step(self, claim, obtained, checks=(), falsified_by="", expr=None,
             section=""):
        if obtained not in self._KINDS:
            raise ValueError(f"`obtained` must be one of {self._KINDS}, "
                             f"got {obtained!r}")
        self.steps.append(Step(claim, obtained, tuple(checks), falsified_by,
                               expr, section))
        return self.steps[-1]

    def table(self):
        """Print the ledger — claim / how obtained / checks / what would falsify."""
        print(f"\nDERIVATION LEDGER — {self.title}")
        print("=" * 76)
        for i, s in enumerate(self.steps, 1):
            tag = {"derived": "derived here", "imported": "IMPORTED",
                   "assumed": "ASSUMED"}[s.obtained]
            print(f"{i}. [{tag}] {s.claim}")
            if s.section:
                print(f"     section        : {s.section}")
            if s.expr is not None:
                _display(s.expr)          # the result itself, typeset
            for c in s.checks:
                print(f"     check          : {c}")
            if s.falsified_by:
                print(f"     falsified by   : {s.falsified_by}")
        n = {k: sum(1 for s in self.steps if s.obtained == k) for k in self._KINDS}
        print("-" * 76)
        print(f"{len(self.steps)} steps: {n['derived']} derived here, "
              f"{n['imported']} imported, {n['assumed']} assumed")

    # ---------------------------------------------------------------- LaTeX
    def to_latex(self, path, subtitle=""):
        """Emit a printable appendix: every step typeset, for checking on paper.

        Uses ``multiline_latex`` where available so long expressions break
        across lines instead of running off the page.
        """
        body = [_TEX_HEADER % (_tex_escape(self.title), _tex_escape(subtitle))]
        for i, s in enumerate(self.steps, 1):
            tag = {"derived": "derived here", "imported": "imported",
                   "assumed": "assumed"}[s.obtained]
            body.append(r"\subsection*{%d.\quad %s}" % (i, _tex_escape(s.claim)))
            body.append(r"\emph{%s}%s\par\medskip" % (
                tag, r" --- \S%s" % _tex_escape(s.section) if s.section else ""))
            if s.expr is not None:
                body.append(_tex_expr(s.expr))
            if s.checks:
                body.append(r"\textbf{Independent checks}\begin{itemize}\itemsep0pt")
                body += [r"\item %s" % _tex_escape(c) for c in s.checks]
                body.append(r"\end{itemize}")
            if s.falsified_by:
                body.append(r"\textbf{Would be falsified by:} %s\par"
                            % _tex_escape(s.falsified_by))
            body.append(r"\medskip\hrule\medskip")
        body.append(r"\end{document}")
        Path(path).write_text("\n".join(body))
        print(f"wrote {path}  ({len(self.steps)} steps)")
        return path


_TEX_HEADER = r"""%% Auto-generated by research/thdm_s3/derive.py -- do not edit by hand.
%% Printable derivation appendix: check the algebra line by line on paper.
\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage[margin=2.2cm]{geometry}
\setlength{\parindent}{0pt}
\title{Derivation appendix: %s\\[4pt]\large\normalfont %s}
\date{}
\begin{document}
\maketitle
"""


#: Unicode that turns up in physics prose but that pdflatex (the engine the repo
#: already uses for LFVHD_3HDMS3.tex) cannot typeset.  Transliterated rather than
#: switching to lualatex, so the appendix builds with the same toolchain.
_TEX_UNICODE = {
    "₀": "$_0$", "₁": "$_1$", "₂": "$_2$", "₃": "$_3$", "₄": "$_4$",
    "₅": "$_5$", "₆": "$_6$", "₇": "$_7$", "₈": "$_8$", "₉": "$_9$",
    "⁰": "$^0$", "¹": "$^1$", "²": "$^2$", "³": "$^3$", "⁴": "$^4$",
    "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$",
    "θ": r"$\theta$", "λ": r"$\lambda$", "μ": r"$\mu$", "ν": r"$\nu$",
    "π": r"$\pi$", "σ": r"$\sigma$", "τ": r"$\tau$", "φ": r"$\varphi$",
    "ψ": r"$\psi$", "Λ": r"$\Lambda$", "Φ": r"$\Phi$", "Σ": r"$\Sigma$",
    "→": r"$\to$", "←": r"$\leftarrow$", "≈": r"$\approx$", "≠": r"$\neq$",
    "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$", "·": r"$\cdot$",
    "±": r"$\pm$", "∞": r"$\infty$", "√": r"$\sqrt{\ }$",
    "—": "---", "–": "--", "✓": r"$\checkmark$", "⟨": "$\\langle$",
    "⟩": "$\\rangle$", "“": "``", "”": "''", "’": "'", "‘": "`",
}


def _tex_escape(s):
    """LaTeX-safe text: escape the special characters, transliterate Unicode."""
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        out = out.replace(a, b)
    for a, b in _TEX_UNICODE.items():
        out = out.replace(a, b)
    # anything still outside Latin-1 would break pdflatex; drop it loudly-ish
    return "".join(ch if ord(ch) < 256 else "?" for ch in out)


def _tex_expr(expr):
    """Typeset one expression, breaking long ones across lines."""
    try:
        from sympy import multiline_latex
        if len(sp.Add.make_args(sp.expand(expr))) > 6:
            return multiline_latex(sp.Symbol("V"), expr, terms_per_line=3,
                                   environment="align*")
    except Exception:
        pass
    return r"\begin{equation*}\small %s \end{equation*}" % sp.latex(expr)
