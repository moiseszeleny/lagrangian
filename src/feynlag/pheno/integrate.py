"""Numerical Dalitz-plot integration for 1→3 decay widths.

The two-body width is a closed form; a three-body width through a resonance is a
2-D integral over the Dalitz region with a Breit–Wigner in the integrand, and
has no closed form in general.  This module is the library's numerical
integration layer.

Two backends: **SciPy** (``scipy.integrate``) when installed — robust adaptive
quadrature — and a dependency-free **numpy Gauss–Legendre** fallback.  SciPy is
the optional ``[numeric]`` extra; it is imported lazily here (the one place it
is used) so the rest of the library stays pure-SymPy/numpy.
"""

import numpy as np

__all__ = ["dalitz_integral", "have_scipy"]


def have_scipy():
    """Whether SciPy is importable (the ``[numeric]`` extra is installed)."""
    try:
        import scipy.integrate  # noqa: F401
        return True
    except ImportError:
        return False


def dalitz_integral(integrand, s23_lo, s23_hi, s12_bounds, backend="auto",
                    n_outer=120, n_inner=60):
    """Integrate ``integrand(s12, s23)`` over the physical Dalitz region.

    ``∬ integrand(s₁₂, s₂₃) ds₁₂ ds₂₃`` with the inner ``s₁₂`` limits a function
    of ``s₂₃``.

    Args:
        integrand: callable ``(s12, s23) -> float``.
        s23_lo, s23_hi: the outer ``s₂₃`` bounds (floats).
        s12_bounds: callable ``s23 -> (s12_lo, s12_hi)``.
        backend: ``"scipy"``, ``"gauss"`` (numpy Gauss–Legendre), or ``"auto"``
            (SciPy if available, else Gauss–Legendre).
        n_outer, n_inner: Gauss–Legendre node counts (the ``gauss`` backend).

    Returns:
        the integral as a float.
    """
    if backend == "auto":
        backend = "scipy" if have_scipy() else "gauss"

    if backend == "scipy":
        from scipy.integrate import dblquad
        # dblquad integrates f(y, x) with y (inner) limits as functions of x.
        val, _ = dblquad(
            lambda s12, s23: integrand(s12, s23),
            s23_lo, s23_hi,
            lambda s23: s12_bounds(s23)[0],
            lambda s23: s12_bounds(s23)[1],
        )
        return val

    if backend == "gauss":
        xo, wo = np.polynomial.legendre.leggauss(n_outer)
        xi, wi = np.polynomial.legendre.leggauss(n_inner)
        half = 0.5 * (s23_hi - s23_lo)
        mid = 0.5 * (s23_hi + s23_lo)
        total = 0.0
        for xo_k, wo_k in zip(xo, wo):
            s23 = half * xo_k + mid
            lo, hi = s12_bounds(s23)
            hh = 0.5 * (hi - lo)
            mm = 0.5 * (hi + lo)
            inner = 0.0
            for xi_j, wi_j in zip(xi, wi):
                inner += wi_j * integrand(hh * xi_j + mm, s23)
            total += wo_k * half * hh * inner
        return total

    raise ValueError(f"unknown backend {backend!r}; use 'scipy', 'gauss' or "
                     f"'auto'")
