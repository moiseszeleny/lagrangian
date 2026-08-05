r"""Loop-induced Higgs decays: $h\to gg,\ \gamma\gamma,\ Z\gamma$ (Tier 3).

These channels vanish identically at tree level — the Higgs is colour- and
charge-neutral, so they proceed only through a top-quark triangle and (for
$\gamma\gamma$/$Z\gamma$) a $W$ loop.  A tree-level library cannot derive them
from the Lagrangian.

This module takes the **effective-vertex** route the roadmap recommends
(§16.3): it imports the standard closed-form one-loop form factors and assembles
the widths from them.  This is a *deliberate, documented exception* to the
derive-from-the-Lagrangian ethos — the same judgment as the FeynRules-style CKM
insertion in :mod:`feynlag.flavor` — because the honest one-loop computation
(Passarino–Veltman reduction, UV renormalisation) is a project the size of the
whole library, while the physical answer is a textbook closed form.

Every formula here is verified numerically against the PDG values (see
``tests/test_loop.py``) and cited: the $gg$/$\gamma\gamma$ form factors and the
$Z\gamma$ closed form follow Djouadi's review [Djouadi08] and Carena et al.
[CGLW13]; the NLO-QCD $gg$ $K$-factor follows Spira et al. [Spira95]; the
spin-0 (charged-scalar) form factor and its coefficient follow Chakraborti
et al. [ChakrabartyEtAl21].

References
----------
[ChakrabartyEtAl21] M. Chakraborti, D. Das, M. Levy, S. Mukherjee, I. Saha,
    "Prospects of light charged scalars in a three Higgs doublet model with
    Z₃ symmetry", arXiv:2104.08146.  Eq. (27) the diphoton signal strength with
    the charged-scalar loops and the coefficient
    $\kappa_i=-m_h^2/2m_{H_i^+}^2$; Eq. (28) the three loop functions, written
    in the reciprocal variable $x=4m^2/m_h^2$.
"""

import cmath
import math

__all__ = [
    "A_half", "A_one", "A_zero", "A_half_zgamma", "A_one_zgamma",
    "f_function", "g_function", "higgs_diphoton_amplitude", "higgs_gg_width",
    "higgs_gammagamma_width", "higgs_zgamma_width",
]

#: key for the spin-1/2 entry of the form-factor table (avoids float equality)
sp_half = "1/2"


# ------------------------------------------------------ auxiliary functions

def f_function(tau):
    r"""The scalar one-loop function $f(\tau)$.

    $f(\tau)=\arcsin^2\sqrt\tau$ for $\tau\le1$ (loop particle above threshold,
    the physical SM case), and the analytically-continued log form (with the
    $-i\pi$) for $\tau>1$ (a light loop particle — kept for BSM spectra).
    """
    tau = complex(tau)
    if tau.real <= 1 and tau.imag == 0:
        return complex(math.asin(math.sqrt(tau.real))**2, 0.0)
    eta = cmath.sqrt(1 - 1 / tau)
    return -0.25 * (cmath.log((1 + eta) / (1 - eta)) - 1j * math.pi)**2


def g_function(tau):
    r"""The second one-loop function $g(\tau)$ (needed for $Z\gamma$).

    $g(\tau)=\sqrt{\tau^{-1}-1}\,\arcsin\sqrt\tau$ for $\tau\le1$; the
    log-continued form for $\tau>1$.
    """
    tau = complex(tau)
    if tau.real <= 1 and tau.imag == 0:
        s = 1 / tau.real - 1
        if s < 0:
            s = 0.0
        return complex(math.sqrt(s) * math.asin(math.sqrt(tau.real)), 0.0)
    eta = cmath.sqrt(1 - 1 / tau)
    return 0.5 * eta * (cmath.log((1 + eta) / (1 - eta)) - 1j * math.pi)


# ------------------------------------------- single-argument form factors

def A_half(tau):
    r"""Spin-$\tfrac12$ (fermion-loop) form factor,
    $A_{1/2}(\tau)=2[\tau+(\tau-1)f(\tau)]/\tau^2$.

    $\tau=m_h^2/4m^2$.  Heavy-loop-particle limit $A_{1/2}\to 4/3$ as
    $\tau\to0$.
    """
    tau = complex(tau)
    return 2 * (tau + (tau - 1) * f_function(tau)) / tau**2


def A_one(tau):
    r"""Spin-1 ($W$-loop) form factor,
    $A_1(\tau)=-[2\tau^2+3\tau+3(2\tau-1)f(\tau)]/\tau^2$.

    Heavy-loop limit $A_1\to -7$ as $\tau\to0$.  Its sign is opposite the
    fermion factor's, so the $W$ and top interfere destructively in
    $h\to\gamma\gamma$.
    """
    tau = complex(tau)
    return -(2 * tau**2 + 3 * tau + 3 * (2 * tau - 1) * f_function(tau)) / tau**2


def A_zero(tau):
    r"""Spin-0 (charged-scalar-loop) form factor,
    $A_0(\tau)=[\tau-f(\tau)]/\tau^2$.

    Needed the moment the model has a charged scalar — a 2HDM/3HDM $H^\pm$
    running in the $h\to\gamma\gamma$ triangle.  Heavy-loop limit
    $A_0\to-\tfrac13$ as $\tau\to0$.

    **On the sign.**  Conventions differ by an overall sign of the whole
    amplitude, which is why this is fixed here by internal consistency rather
    than copied.  [ChakrabartyEtAl21] Eq. (28) writes the three loop functions
    as $F_W\to7$, $F_t\to-4/3$, $F^+\to+1/3$ in the reciprocal variable
    $x=4m^2/m_h^2=1/\tau$; feynlag's $A_1=-F_W$ and $A_{1/2}=-F_t$, so
    consistency *forces* $A_0=-F^+$ and hence the $-\tfrac13$ limit above.
    Pinned by ``tests/test_loop.py``.

    The physical content is the product with its coefficient
    $\kappa=g_{hH^+H^-}v/(2m_{H^\pm}^2)$ — see
    :func:`higgs_diphoton_amplitude`.
    """
    tau = complex(tau)
    return (tau - f_function(tau)) / tau**2


# ------------------------------------------- two-argument Zγ form factors

def _I1(x, y):
    x, y = complex(x), complex(y)
    return (x * y / (2 * (x - y))
            + x**2 * y**2 / (2 * (x - y)**2) * (f_function(1 / x) - f_function(1 / y))
            + x**2 * y / (x - y)**2 * (g_function(1 / x) - g_function(1 / y)))


def _I2(x, y):
    x, y = complex(x), complex(y)
    return -x * y / (2 * (x - y)) * (f_function(1 / x) - f_function(1 / y))


def A_half_zgamma(x, y):
    r"""Fermion form factor for $Z\gamma$: $A_{1/2}^{Z\gamma}=I_1(x,y)-I_2(x,y)$,
    with $x=4m^2/m_h^2$, $y=4m^2/m_Z^2$."""
    return _I1(x, y) - _I2(x, y)


def A_one_zgamma(x, y, sw2):
    r"""$W$ form factor for $Z\gamma$:
    $A_1^{Z\gamma}=4(3-t_W^2)I_2+[(1+2/x)t_W^2-(5+2/x)]I_1$,
    with $t_W^2=s_W^2/c_W^2$."""
    tw2 = sw2 / (1 - sw2)
    return (4 * (3 - tw2) * _I2(x, y)
            + ((1 + 2 / x) * tw2 - (5 + 2 / x)) * _I1(x, y))


# --------------------------------------------------------------- widths

def higgs_diphoton_amplitude(m_h, loops):
    r"""The dimensionless $h\to\gamma\gamma$ triangle amplitude, general content.

    $$\mathcal A=\sum_i c_i\,N_{c,i}\,Q_i^2\,A_{s_i}(m_h^2/4m_i^2)$$

    with $A_s$ the spin-$s$ form factor.  This is the BSM-ready form: the
    coefficients $c_i$ are what carry the model dependence, so a rescaled
    Higgs coupling or an extra charged particle in the loop is expressed
    without touching the form factors.

    Args:
        loops: iterable of ``(c, spin, mass, charge, n_colour)``.

            * **fermion** (``spin=1/2``): ``c`` = $g_{hff}/g^{\rm SM}_{hff}$
              (1 in the SM).
            * **vector** (``spin=1``): ``c`` = $g_{hVV}/g^{\rm SM}_{hVV}$.
            * **scalar** (``spin=0``): ``c`` = $g_{hSS}\,v/(2m_S^2)$ with
              $g_{hSS}$ the mass-dimension-1 trilinear — the normalization of
              [ChakrabartyEtAl21] Eq. (27), whose $\kappa_i=-m_h^2/2m_{H_i^+}^2$
              is this expression at their $g_{hH^+H^-}=-m_h^2/v$.

    The SM is ``[(1, 1, m_W, 1, 1), (1, 1/2, m_t, 2/3, 3)]``.
    """
    factors = {0: A_zero, sp_half: A_half, 1: A_one}
    amp = 0j
    for c, spin, mass, charge, n_colour in loops:
        key = sp_half if abs(float(spin) - 0.5) < 1e-12 else int(spin)
        if key not in factors:
            raise ValueError(f"no loop form factor for spin {spin!r}; "
                             f"have 0, 1/2 and 1")
        amp += c * n_colour * charge**2 * factors[key](m_h**2 / (4 * mass**2))
    return amp


def higgs_gammagamma_width(m_h, m_t, m_W, v, alpha, quarks=((2 / 3, 3),),
                           extra_loops=()):
    r"""$\Gamma(h\to\gamma\gamma)=\frac{\alpha^2 m_h^3}{256\pi^3 v^2}
    \big|\mathcal A\big|^2$ with $\mathcal A$ from
    :func:`higgs_diphoton_amplitude`.

    Args:
        quarks: iterable of ``(Q_f, N_c)`` for the fermion loops (default: the
            top quark, $Q=2/3$, $N_c=3$ — the only numerically relevant one).
        extra_loops: additional ``(c, spin, mass, charge, n_colour)`` entries,
            e.g. a charged Higgs ``(g_hHH*v/(2*mHp**2), 0, mHp, 1, 1)``.
            Empty by default, so the SM result is bit-for-bit unchanged.

    Returns:
        the width (same units as $m_h$; multiply by $10^6$ for keV when
        $m_h$ is in GeV).
    """
    loops = [(1, 1, m_W, 1, 1)]
    loops += [(1, 0.5, m_t, Q, Nc) for Q, Nc in quarks]
    loops += list(extra_loops)
    amp = higgs_diphoton_amplitude(m_h, loops)
    return alpha**2 * m_h**3 / (256 * math.pi**3 * v**2) * abs(amp)**2


def higgs_gg_width(m_h, m_t, v, alpha_s, qcd=False, n_flavors=5):
    r"""$\Gamma(h\to gg)=\frac{\alpha_s^2 m_h^3}{72\pi^3 v^2}
    \big|\tfrac34 A_{1/2}(\tau_t)\big|^2$ (leading order).

    With ``qcd=True`` multiply by the NLO $K$-factor
    $K=1+\frac{\alpha_s}{\pi}\left(\frac{95}{4}-\frac{7}{6}N_f\right)$ [Spira95]
    ($\approx1.64$ for $N_f=5$), which brings the LO ~0.20 MeV up to the
    measured ~0.34 MeV (8.2% branching ratio).
    """
    tau_t = m_h**2 / (4 * m_t**2)
    amp = 0.75 * A_half(tau_t)
    width = alpha_s**2 * m_h**3 / (72 * math.pi**3 * v**2) * abs(amp)**2
    if qcd:
        width *= 1 + alpha_s / math.pi * (95 / 4 - 7 / 6 * n_flavors)
    return width


def higgs_zgamma_width(m_h, m_t, m_W, m_Z, v, alpha, sw2):
    r"""$\Gamma(h\to Z\gamma)=\frac{\alpha^2 m_h^3}{512\pi^3}
    \left(1-\frac{m_Z^2}{m_h^2}\right)^3|\mathcal A|^2$, with

    $\mathcal A=\frac2v\big[\cot\theta_W A_1^{Z\gamma}(\tau_W,\lambda_W)
    +N_c(2Q_t)\frac{T_3^t-2Q_ts_W^2}{s_Wc_W}A_{1/2}^{Z\gamma}(\tau_t,\lambda_t)\big]$

    ($\tau_i=4m_i^2/m_h^2$, $\lambda_i=4m_i^2/m_Z^2$).  Top + $W$ only.
    """
    sw, cw = math.sqrt(sw2), math.sqrt(1 - sw2)
    tau_t, lam_t = 4 * m_t**2 / m_h**2, 4 * m_t**2 / m_Z**2
    tau_W, lam_W = 4 * m_W**2 / m_h**2, 4 * m_W**2 / m_Z**2
    Nc, Qt, T3t = 3, 2 / 3, 0.5
    amp = (2 / v) * (
        (cw / sw) * A_one_zgamma(tau_W, lam_W, sw2)
        + Nc * (2 * Qt) * (T3t - 2 * Qt * sw2) / (sw * cw)
        * A_half_zgamma(tau_t, lam_t))
    return (alpha**2 * m_h**3 / (512 * math.pi**3)
            * (1 - m_Z**2 / m_h**2)**3 * abs(amp)**2)
