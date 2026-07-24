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
[CGLW13]; the NLO-QCD $gg$ $K$-factor follows Spira et al. [Spira95].
"""

import cmath
import math

__all__ = [
    "A_half", "A_one", "A_half_zgamma", "A_one_zgamma", "f_function",
    "g_function", "higgs_gg_width", "higgs_gammagamma_width",
    "higgs_zgamma_width",
]


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

def higgs_gammagamma_width(m_h, m_t, m_W, v, alpha, quarks=((2 / 3, 3),)):
    r"""$\Gamma(h\to\gamma\gamma)=\frac{\alpha^2 m_h^3}{256\pi^3 v^2}
    \big|A_1(\tau_W)+\sum_f N_c Q_f^2 A_{1/2}(\tau_f)\big|^2$.

    Args:
        quarks: iterable of ``(Q_f, N_c)`` for the fermion loops (default: the
            top quark, $Q=2/3$, $N_c=3$ — the only numerically relevant one).

    Returns:
        the width (same units as $m_h$; multiply by $10^6$ for keV when
        $m_h$ is in GeV).
    """
    tau_W = m_h**2 / (4 * m_W**2)
    amp = A_one(tau_W)
    for Q, Nc in quarks:
        tau_f = m_h**2 / (4 * m_t**2)
        amp += Nc * Q**2 * A_half(tau_f)
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
