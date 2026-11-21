"""Malaria Transmission Simulator (Ross-Macdonald Mathematical Model).

A unified, self-contained single-file Python application implementing both the
baseline Ross-Macdonald model (with extrinsic incubation period tau) and an
extended climate-driven model using 4th-order Runge-Kutta (RK4) integration.
Presents dynamic transmission plots alongside publication-ready LaTeX formulas,
automated numerical verification checks, sensitivity analysis, and PNG/CSV exports.

Requirements:
    pip install numpy matplotlib
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import csv
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ==============================================================================

# Graphical User Interface (GUI) Settings (Pure white minimal theme)
APP_TITLE = "Malaria Model: Ross-Macdonald Transmission Simulator"
APP_SUBTITLE = "Baseline & Climate-Driven Host-Vector Models, RK4 Integration & Analytical Solution"
WINDOW_SIZE = "980x980"
BG_COLOR = "#FFFFFF"
FG_COLOR = "#111827"
MUTED_COLOR = "#4B5563"
BORDER_COLOR = "#E5E7EB"
CARD_BG = "#FAFAFA"
ACCENT_GREEN = "#15803D"
ACCENT_RED = "#B91C1C"

# Default Baseline Model Parameters (based on Gordon & Guilfoos Chapter 14)
# Tuple format: (Label text, default value, parameter key)
BASELINE_FIELDS = [
    ("Human Pop. (N):", "1000", "N"),
    ("Mosquito Pop. (M):", "5000", "M"),
    ("Biting Rate (a):", "0.25", "a"),
    ("Prob M->H (b):", "0.30", "b"),
    ("Prob H->M (c):", "0.15", "c"),
    ("Recovery Rate (r):", "0.01", "r"),
    ("Mosq. Mortality (μ_m):", "0.10", "mu_m"),
    ("EIP (τ, days):", "10", "tau"),
    ("Duration (Days):", "180", "days"),
]

# Default Extended (Climate-Driven) Model Parameters
EXTENDED_FIELDS = [
    ("Mean Temp (T_mean, °C):", "25", "T_mean"),
    ("Temp Amp (A_T, °C):", "4", "A_T"),
    ("Peak Day (t_p):", "150", "t_p"),
    ("Seasonality (κ):", "0.5", "kappa"),
]

# Simulation Constants
INITIAL_INFECTED_HUMANS_RATIO = 0.01      # 1% of human population initially infected
INITIAL_INFECTED_MOSQUITOES_RATIO = 0.00  # 0% of mosquito population initially infected
TIME_STEP = 0.1                           # RK4 integration step size (0.1 day)


# ==============================================================================
# SECTION 2: MATHEMATICAL MODEL & SIMULATION LOGIC
# ==============================================================================

@dataclass
class SimulationParameters:
    """Input parameters for baseline and climate-driven Ross-Macdonald models."""
    N: float             # Total human population size
    M: float             # Total mosquito population size
    a: float             # Baseline biting rate (bites per mosquito per day)
    b: float             # Transmission prob: infectious mosquito -> susceptible human
    c: float             # Transmission prob: infectious human -> susceptible mosquito
    r: float             # Human recovery rate (1/r = mean infectious duration in days)
    mu_m: float          # Baseline mosquito mortality rate (day^-1)
    tau: float           # Extrinsic incubation period (days)
    days: int            # Total simulation duration in days
    T_mean: float = 25.0 # Mean annual temperature (°C)
    A_T: float = 4.0     # Seasonal temperature amplitude (°C)
    t_p: float = 150.0   # Day of seasonal peak (days)
    kappa: float = 0.5   # Seasonal mosquito density amplitude (0 <= kappa <= 1)


@dataclass
class SimulationResult:
    """Baseline simulation output including daily trajectories, equilibria, and LaTeX."""
    R0: float
    m: float
    t: np.ndarray
    X: np.ndarray             # Daily infected human fraction
    Y: np.ndarray             # Daily infected mosquito fraction
    latex_code: str           # Ready-to-copy LaTeX code block
    days: int
    numerator: float          # m * a^2 * b * c * exp(-mu_m * tau)
    denominator: float        # r * mu_m
    c_eff: float              # Effective human->mosquito transmission c * exp(-mu_m * tau)
    x_star: Optional[float]   # Analytic human endemic equilibrium
    y_star: Optional[float]   # Analytic mosquito endemic equilibrium


@dataclass
class ExtendedSimulationResult:
    """Extended climate-driven simulation output (daily trajectories)."""
    t: np.ndarray
    x: np.ndarray             # Daily infected human fraction
    z: np.ndarray             # Daily latent (exposed) mosquito fraction
    v: np.ndarray             # Daily infectious mosquito fraction
    T: np.ndarray             # Daily temperature (°C)
    m: np.ndarray             # Daily mosquito-to-human ratio m(t)
    days: int


def _rk4_step(rhs: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Advance a state vector y by one step dt using 4th-order Runge-Kutta (RK4)."""
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def calculate_r0(m: float, a: float, b: float, c: float, r: float, mu_m: float, tau: float) -> float:
    r"""Calculate the Basic Reproduction Number (R0) with extrinsic incubation period tau.

    Mathematical Formulation:
        R_0 = \frac{m \cdot a^2 \cdot b \cdot c \cdot e^{-\mu_m \tau}}{r \cdot \mu_m}
    """
    numerator = m * (a ** 2) * b * c * float(np.exp(-mu_m * tau))
    denominator = r * mu_m
    return numerator / denominator


def simulate_transmission(
    params: SimulationParameters,
    dt: float = TIME_STEP,
    x0: float = INITIAL_INFECTED_HUMANS_RATIO,
    y0: float = INITIAL_INFECTED_MOSQUITOES_RATIO,
) -> SimulationResult:
    r"""Simulate the baseline Ross-Macdonald ODEs with EIP tau using RK4 (daily output).

    Governing Differential System:
        dX/dt = m * a * b * Y * (1 - X) - r * X
        dY/dt = a * c * exp(-mu_m * tau) * X * (1 - Y) - mu_m * Y
    """
    m = params.M / params.N
    survival = float(np.exp(-params.mu_m * params.tau))
    c_eff = params.c * survival
    numerator = m * (params.a ** 2) * params.b * c_eff
    denominator = params.r * params.mu_m
    R0 = numerator / denominator

    # Analytic endemic equilibrium (correct formula: R0 - 1 over R0 + a*c_eff/mu_m)
    if R0 > 1.0:
        ac_over_mu = (params.a * c_eff) / params.mu_m
        x_star = (R0 - 1.0) / (R0 + ac_over_mu)
        y_star = (params.a * c_eff * x_star) / (params.a * c_eff * x_star + params.mu_m)
    else:
        x_star = 0.0
        y_star = 0.0

    # RK4 integration with step dt, recording daily values
    substeps = max(1, int(round(1.0 / dt)))
    step_dt = 1.0 / substeps

    t_daily = np.arange(0, params.days + 1, dtype=float)
    X_daily = np.zeros(params.days + 1, dtype=float)
    Y_daily = np.zeros(params.days + 1, dtype=float)
    state = np.array([x0, y0], dtype=float)
    X_daily[0], Y_daily[0] = state[0], state[1]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        x_val, y_val = y[0], y[1]
        dx = m * params.a * params.b * y_val * (1.0 - x_val) - params.r * x_val
        dy = params.a * c_eff * x_val * (1.0 - y_val) - params.mu_m * y_val
        return np.array([dx, dy], dtype=float)

    t_curr = 0.0
    for day in range(params.days):
        for _ in range(substeps):
            state = _rk4_step(rhs, t_curr, state, step_dt)
            t_curr += step_dt
        X_daily[day + 1] = state[0]
        Y_daily[day + 1] = state[1]

    # Publication-ready LaTeX source code block
    latex_code = (
        "% ==========================================================\n"
        "% Ross-Macdonald Malaria Model: Formulas & Substituted Solution\n"
        "% ==========================================================\n\n"
        "% 1. Model Inputs Provided:\n"
        f"% N = {params.N:.0f}, M = {params.M:.0f}, a = {params.a}, b = {params.b}, "
        f"c = {params.c}, r = {params.r}, mu_m = {params.mu_m}, tau = {params.tau}, Days = {params.days}\n\n"
        "% 2. Governing Differential Equations (with EIP tau):\n"
        "\\begin{aligned}\n"
        "\\frac{dX}{dt} &= m \\cdot a \\cdot b \\cdot Y(t)\\,(1 - X(t)) - r \\cdot X(t) \\\\[6pt]\n"
        "\\frac{dY}{dt} &= a \\cdot c \\cdot e^{-\\mu_m \\tau} \\cdot X(t)\\,(1 - Y(t)) - \\mu_m \\cdot Y(t)\n"
        "\\end{aligned}\n\n"
        "% 3. Mosquito-to-Human Population Ratio:\n"
        f"m = \\frac{{M}}{{N}} = \\frac{{{params.M:.0f}}}{{{params.N:.0f}}} = {m:.4f}\n\n"
        "% 4. Basic Reproduction Number (R_0) & Analytical Evaluation:\n"
        "\\begin{aligned}\n"
        f"R_0 &= \\frac{{m \\cdot a^2 \\cdot b \\cdot c \\cdot e^{{-\\mu_m \\tau}}}}{{r \\cdot \\mu_m}} \\\\[6pt]\n"
        f"    &= \\frac{{{m:.4f} \\times ({params.a})^2 \\times {params.b} \\times {params.c} \\times e^{{-{params.mu_m} \\cdot {params.tau}}}}}{{{params.r} \\times {params.mu_m}}} \\\\[6pt]\n"
        f"    &= \\frac{{{numerator:.7f}}}{{{denominator:.7f}}} \\\\[6pt]\n"
        f"    &= {R0:.4f} \\quad \\left(R_0 {' > 1 \\implies \\text{Endemic Persistence}' if R0 > 1 else ' \\leq 1 \\implies \\text{Disease Dies Out}'}\\right)\n"
        "\\end{aligned}\n\n"
        "% 5. Analytic Steady-State Equilibrium:\n"
        "\\begin{aligned}\n"
        f"X^* &= \\frac{{R_0 - 1}}{{R_0 + \\frac{{a \\cdot c \\cdot e^{{-\\mu_m \\tau}}}}{{\\mu_m}}}} = {x_star:.4f} \\quad ({x_star*100:.2f}\\%)\\\\[6pt]\n"
        f"Y^* &= \\frac{{a \\cdot c \\cdot e^{{-\\mu_m \\tau}} \\cdot X^*}}{{a \\cdot c \\cdot e^{{-\\mu_m \\tau}} \\cdot X^* + \\mu_m}} = {y_star:.4f} \\quad ({y_star*100:.2f}\\%)\n"
        "\\end{aligned}\n"
    )

    return SimulationResult(
        R0=R0,
        m=m,
        t=t_daily,
        X=X_daily,
        Y=Y_daily,
        latex_code=latex_code,
        days=params.days,
        numerator=numerator,
        denominator=denominator,
        c_eff=c_eff,
        x_star=x_star,
        y_star=y_star,
    )


def _climate_forcing(t: float, params: SimulationParameters) -> Tuple[float, float, float, float, float]:
    """Compute temperature T(t) and climate-driven rates a(T), mu(T), tau(T), m(t)."""
    phase = 2.0 * np.pi * (t - params.t_p) / 365.0
    sin_val = float(np.sin(phase))
    T = params.T_mean + params.A_T * sin_val
    a_T = 2.03e-4 * T * max(T - 11.7, 0.0) * float(np.sqrt(max(42.3 - T, 0.0)))
    mu_T = 0.08 + 0.002 * ((T - 25.0) ** 2)
    tau_T = 111.0 / max(T - 16.0, 1.0)
    m0 = params.M / params.N
    m_t = m0 * (1.0 + params.kappa * sin_val)
    return T, a_T, mu_T, tau_T, m_t


def simulate_extended(
    params: SimulationParameters,
    dt: float = TIME_STEP,
    x0: float = INITIAL_INFECTED_HUMANS_RATIO,
    z0: float = 0.0,
    v0: float = 0.0,
) -> ExtendedSimulationResult:
    """Simulate the climate-driven extended Ross-Macdonald model using RK4 (daily output).

    State vector: [x (infected humans), z (latent mosquitoes), v (infectious mosquitoes)].
    """
    substeps = max(1, int(round(1.0 / dt)))
    step_dt = 1.0 / substeps

    t_daily = np.arange(0, params.days + 1, dtype=float)
    x_daily = np.zeros(params.days + 1, dtype=float)
    z_daily = np.zeros(params.days + 1, dtype=float)
    v_daily = np.zeros(params.days + 1, dtype=float)
    T_daily = np.zeros(params.days + 1, dtype=float)
    m_daily = np.zeros(params.days + 1, dtype=float)

    state = np.array([x0, z0, v0], dtype=float)
    T0, _, _, _, m0_t = _climate_forcing(0.0, params)
    x_daily[0], z_daily[0], v_daily[0] = state[0], state[1], state[2]
    T_daily[0], m_daily[0] = T0, m0_t

    def rhs(t_val: float, y: np.ndarray) -> np.ndarray:
        x_val, z_val, v_val = y[0], y[1], y[2]
        _, a_T, mu_T, tau_T, m_t = _climate_forcing(t_val, params)
        dx = m_t * a_T * params.b * v_val * (1.0 - x_val) - params.r * x_val
        dz = a_T * params.c * x_val * (1.0 - z_val - v_val) - (1.0 / tau_T + mu_T) * z_val
        dv = z_val / tau_T - mu_T * v_val
        return np.array([dx, dz, dv], dtype=float)

    t_curr = 0.0
    for day in range(params.days):
        for _ in range(substeps):
            state = _rk4_step(rhs, t_curr, state, step_dt)
            t_curr += step_dt
        x_daily[day + 1], z_daily[day + 1], v_daily[day + 1] = state[0], state[1], state[2]
        T_d, _, _, _, m_d = _climate_forcing(float(day + 1), params)
        T_daily[day + 1] = T_d
        m_daily[day + 1] = m_d

    return ExtendedSimulationResult(
        t=t_daily,
        x=x_daily,
        z=z_daily,
        v=v_daily,
        T=T_daily,
        m=m_daily,
        days=params.days,
    )


def verify(params: Optional[SimulationParameters] = None) -> Tuple[bool, str]:
    """Run automated verification assertions on both baseline and extended RK4 solvers.

    Checks:
      1. Bounds: 0 <= X, Y <= 1 and 0 <= x, z, v <= 1 and z + v <= 1 for all t; no NaNs.
      2. Zero-infection invariance: x(0) = 0 (with zero infected vectors) remains 0.
      3. Analytic convergence: baseline converges to x_star when R0 > 1 and decays to 0 when R0 < 1.
      4. Temporal convergence: halving dt (0.1 -> 0.05) changes trajectories by < 1e-4.
    """
    if params is None:
        params = SimulationParameters(N=1000, M=5000, a=0.25, b=0.30, c=0.15, r=0.01, mu_m=0.10, tau=10, days=180)

    report = []
    try:
        # 1. Bounds & NaN check on both models
        res_base = simulate_transmission(params, dt=0.1)
        res_ext = simulate_extended(params, dt=0.1)

        assert not np.isnan(res_base.X).any() and not np.isnan(res_base.Y).any(), "NaN in baseline output"
        assert np.all((res_base.X >= 0.0) & (res_base.X <= 1.0)), "Baseline X out of [0, 1]"
        assert np.all((res_base.Y >= 0.0) & (res_base.Y <= 1.0)), "Baseline Y out of [0, 1]"

        assert not (np.isnan(res_ext.x).any() or np.isnan(res_ext.z).any() or np.isnan(res_ext.v).any()), "NaN in extended output"
        assert np.all((res_ext.x >= 0.0) & (res_ext.x <= 1.0)), "Extended x out of [0, 1]"
        assert np.all((res_ext.z >= 0.0) & (res_ext.z <= 1.0)), "Extended z out of [0, 1]"
        assert np.all((res_ext.v >= 0.0) & (res_ext.v <= 1.0)), "Extended v out of [0, 1]"
        assert np.all(res_ext.z + res_ext.v <= 1.0 + 1e-12), "Extended z + v > 1"
        report.append("PASS: Bounds (0<=x,z,v<=1, z+v<=1) & No NaN")

        # 2. Zero initial infection invariance: x(0)=0 stays 0
        z_base = simulate_transmission(params, dt=0.1, x0=0.0, y0=0.0)
        z_ext = simulate_extended(params, dt=0.1, x0=0.0, z0=0.0, v0=0.0)
        assert np.allclose(z_base.X, 0.0, atol=1e-12), "Baseline x(0)=0 did not stay 0"
        assert np.allclose(z_ext.x, 0.0, atol=1e-12), "Extended x(0)=0 did not stay 0"
        report.append("PASS: x(0)=0 invariance")

        # 3. Analytic convergence (R0 > 1 -> x_star; R0 < 1 -> 0)
        p_endemic = SimulationParameters(N=1000, M=5000, a=0.25, b=0.30, c=0.15, r=0.01, mu_m=0.10, tau=10, days=2000)
        r_endemic = simulate_transmission(p_endemic, dt=0.1)
        assert r_endemic.R0 > 1.0
        err_eq = abs(r_endemic.X[-1] - r_endemic.x_star)
        assert err_eq < 1e-4, f"Did not converge to x_star: diff={err_eq}"

        p_die = SimulationParameters(N=1000, M=100, a=0.10, b=0.30, c=0.15, r=0.05, mu_m=0.10, tau=10, days=1000)
        r_die = simulate_transmission(p_die, dt=0.1)
        assert r_die.R0 < 1.0
        assert r_die.X[-1] < 1e-4, f"Did not decay to 0: X[-1]={r_die.X[-1]}"
        report.append(f"PASS: Analytic convergence (err={err_eq:.2e} for R0>1, X_end={r_die.X[-1]:.2e} for R0<1)")

        # 4. dt halving convergence (< 1e-4)
        base_half = simulate_transmission(params, dt=0.05)
        ext_half = simulate_extended(params, dt=0.05)
        diff_base = float(np.max(np.abs(res_base.X - base_half.X)))
        diff_ext = float(np.max(np.abs(res_ext.x - ext_half.x)))
        assert diff_base < 1e-4, f"Baseline dt halving diff {diff_base} >= 1e-4"
        assert diff_ext < 1e-4, f"Extended dt halving diff {diff_ext} >= 1e-4"
        report.append(f"PASS: dt halving (base={diff_base:.2e}, ext={diff_ext:.2e} < 1e-4)")

        return True, "\n".join(report)

    except AssertionError as err:
        report.append(f"FAIL: {err}")
        return False, "\n".join(report)


# ==============================================================================
# SECTION 3: VISUALIZATION & LATEX RENDERING (UNIFIED FIGURE)
# ==============================================================================

def create_unified_figure(
    params: SimulationParameters,
    result: SimulationResult,
    ext_result: Optional[ExtendedSimulationResult] = None,
) -> Figure:
    """Generate a single unified figure: transmission plot on top, large vertical LaTeX solution below."""
    fig_height = 10.2 if ext_result is not None else 9.2
    fig = plt.figure(figsize=(9.2, fig_height), dpi=100, facecolor=BG_COLOR)

    # --------------------------------------------------------------------------
    # TOP SUBPLOT: Dynamic Transmission Curves (+ Temperature if Extended)
    # --------------------------------------------------------------------------
    plot_width = 0.80 if ext_result is not None else 0.87
    ax_plot = fig.add_axes([0.08, 0.65, plot_width, 0.30])
    ax_plot.set_facecolor(BG_COLOR)

    if ext_result is None:
        ax_plot.plot(result.t, result.X * 100, label="Infected Humans (%)", color="#D9381E", linewidth=2.2)
        ax_plot.plot(result.t, result.Y * 100, label="Infected Mosquitoes (%)", color="#1F77B4", linestyle="--", linewidth=2.2)
        ax_plot.set_title(
            f"Baseline Malaria Transmission Dynamics over {result.days} Days (RK4)",
            fontsize=11.5, fontweight="bold", color=FG_COLOR, pad=8
        )
        ax_plot.legend(frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR, fontsize=9, loc="center right")
    else:
        l1, = ax_plot.plot(result.t, result.X * 100, label="Baseline Humans (%)", color="#D9381E", linewidth=2.0)
        l2, = ax_plot.plot(ext_result.t, ext_result.x * 100, label="Extended Humans (%)", color="#15803D", linewidth=2.2)

        ax_temp = ax_plot.twinx()
        l3, = ax_temp.plot(ext_result.t, ext_result.T, label="Temperature T(t) (°C)", color="#6B7280", linestyle=":", linewidth=1.6)
        ax_temp.set_ylabel("Temperature (°C)", fontsize=9.5, color=MUTED_COLOR)
        ax_temp.tick_params(axis="y", colors=MUTED_COLOR, labelsize=8.5)

        ax_plot.set_title(
            f"Baseline vs. Extended (Climate-Driven) Dynamics over {result.days} Days (RK4)",
            fontsize=11, fontweight="bold", color=FG_COLOR, pad=8
        )
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax_plot.legend(lines, labels, frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR, fontsize=8.5, loc="upper left")

    ax_plot.set_xlabel("Time (Days)", fontsize=9.5, color=MUTED_COLOR)
    ax_plot.set_ylabel("Infected Proportion (%)", fontsize=9.5, color=MUTED_COLOR)
    ax_plot.grid(True, linestyle=":", alpha=0.6, color="#D1D5DB")
    ax_plot.tick_params(colors=MUTED_COLOR, labelsize=8.5)

    # --------------------------------------------------------------------------
    # BOTTOM SUBPLOT: Vertical Step-by-Step LaTeX Formulas & Solution
    # --------------------------------------------------------------------------
    ax_formula = fig.add_axes([0.04, 0.02, 0.92, 0.56])
    ax_formula.set_facecolor(BG_COLOR)
    ax_formula.axis("off")

    card_rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax_formula.add_patch(card_rect)

    y_pos = 0.96

    def add_row(text: str, dy: float, fontsize: float = 11.0, bold: bool = False, color: str = FG_COLOR, x: float = 0.05):
        nonlocal y_pos
        ax_formula.text(
            x, y_pos, text,
            fontsize=fontsize,
            fontweight=("bold" if bold else "normal"),
            color=color,
            va="top",
        )
        y_pos -= dy

    # 1. Inputs Given
    add_row("1. Model Inputs Provided:", 0.042, fontsize=10.5, bold=True, color=FG_COLOR, x=0.03)
    base_inputs_math = (
        rf"$N={params.N:.0f},\ M={params.M:.0f},\ a={params.a},\ b={params.b},\ "
        rf"c={params.c},\ r={params.r},\ \mu_m={params.mu_m},\ \tau={params.tau},\ \text{{Days}}={params.days}$"
    )
    add_row(base_inputs_math, 0.046, fontsize=10.2, color="#1F2937")
    if ext_result is not None:
        ext_inputs_math = (
            rf"$\text{{Climate: }}\ T_{{\text{{mean}}}}={params.T_mean}^\circ\text{{C}},\ "
            rf"A_T={params.A_T}^\circ\text{{C}},\ t_p={params.t_p},\ \kappa={params.kappa}$"
        )
        add_row(ext_inputs_math, 0.046, fontsize=10.2, color="#1F2937")

    # 2. Governing ODEs (stacked vertically)
    y_pos -= 0.012
    if ext_result is None:
        add_row(r"2. Governing Ross-Macdonald Differential Equations (with EIP $\tau$):", 0.042, fontsize=10.5, bold=True, color=MUTED_COLOR, x=0.03)
        add_row(r"$\frac{dX}{dt} = m \cdot a \cdot b \cdot Y(1 - X) - r \cdot X$", 0.062, fontsize=12.0, color="#1E3A8A")
        add_row(r"$\frac{dY}{dt} = a \cdot c \cdot e^{-\mu_m \tau} X(1 - Y) - \mu_m \cdot Y$", 0.062, fontsize=12.0, color="#1E3A8A")
    else:
        add_row("2. Extended Climate-Driven System (x: Humans, z: Latent Mosq., v: Infectious Mosq.):", 0.040, fontsize=10.2, bold=True, color=MUTED_COLOR, x=0.03)
        add_row(r"$\frac{dx}{dt} = m(t)\,a(T)\,b\,v(1 - x) - r\,x$", 0.056, fontsize=11.5, color="#1E3A8A")
        add_row(r"$\frac{dz}{dt} = a(T)\,c\,x(1 - z - v) - \left(\frac{1}{\tau(T)} + \mu(T)\right)z$", 0.060, fontsize=11.5, color="#1E3A8A")
        add_row(r"$\frac{dv}{dt} = \frac{z}{\tau(T)} - \mu(T)\,v$", 0.056, fontsize=11.5, color="#1E3A8A")

    # 3. Step-by-Step R0 Solution (vertical process)
    y_pos -= 0.012
    add_row(r"3. Step-by-Step Basic Reproduction Number ($R_0$) Solution:", 0.042, fontsize=10.5, bold=True, color=MUTED_COLOR, x=0.03)
    add_row(rf"$\text{{Step 1 (Ratio): }}\quad m = \frac{{M}}{{N}} = \frac{{{params.M:.0f}}}{{{params.N:.0f}}} = {result.m:.4f}$", 0.062, fontsize=11.2)
    add_row(r"$\text{Step 2 (Formula): }\quad R_0 = \frac{m \cdot a^2 \cdot b \cdot c \cdot e^{-\mu_m \tau}}{r \cdot \mu_m}$", 0.066, fontsize=11.5)
    add_row(
        rf"$\text{{Step 3 (Substitute): }}\quad R_0 = \frac{{{result.m:.2f} \times ({params.a})^2 \times {params.b} \times {params.c} \times e^{{-{params.mu_m}\cdot {params.tau}}}}}{{{params.r} \times {params.mu_m}}}$",
        0.066, fontsize=11.2
    )
    add_row(
        rf"$\text{{Step 4 (Evaluate): }}\quad R_0 = \frac{{{result.numerator:.6f}}}{{{result.denominator:.6f}}} = \mathbf{{{result.R0:.4f}}}$",
        0.066, fontsize=11.5, bold=True
    )

    # 4. Outcome & Analytic Equilibrium (vertical process)
    y_pos -= 0.012
    add_row("4. Epidemiological Outcome & Analytic Steady-State:", 0.042, fontsize=10.5, bold=True, color=MUTED_COLOR, x=0.03)
    if result.R0 > 1.0 and result.x_star is not None and result.y_star is not None:
        tau_term = r"e^{-\mu_m \tau}" if params.tau != 0 else ""
        add_row(rf"$R_0 = {result.R0:.4f} > 1 \quad \Longrightarrow \quad \text{{Endemic Persistence}}$", 0.048, fontsize=11.0, bold=True, color=ACCENT_GREEN)
        add_row(
            rf"$X^* = \frac{{R_0 - 1}}{{R_0 + \frac{{a c {tau_term}}}{{\mu_m}}}} = {result.x_star * 100:.2f}\%$",
            0.068, fontsize=11.5, color=ACCENT_GREEN
        )
        add_row(
            rf"$Y^* = \frac{{a c {tau_term} X^*}}{{a c {tau_term} X^* + \mu_m}} = {result.y_star * 100:.2f}\%$",
            0.062, fontsize=11.5, color=ACCENT_GREEN
        )
    else:
        add_row(rf"$R_0 = {result.R0:.4f} \leq 1 \quad \Longrightarrow \quad \text{{Disease Dies Out}}$", 0.048, fontsize=11.0, bold=True, color=ACCENT_RED)
        add_row(r"$X^* = 0.00\%, \qquad Y^* = 0.00\%$", 0.055, fontsize=11.5, color=ACCENT_RED)

    if ext_result is not None:
        ext_peak = float(np.max(ext_result.x) * 100)
        ext_day = int(np.argmax(ext_result.x))
        add_row(rf"$\text{{Extended Model Peak Infected Humans: }}\ {ext_peak:.2f}\% \text{{ (Day {ext_day})}}$", 0.045, fontsize=10.8, bold=True, color=ACCENT_GREEN)

    return fig


def run_sensitivity_analysis(params: SimulationParameters, out_dir: str) -> Tuple[str, str]:
    """Perform one-at-a-time sensitivity analysis and export CSV + multi-panel PNG."""
    scenarios = [
        ("Biting Rate (a)", "a", [0.15, 0.30, 0.45], ["a=0.15", "a=0.30", "a=0.45"]),
        ("Mosq->Human Prob (b)", "b", [0.25, 0.50, 0.75], ["b=0.25", "b=0.50", "b=0.75"]),
        ("Human Recovery (r)", "r", [0.005, 0.01, 0.02], ["r=0.005", "r=0.01", "r=0.02"]),
        ("Vector Control (M)", "M", [params.M * 1.0, params.M * 0.75, params.M * 0.50], ["0% red.", "25% red.", "50% red."]),
    ]

    csv_path = os.path.join(out_dir, "sensitivity_results.csv")
    png_path = os.path.join(out_dir, "sensitivity_figure.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=120, facecolor=BG_COLOR)
    axes_flat = axes.flatten()
    colors = ["#1F77B4", "#D9381E", "#15803D"]

    rows = []
    for idx, (title, attr, vals, labels) in enumerate(scenarios):
        ax = axes_flat[idx]
        ax.set_facecolor(BG_COLOR)
        for j, (val, lbl) in enumerate(zip(vals, labels)):
            p_mod = SimulationParameters(**{**params.__dict__, attr: val})
            res_b = simulate_transmission(p_mod)
            res_e = simulate_extended(p_mod)

            b_peak = float(np.max(res_b.X))
            b_day = int(np.argmax(res_b.X))
            e_peak = float(np.max(res_e.x))
            e_day = int(np.argmax(res_e.x))

            rows.append([
                title, lbl, f"{val:.4f}",
                f"{b_peak:.6f}", b_day,
                f"{e_peak:.6f}", e_day,
            ])

            ax.plot(res_b.t, res_b.X * 100, color=colors[j], linestyle="-", linewidth=1.8, label=f"Base {lbl}")
            ax.plot(res_e.t, res_e.x * 100, color=colors[j], linestyle="--", linewidth=1.6, label=f"Ext {lbl}")

        ax.set_title(title, fontsize=10, fontweight="bold", color=FG_COLOR)
        ax.set_xlabel("Time (Days)", fontsize=8.5, color=MUTED_COLOR)
        ax.set_ylabel("Infected Humans (%)", fontsize=8.5, color=MUTED_COLOR)
        ax.grid(True, linestyle=":", alpha=0.6, color="#D1D5DB")
        ax.legend(fontsize=7, ncol=2, frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR)

    fig.suptitle("One-at-a-Time Sensitivity Analysis (Solid: Baseline, Dashed: Extended)", fontsize=12, fontweight="bold", color=FG_COLOR)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Parameter", "Scenario", "Value",
            "Baseline_Peak_Infected_Fraction", "Baseline_Peak_Day",
            "Extended_Peak_Infected_Fraction", "Extended_Peak_Day",
        ])
        writer.writerows(rows)

    return csv_path, png_path


# ==============================================================================
# SECTION 4: GRAPHICAL USER INTERFACE (TKINTER)
# ==============================================================================

class MalariaModelApp:
    """Consistent, pure white Tkinter GUI with unified graph, LaTeX panel, checks, and exports."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG_COLOR)

        self.fig_canvas = None
        self.current_fig = None
        self.current_result = None
        self.current_ext_result = None
        self.current_params = None
        self.extended_var = tk.BooleanVar(value=False)

        self._build_header()
        self._build_input_panel()
        self._build_action_bar()
        self._build_kpi_banner()
        self._build_canvas_panel()

        # Automatically execute initial simulation so plot and formulas render on launch
        self.run_model()

    def _build_header(self):
        """Build minimal clean header."""
        header_frame = tk.Frame(self.root, bg=BG_COLOR, pady=5)
        header_frame.pack(fill="x", padx=15)

        lbl_title = tk.Label(header_frame, text=APP_TITLE, font=("Arial", 13, "bold"), fg=FG_COLOR, bg=BG_COLOR)
        lbl_title.pack()
        lbl_sub = tk.Label(header_frame, text=APP_SUBTITLE, font=("Arial", 8), fg=MUTED_COLOR, bg=BG_COLOR)
        lbl_sub.pack(pady=(1, 0))

    def _build_input_panel(self):
        """Build input parameter fields (baseline + climate-driven extended) in a uniformly aligned grid."""
        input_frame = tk.LabelFrame(
            self.root, text="Model Parameters", padx=12, pady=8,
            font=("Arial", 9, "bold"), bg=BG_COLOR, fg=FG_COLOR, relief="solid", bd=1,
        )
        input_frame.pack(fill="x", padx=15, pady=4)

        for col_idx in range(6):
            weight = 1 if col_idx % 2 == 0 else 0
            input_frame.columnconfigure(col_idx, weight=weight)

        self.entries = {}
        entry_style = dict(
            width=11, font=("Arial", 9), bg=BG_COLOR, fg=FG_COLOR,
            relief="solid", bd=1, highlightthickness=1,
            highlightbackground=BORDER_COLOR, highlightcolor=FG_COLOR,
        )

        # Baseline fields in 3 rows x 3 columns
        for idx, (label_text, default_val, key) in enumerate(BASELINE_FIELDS):
            row = idx // 3
            col = (idx % 3) * 2
            lbl = tk.Label(input_frame, text=label_text, anchor="w", font=("Arial", 8, "bold"), bg=BG_COLOR, fg=MUTED_COLOR)
            lbl.grid(row=row, column=col, sticky="w", padx=(6, 4), pady=3)

            ent = tk.Entry(input_frame, **entry_style)
            ent.insert(0, default_val)
            ent.grid(row=row, column=col + 1, sticky="w", padx=(0, 14), pady=3)
            self.entries[key] = ent

        # Subtle horizontal divider
        divider = tk.Frame(input_frame, height=1, bg=BORDER_COLOR)
        divider.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(6, 6))

        # Extended toggle checkbox aligned in left column, climate fields aligned in columns 2..5
        chk_ext = tk.Checkbutton(
            input_frame,
            text="Extended (climate-driven) model",
            variable=self.extended_var,
            font=("Arial", 8, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
            activebackground=BG_COLOR,
            selectcolor=BG_COLOR,
            cursor="hand2",
            command=self.run_model,
        )
        chk_ext.grid(row=4, column=0, columnspan=2, sticky="w", padx=(4, 8), pady=3)

        for idx, (label_text, default_val, key) in enumerate(EXTENDED_FIELDS):
            row = 4 + (idx // 2)
            col = 2 + (idx % 2) * 2
            lbl = tk.Label(input_frame, text=label_text, anchor="w", font=("Arial", 8, "bold"), bg=BG_COLOR, fg=MUTED_COLOR)
            lbl.grid(row=row, column=col, sticky="w", padx=(6, 4), pady=3)

            ent = tk.Entry(input_frame, **entry_style)
            ent.insert(0, default_val)
            ent.grid(row=row, column=col + 1, sticky="w", padx=(0, 14), pady=3)
            self.entries[key] = ent

    def _build_action_bar(self):
        """Build consistent action bar with Clear, Copy LaTeX, Checks, Sensitivity, Exports, and Calculate."""
        bar_frame = tk.Frame(self.root, bg=BG_COLOR)
        bar_frame.pack(fill="x", padx=15, pady=4)

        btn_style = dict(
            bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 8, "bold"),
            activebackground="#F3F4F6", relief="solid", bd=1, padx=10, pady=3, cursor="hand2",
        )

        tk.Button(bar_frame, text="Clear", command=self.clear_fields, **btn_style).pack(side="left")
        tk.Button(bar_frame, text="Copy LaTeX", command=self.copy_latex_code, **btn_style).pack(side="left", padx=(6, 0))
        tk.Button(bar_frame, text="Run checks", command=self.run_checks, **btn_style).pack(side="left", padx=(6, 0))
        tk.Button(bar_frame, text="Sensitivity", command=self.run_sensitivity, **btn_style).pack(side="left", padx=(6, 0))
        tk.Button(bar_frame, text="Export PNG", command=self.export_png, **btn_style).pack(side="left", padx=(6, 0))
        tk.Button(bar_frame, text="Export CSV", command=self.export_baseline_csv, **btn_style).pack(side="left", padx=(6, 0))

        self.lbl_copy_status = tk.Label(bar_frame, text="", font=("Arial", 8, "italic"), bg=BG_COLOR, fg=ACCENT_GREEN)
        self.lbl_copy_status.pack(side="left", padx=8)

        tk.Button(
            bar_frame, text="▶ Calculate & Simulate", width=21, bg=FG_COLOR, fg="#FFFFFF",
            font=("Arial", 9, "bold"), activebackground="#374151", activeforeground="#FFFFFF",
            relief="solid", bd=1, pady=2, cursor="hand2", command=self.run_model,
        ).pack(side="right")

    def _build_kpi_banner(self):
        """Build evenly aligned 3-column horizontal status banner."""
        self.banner_frame = tk.Frame(self.root, bg=CARD_BG, relief="solid", bd=1, padx=12, pady=6)
        self.banner_frame.pack(fill="x", padx=15, pady=4)
        self.banner_frame.columnconfigure(0, weight=1)
        self.banner_frame.columnconfigure(1, weight=1)
        self.banner_frame.columnconfigure(2, weight=1)

        self.lbl_r0 = tk.Label(self.banner_frame, text="R₀: --", font=("Arial", 9, "bold"), bg=CARD_BG, fg=FG_COLOR, anchor="w")
        self.lbl_r0.grid(row=0, column=0, sticky="w")

        self.lbl_status = tk.Label(self.banner_frame, text="Status: --", font=("Arial", 9, "bold"), bg=CARD_BG, fg=MUTED_COLOR, anchor="center")
        self.lbl_status.grid(row=0, column=1, sticky="ew")

        self.lbl_eq = tk.Label(self.banner_frame, text="Analytic Steady-State: --", font=("Arial", 8, "bold"), bg=CARD_BG, fg=MUTED_COLOR, anchor="e")
        self.lbl_eq.grid(row=0, column=2, sticky="e")

    def _build_canvas_panel(self):
        """Build scrollable container for the unified Matplotlib figure (plot + vertical formulas)."""
        outer = tk.Frame(self.root, bg=BG_COLOR)
        outer.pack(fill="both", expand=True, padx=15, pady=4)

        self.scroll_canvas = tk.Canvas(outer, bg=BG_COLOR, highlightthickness=0)
        self.v_scrollbar = tk.Scrollbar(outer, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.v_scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.canvas_container = tk.Frame(self.scroll_canvas, bg=BG_COLOR)
        self.scroll_window = self.scroll_canvas.create_window((0, 0), window=self.canvas_container, anchor="nw")

        self.canvas_container.bind(
            "<Configure>",
            lambda _e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")),
        )
        self.scroll_canvas.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.itemconfigure(self.scroll_window, width=e.width),
        )
        self.scroll_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _parse_params(self) -> Optional[SimulationParameters]:
        """Validate and parse GUI entries into SimulationParameters."""
        try:
            params = SimulationParameters(
                N=float(self.entries["N"].get()),
                M=float(self.entries["M"].get()),
                a=float(self.entries["a"].get()),
                b=float(self.entries["b"].get()),
                c=float(self.entries["c"].get()),
                r=float(self.entries["r"].get()),
                mu_m=float(self.entries["mu_m"].get()),
                tau=float(self.entries["tau"].get()),
                days=int(self.entries["days"].get()),
                T_mean=float(self.entries["T_mean"].get()),
                A_T=float(self.entries["A_T"].get()),
                t_p=float(self.entries["t_p"].get()),
                kappa=float(self.entries["kappa"].get()),
            )
            if params.N <= 0 or params.M <= 0 or params.days <= 0 or params.tau < 0:
                raise ValueError("Populations and days must be positive, and tau >= 0.")
            return params
        except ValueError as err:
            messagebox.showerror("Input Error", f"Please ensure all fields contain valid numerical values.\n({err})")
            return None

    def copy_latex_code(self):
        """Copy the full substituted LaTeX equations to the system clipboard."""
        if not self.current_result:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_result.latex_code)
        self.lbl_copy_status.config(text="✓ LaTeX copied!")
        self.root.after(2200, lambda: self.lbl_copy_status.config(text=""))

    def run_checks(self):
        """Run verify() suite and display pass/fail result."""
        params = self._parse_params()
        if params is None:
            return
        ok, details = verify(params)
        if ok:
            messagebox.showinfo("Verification Checks: PASS", details)
        else:
            messagebox.showerror("Verification Checks: FAIL", details)

    def run_sensitivity(self):
        """Run OAT sensitivity analysis and export CSV + multi-panel PNG."""
        params = self._parse_params()
        if params is None:
            return
        out_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path, png_path = run_sensitivity_analysis(params, out_dir)
        messagebox.showinfo(
            "Sensitivity Analysis Complete",
            f"Exported results:\n• CSV: {csv_path}\n• PNG: {png_path}",
        )

    def export_png(self):
        """Export the current main unified figure as a PNG file."""
        if self.current_fig is None:
            messagebox.showwarning("Export PNG", "No plot available to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialfile="malaria_simulation.png",
        )
        if path:
            self.current_fig.savefig(path, dpi=150, facecolor=BG_COLOR)
            messagebox.showinfo("Export PNG", f"Saved figure to:\n{path}")

    def export_baseline_csv(self):
        """Export baseline model values at days 0, 100, 200, 365 to CSV."""
        params = self._parse_params()
        if params is None:
            return
        target_days = [0, 100, 200, 365]
        p_full = SimulationParameters(**{**params.__dict__, "days": max(params.days, 365)})
        res = simulate_transmission(p_full)

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")],
            initialfile="baseline_table.csv",
        )
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Day", "Infected_Humans_Fraction", "Infected_Humans_Count", "Infected_Mosquitoes_Fraction", "Infected_Mosquitoes_Count"])
                for d in target_days:
                    writer.writerow([d, f"{res.X[d]:.6f}", f"{res.X[d] * params.N:.2f}", f"{res.Y[d]:.6f}", f"{res.Y[d] * params.M:.2f}"])
            messagebox.showinfo("Export CSV", f"Saved baseline table (days 0, 100, 200, 365) to:\n{path}")

    def clear_fields(self):
        """Clear all inputs, reset status labels, and clear the canvas."""
        for ent in self.entries.values():
            ent.delete(0, tk.END)
        self.lbl_r0.config(text="R₀: --")
        self.lbl_status.config(text="Status: --", fg=MUTED_COLOR)
        self.lbl_eq.config(text="Analytic Steady-State: --")
        self.current_result = None
        self.current_ext_result = None
        self._clear_canvas()

    def _clear_canvas(self):
        """Destroy current figure canvas and free memory."""
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()
            self.fig_canvas = None
        if self.current_fig:
            plt.close(self.current_fig)
            self.current_fig = None

    def run_model(self):
        """Validate inputs, run RK4 simulation(s), and render unified plot + LaTeX formulas."""
        params = self._parse_params()
        if params is None:
            return

        self.current_params = params
        self.current_result = simulate_transmission(params)
        self.current_ext_result = simulate_extended(params) if self.extended_var.get() else None

        # Update KPI banner metrics
        self.lbl_r0.config(text=f"Baseline R₀: {self.current_result.R0:.4f}")
        if self.current_result.R0 > 1.0:
            self.lbl_status.config(text="Status: Endemic Persistence (R₀ > 1)", fg=ACCENT_GREEN)
            self.lbl_eq.config(
                text=f"Analytic Steady-State: X* = {self.current_result.x_star * 100:.2f}%, Y* = {self.current_result.y_star * 100:.2f}%"
            )
        else:
            self.lbl_status.config(text="Status: Disease Dies Out (R₀ ≤ 1)", fg=ACCENT_RED)
            self.lbl_eq.config(text="Analytic Steady-State: (X*, Y*) = (0.00%, 0.00%)")

        # Re-render unified canvas (plot on top, LaTeX formulas with inputs below)
        self._clear_canvas()
        self.current_fig = create_unified_figure(params, self.current_result, self.current_ext_result)
        self.fig_canvas = FigureCanvasTkAgg(self.current_fig, master=self.canvas_container)
        self.fig_canvas.draw()
        widget = self.fig_canvas.get_tk_widget()
        widget.configure(bg=BG_COLOR)
        widget.pack(fill="both", expand=True)


# ==============================================================================
# SECTION 5: APPLICATION ENTRY POINT
# ==============================================================================

def main():
    """Initialize and run the Tkinter application main loop."""
    root = tk.Tk()
    app = MalariaModelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
