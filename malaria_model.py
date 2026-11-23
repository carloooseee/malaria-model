"""Malaria Transmission Simulator — Gordon & Guilfoos Chapter 14 Model.

A self-contained single-file Python application implementing the exact 5-compartment
Malaria Modeling Project (Chapter 14 of Gordon & Guilfoos):
  - Humans: Healthy (H), Sick (S), Immune (I)
  - Mosquitoes: Healthy (M_H), Infected (M_I)

Includes 4th-order Runge-Kutta (RK4) integration (dt = 0.1 day, daily output),
all three optional extensions from the project specification:
  1. Relapse / Remission (17% relapse probability)
  2. Seasonal Mosquito Birth & Death Rates (cosine forcing)
  3. Ghana Insecticide-Treated Net (ITN) Policy & Cost-Benefit Analysis
     (calibrated with Asante & Asenso-Okyere, 2003, WHO/AFRO Ghana Report)

Also provides automated verification checks, immunity & bite-rate sensitivity
analysis, vertical LaTeX equation rendering, and PNG/CSV exports.

Requirements:
    pip install numpy matplotlib
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import csv
import math
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
APP_TITLE = "Malaria Modeling Project — Chapter 14 Simulator"
APP_SUBTITLE = "5-Compartment Villager & Mosquito Population Model (RK4) with Seasonal, Relapse & Ghana ITN Extensions"
WINDOW_SIZE = "1020x980"
BG_COLOR = "#FFFFFF"
FG_COLOR = "#111827"
MUTED_COLOR = "#4B5563"
BORDER_COLOR = "#E5E7EB"
CARD_BG = "#FAFAFA"
ACCENT_GREEN = "#15803D"
ACCENT_RED = "#B91C1C"
ACCENT_BLUE = "#1E3A8A"

# Default Chapter 14 Baseline Parameters (exact values from MALARIA-MODELING-PROJECT.pdf)
# Tuple format: (Label text, default value, parameter key)
POPULATION_FIELDS = [
    ("Healthy Humans (H0):", "499", "H0"),
    ("Sick Humans (S0):", "1", "S0"),
    ("Immune Humans (I0):", "0", "I0"),
    ("Healthy Mosq. (MH0):", "1000", "MH0"),
    ("Infected Mosq. (MI0):", "0", "MI0"),
    ("Duration (Days):", "200", "days"),
]

RATE_FIELDS = [
    ("Human Birth Rate:", "0.000092", "birth_rate"),
    ("Human Death Rate:", "0.00001945", "death_rate"),
    ("Malaria Death Rate:", "0.00016123", "malaria_death_rate"),
    ("Human Immunity Rate:", "0.01", "immunity_rate"),
    ("Mosq. Birth Rate:", "0.01", "mosq_birth_rate"),
    ("Mosq. Death Rate:", "0.011", "mosq_death_rate"),
    ("Mosq. Bite Rate:", "0.24", "bite_rate"),
    ("Ghana COI ($/case):", "15.79", "ghana_coi"),
    ("Ghana Family Size:", "5.14", "family_size"),
]

# Simulation & Extension Constants (from MALARIA-MODELING-PROJECT.pdf)
TIME_STEP = 0.1                      # RK4 integration step size (0.1 day)
MAX_MOSQ_BIRTH_RATE = 0.02           # Optional Addition 2: seasonal peak mbr
MAX_MOSQ_DEATH_RATE = 0.022          # Optional Addition 2: seasonal peak mdr
RELAPSE_PROBABILITY = 0.17           # Optional Addition 1: 17% relapse probability
ITN_REDUCTION = 0.50                 # Optional Addition 3: 50% incidence reduction vs no nets
UNTREATED_REDUCTION = 0.1803         # Derived from 39% ITN reduction vs untreated: 1 - 0.50/(1-0.39)
ITN_COST_USD = 3.50                  # Cost per treated net (covers 2 people)
UNTREATED_NET_COST_USD = 2.50        # Cost per untreated net (covers 2 people)
PEOPLE_PER_NET = 2.0                 # Coverage per net


# ==============================================================================
# SECTION 2: MATHEMATICAL MODEL & SIMULATION LOGIC
# ==============================================================================

@dataclass
class SimulationParameters:
    """Input parameters for the Chapter 14 Malaria Modeling Project."""
    H0: float = 499.0                 # Starting healthy villagers
    S0: float = 1.0                   # Starting sick villagers
    I0: float = 0.0                   # Starting immune villagers
    MH0: float = 1000.0               # Starting healthy mosquitoes
    MI0: float = 0.0                  # Starting infected mosquitoes
    days: int = 200                   # Simulation duration (200 or 2000 days)
    birth_rate: float = 0.000092      # Human birth rate /day (0.03358/yr)
    death_rate: float = 0.00001945    # Human natural death rate /day (0.0071/yr)
    malaria_death_rate: float = 0.00016123  # Malaria-induced death rate /day (0.0588486/yr)
    immunity_rate: float = 0.01       # Rate sick villagers gain immunity /day
    mosq_birth_rate: float = 0.01     # Baseline mosquito birth rate /day
    mosq_death_rate: float = 0.011    # Baseline mosquito death rate /day
    bite_rate: float = 0.24           # Mosquito bite rate /day
    ghana_coi: float = 15.79          # Avg household cost/episode in Ghana (US$, Asante & Asenso-Okyere 2003)
    family_size: float = 5.14         # Avg Ghana household size ($13.51 / $2.63 from Ghana report)
    use_seasonal: bool = False        # Optional Addition 2: seasonal cosine mbr/mdr
    use_relapse: bool = False         # Optional Addition 1: 17% relapse probability
    net_policy: str = "none"          # Optional Addition 3: "none", "untreated", or "itn"

    @property
    def recovery_rate(self) -> float:
        """Human recovery rate = 1 - malaria_death_rate (exact Ch. 14 formula)."""
        return 1.0 - self.malaria_death_rate


@dataclass
class SimulationResult:
    """Daily population trajectories, cumulative burden, and LaTeX output."""
    t: np.ndarray
    H: np.ndarray                     # Healthy villagers over time
    S: np.ndarray                     # Sick villagers over time
    I: np.ndarray                     # Immune villagers over time
    MH: np.ndarray                    # Healthy mosquitoes over time
    MI: np.ndarray                    # Infected mosquitoes over time
    cum_infections: float             # Cumulative human malaria episodes
    cum_malaria_deaths: float         # Cumulative malaria-induced deaths
    peak_sick: float                  # Maximum simultaneous sick villagers
    peak_sick_day: int                # Day of peak sick villagers
    days: int
    latex_code: str
    # Economic / ITN Policy metrics (populated when comparing against no-net baseline)
    nets_needed: int = 0
    households: float = 0.0
    net_program_cost: float = 0.0
    cases_averted: float = 0.0
    economic_savings: float = 0.0
    net_benefit: float = 0.0
    bcr: float = 0.0


def _rk4_step(rhs: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Advance state vector y by one step dt using 4th-order Runge-Kutta (RK4)."""
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _seasonal_mosquito_rates(t: float, params: SimulationParameters) -> Tuple[float, float]:
    """Compute mosquito birth rate (mbr) and death rate (mdr) at time t (days)."""
    if not params.use_seasonal:
        return params.mosq_birth_rate, params.mosq_death_rate
    # Exact Optional Addition 2 formulas from MALARIA-MODELING-PROJECT.pdf:
    # mbr = (cos(t * 2*pi / 365) + 1) * max_mosquito_birth_rate / 2
    # mdr = (cos(t * 2*pi / 365) + 1) * max_mosquito_death_rate / 2
    factor = (float(np.cos((t * 2.0 * np.pi) / 365.0)) + 1.0) / 2.0
    return factor * MAX_MOSQ_BIRTH_RATE, factor * MAX_MOSQ_DEATH_RATE


def _effective_bite_rate(params: SimulationParameters) -> float:
    """Adjust effective human infection bite rate based on mosquito net policy (Option 3)."""
    if params.net_policy == "itn":
        return params.bite_rate * (1.0 - ITN_REDUCTION)
    if params.net_policy == "untreated":
        return params.bite_rate * (1.0 - UNTREATED_REDUCTION)
    return params.bite_rate


def simulate_malaria(params: SimulationParameters, dt: float = TIME_STEP) -> SimulationResult:
    """Simulate the 5-compartment Chapter 14 malaria model using RK4 (daily output).

    State vector: [H, S, I, MH, MI, cum_inf, cum_m_deaths]
    """
    substeps = max(1, int(round(1.0 / dt)))
    step_dt = 1.0 / substeps
    rec_rate = params.recovery_rate
    eff_bite = _effective_bite_rate(params)
    # Option 1 (17% relapse): 17% of recovering villagers relapse back into Sick,
    # leaving 83% net recovery back to Healthy.
    net_rec_rate = rec_rate * (1.0 - RELAPSE_PROBABILITY) if params.use_relapse else rec_rate

    t_daily = np.arange(0, params.days + 1, dtype=float)
    H_d = np.zeros(params.days + 1, dtype=float)
    S_d = np.zeros(params.days + 1, dtype=float)
    I_d = np.zeros(params.days + 1, dtype=float)
    MH_d = np.zeros(params.days + 1, dtype=float)
    MI_d = np.zeros(params.days + 1, dtype=float)

    state = np.array([params.H0, params.S0, params.I0, params.MH0, params.MI0, 0.0, 0.0], dtype=float)
    H_d[0], S_d[0], I_d[0], MH_d[0], MI_d[0] = state[:5]

    def rhs(t_val: float, y: np.ndarray) -> np.ndarray:
        H, S, I, MH, MI = max(y[0], 0.0), max(y[1], 0.0), max(y[2], 0.0), max(y[3], 0.0), max(y[4], 0.0)
        humans = H + S + I
        mosq = MH + MI

        mbr, mdr = _seasonal_mosquito_rates(t_val, params)

        births = params.birth_rate * humans
        healthy_dies = H * params.death_rate
        infected = (eff_bite * H * (MI / mosq)) if mosq > 1e-12 else 0.0
        sick_dies = (params.malaria_death_rate + params.death_rate) * S
        malaria_deaths = params.malaria_death_rate * S
        immune_dies = params.death_rate * I
        recover = S * net_rec_rate
        immune = S * params.immunity_rate

        mosq_births = mosq * mbr
        mosq_dies = MH * mdr
        mosq_infection = (S / (humans + 1.0)) * MH * params.bite_rate
        inf_mosq_dies = MI * mdr

        dH = births + recover - healthy_dies - infected
        dS = infected - recover - immune - sick_dies
        dI = immune - immune_dies
        dMH = mosq_births - mosq_dies - mosq_infection
        dMI = mosq_infection - inf_mosq_dies

        return np.array([dH, dS, dI, dMH, dMI, infected, malaria_deaths], dtype=float)

    t_curr = 0.0
    for day in range(params.days):
        for _ in range(substeps):
            state = _rk4_step(rhs, t_curr, state, step_dt)
            state[:5] = np.maximum(state[:5], 0.0)
            t_curr += step_dt
        H_d[day + 1], S_d[day + 1], I_d[day + 1], MH_d[day + 1], MI_d[day + 1] = state[:5]

    peak_sick = float(np.max(S_d))
    peak_sick_day = int(np.argmax(S_d))
    cum_inf = float(state[5])
    cum_deaths = float(state[6])

    # Option 3: Ghana ITN / Untreated Net Cost-Benefit Evaluation
    init_pop = params.H0 + params.S0 + params.I0
    nets_needed = int(math.ceil(init_pop / PEOPLE_PER_NET))
    households = init_pop / max(params.family_size, 1.0)
    net_cost = 0.0
    cases_averted = 0.0
    econ_savings = 0.0
    net_benefit = 0.0
    bcr = 0.0

    if params.net_policy in ("itn", "untreated"):
        unit_cost = ITN_COST_USD if params.net_policy == "itn" else UNTREATED_NET_COST_USD
        net_cost = nets_needed * unit_cost
        base_p = SimulationParameters(**{**params.__dict__, "net_policy": "none"})
        base_res = simulate_malaria(base_p, dt=dt)
        cases_averted = max(0.0, base_res.cum_infections - cum_inf)
        econ_savings = cases_averted * params.ghana_coi
        net_benefit = econ_savings - net_cost
        bcr = econ_savings / net_cost if net_cost > 0 else 0.0

    # LaTeX Export Block
    latex_code = (
        "% ==========================================================\n"
        "% Chapter 14 Malaria Modeling Project — Equations & Summary\n"
        "% ==========================================================\n\n"
        "% 1. Human & Mosquito Compartmental Equations:\n"
        "\\begin{aligned}\n"
        "\\frac{dH}{dt} &= b_h N + r_{\\text{eff}} S - d_h H - \\beta_{\\text{eff}} H \\frac{M_I}{M_H + M_I} \\\\[4pt]\n"
        "\\frac{dS}{dt} &= \\beta_{\\text{eff}} H \\frac{M_I}{M_H + M_I} - (r_{\\text{eff}} + i_h + d_m + d_h) S \\\\[4pt]\n"
        "\\frac{dI}{dt} &= i_h S - d_h I \\\\[4pt]\n"
        "\\frac{dM_H}{dt} &= b_m(t)(M_H + M_I) - d_{\\text{mos}}(t) M_H - \\beta \\frac{S}{N + 1} M_H \\\\[4pt]\n"
        "\\frac{dM_I}{dt} &= \\beta \\frac{S}{N + 1} M_H - d_{\\text{mos}}(t) M_I\n"
        "\\end{aligned}\n\n"
        f"% 2. Simulation Output (Day {params.days}):\n"
        f"% H = {H_d[-1]:.2f}, S = {S_d[-1]:.2f}, I = {I_d[-1]:.2f}, "
        f"M_H = {MH_d[-1]:.2f}, M_I = {MI_d[-1]:.2f}\n"
        f"% Peak Sick = {peak_sick:.2f} (Day {peak_sick_day}), Cumulative Episodes = {cum_inf:.1f}\n"
    )

    return SimulationResult(
        t=t_daily, H=H_d, S=S_d, I=I_d, MH=MH_d, MI=MI_d,
        cum_infections=cum_inf, cum_malaria_deaths=cum_deaths,
        peak_sick=peak_sick, peak_sick_day=peak_sick_day,
        days=params.days, latex_code=latex_code,
        nets_needed=nets_needed, households=households,
        net_program_cost=net_cost, cases_averted=cases_averted,
        economic_savings=econ_savings, net_benefit=net_benefit, bcr=bcr,
    )


def verify(params: Optional[SimulationParameters] = None) -> Tuple[bool, str]:
    """Run automated verification checks on the Chapter 14 RK4 simulator.

    Checks:
      1. Non-negativity & No NaN across all 5 compartments (baseline & all extensions).
      2. Zero-infection invariance: S0=0, MI0=0 stays at S(t)=0, MI(t)=0 for all t.
      3. Step-size convergence: halving dt (0.1 -> 0.05) changes trajectories by < 1e-4.
      4. Policy monotonicity: ITN policy (50% reduction) reduces cumulative cases vs. no nets.
    """
    if params is None:
        params = SimulationParameters(days=200)

    report = []
    try:
        # 1. Non-negativity & NaN check (baseline + seasonal + relapse + ITN)
        res = simulate_malaria(params, dt=0.1)
        for name, arr in [("H", res.H), ("S", res.S), ("I", res.I), ("MH", res.MH), ("MI", res.MI)]:
            assert not np.isnan(arr).any(), f"NaN found in {name}"
            assert np.all(arr >= 0.0), f"Negative population found in {name}"
        p_all = SimulationParameters(**{**params.__dict__, "use_seasonal": True, "use_relapse": True, "net_policy": "itn"})
        res_all = simulate_malaria(p_all, dt=0.1)
        for name, arr in [("H", res_all.H), ("S", res_all.S), ("I", res_all.I), ("MH", res_all.MH), ("MI", res_all.MI)]:
            assert not np.isnan(arr).any() and np.all(arr >= 0.0), f"Invalid values in extended {name}"
        report.append("PASS: Non-negativity (H, S, I, MH, MI >= 0) & No NaN")

        # 2. Zero-infection invariance
        p_zero = SimulationParameters(**{**params.__dict__, "S0": 0.0, "MI0": 0.0})
        res_zero = simulate_malaria(p_zero, dt=0.1)
        assert np.allclose(res_zero.S, 0.0, atol=1e-12), "S did not stay 0 when S0=MI0=0"
        assert np.allclose(res_zero.MI, 0.0, atol=1e-12), "MI did not stay 0 when S0=MI0=0"
        report.append("PASS: Disease-free invariance (S0=0, MI0=0 stays 0)")

        # 3. Step-size halving convergence (< 1e-4)
        res_half = simulate_malaria(params, dt=0.05)
        max_diff = max(
            float(np.max(np.abs(res.H - res_half.H))),
            float(np.max(np.abs(res.S - res_half.S))),
            float(np.max(np.abs(res.I - res_half.I))),
            float(np.max(np.abs(res.MH - res_half.MH))),
            float(np.max(np.abs(res.MI - res_half.MI))),
        )
        assert max_diff < 1e-4, f"dt halving max diff {max_diff:.2e} >= 1e-4"
        report.append(f"PASS: RK4 step-size convergence (dt 0.1->0.05 max diff = {max_diff:.2e} < 1e-4)")

        # 4. ITN effectiveness check
        p_base = SimulationParameters(**{**params.__dict__, "net_policy": "none"})
        p_itn = SimulationParameters(**{**params.__dict__, "net_policy": "itn"})
        r_base = simulate_malaria(p_base, dt=0.1)
        r_itn = simulate_malaria(p_itn, dt=0.1)
        assert r_itn.cum_infections < r_base.cum_infections, "ITN did not reduce cumulative infections"
        pct_drop = (1.0 - r_itn.cum_infections / r_base.cum_infections) * 100.0
        report.append(f"PASS: ITN policy reduces cumulative infections by {pct_drop:.1f}% ({r_base.cum_infections:.1f} -> {r_itn.cum_infections:.1f})")

        return True, "\n".join(report)
    except AssertionError as err:
        report.append(f"FAIL: {err}")
        return False, "\n".join(report)


# ==============================================================================
# SECTION 3: VISUALIZATION & LATEX RENDERING (UNIFIED FIGURE)
# ==============================================================================

def create_unified_figure(params: SimulationParameters, result: SimulationResult) -> Figure:
    """Generate unified figure: Human & Mosquito population plots on top, vertical LaTeX card below."""
    fig = plt.figure(figsize=(9.6, 10.6), dpi=100, facecolor=BG_COLOR)

    # --------------------------------------------------------------------------
    # TOP-LEFT SUBPLOT: Human Population Sectors (Healthy, Sick, Immune)
    # --------------------------------------------------------------------------
    ax_h = fig.add_axes([0.07, 0.67, 0.41, 0.27])
    ax_h.set_facecolor(BG_COLOR)
    ax_h.plot(result.t, result.H, label="Healthy (H)", color="#15803D", linewidth=2.1)
    ax_h.plot(result.t, result.S, label="Sick (S)", color="#D9381E", linewidth=2.1)
    ax_h.plot(result.t, result.I, label="Immune (I)", color="#1F77B4", linestyle="--", linewidth=2.1)
    ax_h.set_title(f"Human Population Sectors ({result.days} Days)", fontsize=10.5, fontweight="bold", color=FG_COLOR, pad=6)
    ax_h.set_xlabel("Time (Days)", fontsize=9, color=MUTED_COLOR)
    ax_h.set_ylabel("Villagers (Count)", fontsize=9, color=MUTED_COLOR)
    ax_h.grid(True, linestyle=":", alpha=0.6, color="#D1D5DB")
    ax_h.legend(frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR, fontsize=8.2)
    ax_h.tick_params(colors=MUTED_COLOR, labelsize=8.2)

    # --------------------------------------------------------------------------
    # TOP-RIGHT SUBPLOT: Mosquito Population Sectors (Healthy, Infected)
    # --------------------------------------------------------------------------
    ax_m = fig.add_axes([0.56, 0.67, 0.40, 0.27])
    ax_m.set_facecolor(BG_COLOR)
    ax_m.plot(result.t, result.MH, label="Healthy Mosq. (M_H)", color="#0D9488", linewidth=2.1)
    ax_m.plot(result.t, result.MI, label="Infected Mosq. (M_I)", color="#9333EA", linestyle="--", linewidth=2.1)
    ax_m.set_title(f"Mosquito Population Sectors ({result.days} Days)", fontsize=10.5, fontweight="bold", color=FG_COLOR, pad=6)
    ax_m.set_xlabel("Time (Days)", fontsize=9, color=MUTED_COLOR)
    ax_m.set_ylabel("Mosquitoes (Count)", fontsize=9, color=MUTED_COLOR)
    ax_m.grid(True, linestyle=":", alpha=0.6, color="#D1D5DB")
    ax_m.legend(frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR, fontsize=8.2)
    ax_m.tick_params(colors=MUTED_COLOR, labelsize=8.2)

    # --------------------------------------------------------------------------
    # BOTTOM SUBPLOT: Vertical Step-by-Step LaTeX Formulation & Summary Card
    # --------------------------------------------------------------------------
    ax_f = fig.add_axes([0.04, 0.02, 0.92, 0.58])
    ax_f.set_facecolor(BG_COLOR)
    ax_f.axis("off")
    ax_f.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=BORDER_COLOR, linewidth=1.2))

    y_pos = 0.96

    def add_row(text: str, dy: float, fontsize: float = 10.5, bold: bool = False, color: str = FG_COLOR, x: float = 0.05):
        nonlocal y_pos
        ax_f.text(x, y_pos, text, fontsize=fontsize, fontweight=("bold" if bold else "normal"), color=color, va="top")
        y_pos -= dy

    # 1. Inputs & Active Extensions
    add_row("1. Chapter 14 Model Inputs & Active Extensions:", 0.038, fontsize=10.2, bold=True, x=0.03)
    add_row(
        rf"$H_0={params.H0:.0f},\ S_0={params.S0:.0f},\ I_0={params.I0:.0f},\ "
        rf"M_{{H0}}={params.MH0:.0f},\ M_{{I0}}={params.MI0:.0f},\ \text{{Days}}={params.days}$",
        0.040, fontsize=9.8, color="#1F2937"
    )
    add_row(
        rf"$b_h={params.birth_rate},\ d_h={params.death_rate},\ d_m={params.malaria_death_rate},\ "
        rf"r_h=1-d_m={params.recovery_rate:.6f},\ i_h={params.immunity_rate},\ \beta={params.bite_rate}$",
        0.040, fontsize=9.6, color="#1F2937"
    )
    ext_tags = []
    if params.use_seasonal:
        ext_tags.append("Seasonal mbr/mdr (Opt 2)")
    if params.use_relapse:
        ext_tags.append("17% Relapse (Opt 1)")
    if params.net_policy != "none":
        ext_tags.append(f"Ghana {params.net_policy.upper()} Policy (Opt 3)")
    ext_label = ", ".join(ext_tags) if ext_tags else "Baseline Model Only (No Extensions Enabled)"
    add_row(f"Active Extensions: {ext_label}", 0.042, fontsize=9.5, bold=True, color=ACCENT_BLUE)

    # 2. Governing System (Vertical)
    y_pos -= 0.008
    add_row("2. Governing 5-Compartment Equations (RK4, dt = 0.1 day):", 0.038, fontsize=10.2, bold=True, color=MUTED_COLOR, x=0.03)
    add_row(r"$\frac{dH}{dt} = b_h N + r_{\mathrm{eff}} S - d_h H - \beta_{\mathrm{eff}} H \frac{M_I}{M_H + M_I}$", 0.056, fontsize=11.0, color=ACCENT_BLUE)
    add_row(r"$\frac{dS}{dt} = \beta_{\mathrm{eff}} H \frac{M_I}{M_H + M_I} - (r_{\mathrm{eff}} + i_h + d_m + d_h) S$", 0.056, fontsize=11.0, color=ACCENT_BLUE)
    add_row(r"$\frac{dI}{dt} = i_h S - d_h I$", 0.048, fontsize=11.0, color=ACCENT_BLUE)
    add_row(r"$\frac{dM_H}{dt} = b_m(t)(M_H + M_I) - d_{\mathrm{mos}}(t) M_H - \beta \frac{S}{N + 1} M_H$", 0.056, fontsize=11.0, color=ACCENT_BLUE)
    add_row(r"$\frac{dM_I}{dt} = \beta \frac{S}{N + 1} M_H - d_{\mathrm{mos}}(t) M_I$", 0.056, fontsize=11.0, color=ACCENT_BLUE)

    if params.use_seasonal:
        add_row(
            r"$\text{Seasonal Forcing: }\quad b_m(t)=\frac{\cos(2\pi t/365)+1}{2}(0.02),\quad d_{\mathrm{mos}}(t)=\frac{\cos(2\pi t/365)+1}{2}(0.022)$",
            0.052, fontsize=9.8, color="#4B5563"
        )

    # 3. Simulation Results at Final Day
    y_pos -= 0.008
    add_row(f"3. Population Breakdown at Day {params.days} & Disease Burden:", 0.038, fontsize=10.2, bold=True, color=MUTED_COLOR, x=0.03)
    total_h = result.H[-1] + result.S[-1] + result.I[-1]
    total_m = result.MH[-1] + result.MI[-1]
    add_row(
        rf"$\text{{Humans: }}\ H={result.H[-1]:.1f},\ S={result.S[-1]:.2f},\ I={result.I[-1]:.1f}\ (\text{{Total }} N={total_h:.1f})$",
        0.044, fontsize=10.4
    )
    add_row(
        rf"$\text{{Mosquitoes: }}\ M_H={result.MH[-1]:.1f},\ M_I={result.MI[-1]:.1f}\ (\text{{Total }} M={total_m:.1f})$",
        0.044, fontsize=10.4
    )
    add_row(
        rf"$\text{{Peak Sick: }}\ {result.peak_sick:.2f}\ \text{{(Day {result.peak_sick_day})}}\quad|\quad "
        rf"\text{{Cumulative Cases: }}\ {result.cum_infections:.1f}\quad|\quad \text{{Malaria Deaths: }}\ {result.cum_malaria_deaths:.2f}$",
        0.048, fontsize=10.2, bold=True
    )

    # 4. Ghana ITN Policy Cost-Benefit (when enabled)
    if params.net_policy in ("itn", "untreated"):
        y_pos -= 0.006
        policy_name = "Insecticide-Treated Nets (ITN, 50% red., $3.50)" if params.net_policy == "itn" else "Untreated Nets (18% red., $2.50)"
        add_row(f"4. Ghana Village Cost-Benefit Analysis — {policy_name}:", 0.038, fontsize=10.2, bold=True, color=ACCENT_GREEN, x=0.03)
        add_row(
            f"Nets (2/net): {result.nets_needed} (~{result.households:.1f} Ghana households)  |  Cost: ${result.net_program_cost:,.2f}",
            0.038, fontsize=9.4, color=FG_COLOR
        )
        add_row(
            f"Cases Averted: {result.cases_averted:,.1f}  |  Saved (@${params.ghana_coi:.2f}): ${result.economic_savings:,.2f}  |  Net: ${result.net_benefit:,.2f}  |  BCR: {result.bcr:.1f}x",
            0.038, fontsize=9.0, bold=True, color=ACCENT_GREEN
        )

    return fig


def run_sensitivity_analysis(params: SimulationParameters, out_dir: str) -> Tuple[str, str]:
    """Perform required Chapter 14 sensitivity analysis on Immunity Rate and Bite Rate.

    Exports sensitivity_results.csv and multi-panel sensitivity_figure.png.
    """
    scenarios = [
        ("Human Immunity Rate (i_h)", "immunity_rate", [0.005, 0.01, 0.02, 0.05], ["i_h=0.005", "i_h=0.01", "i_h=0.02", "i_h=0.05"]),
        ("Mosquito Bite Rate (beta)", "bite_rate", [0.12, 0.24, 0.36, 0.48], ["beta=0.12", "beta=0.24", "beta=0.36", "beta=0.48"]),
    ]

    csv_path = os.path.join(out_dir, "sensitivity_results.csv")
    png_path = os.path.join(out_dir, "sensitivity_figure.png")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), dpi=120, facecolor=BG_COLOR)
    colors = ["#1F77B4", "#D9381E", "#15803D", "#9333EA"]
    rows = []

    for row_idx, (title, attr, vals, labels) in enumerate(scenarios):
        ax_sick = axes[row_idx, 0]
        ax_imm = axes[row_idx, 1]
        ax_sick.set_facecolor(BG_COLOR)
        ax_imm.set_facecolor(BG_COLOR)

        for j, (val, lbl) in enumerate(zip(vals, labels)):
            p_mod = SimulationParameters(**{**params.__dict__, attr: val})
            res = simulate_malaria(p_mod)
            rows.append([
                title, lbl, f"{val:.5f}", params.days,
                f"{res.peak_sick:.2f}", res.peak_sick_day,
                f"{res.cum_infections:.1f}", f"{res.H[-1]:.1f}", f"{res.S[-1]:.2f}", f"{res.I[-1]:.1f}",
            ])
            ax_sick.plot(res.t, res.S, color=colors[j], linewidth=1.9, label=lbl)
            ax_imm.plot(res.t, res.I, color=colors[j], linewidth=1.9, label=lbl)

        ax_sick.set_title(f"{title} — Sick Villagers (S)", fontsize=10, fontweight="bold", color=FG_COLOR)
        ax_sick.set_xlabel("Time (Days)", fontsize=8.5, color=MUTED_COLOR)
        ax_sick.set_ylabel("Sick Villagers", fontsize=8.5, color=MUTED_COLOR)
        ax_sick.grid(True, linestyle=":", alpha=0.6, color="#D1D5DB")
        ax_sick.legend(fontsize=8, frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR)

        ax_imm.set_title(f"{title} — Immune Villagers (I)", fontsize=10, fontweight="bold", color=FG_COLOR)
        ax_imm.set_xlabel("Time (Days)", fontsize=8.5, color=MUTED_COLOR)
        ax_imm.set_ylabel("Immune Villagers", fontsize=8.5, color=MUTED_COLOR)
        ax_imm.grid(True, linestyle=":", alpha=0.6, color="#D1D5DB")
        ax_imm.legend(fontsize=8, frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR)

    fig.suptitle(f"Chapter 14 Parameter Sensitivity Analysis ({params.days} Days)", fontsize=12, fontweight="bold", color=FG_COLOR)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Parameter", "Scenario", "Value", "Horizon_Days",
            "Peak_Sick", "Peak_Sick_Day", "Cumulative_Cases",
            "Final_Healthy", "Final_Sick", "Final_Immune",
        ])
        writer.writerows(rows)

    return csv_path, png_path


# ==============================================================================
# SECTION 4: GRAPHICAL USER INTERFACE (TKINTER)
# ==============================================================================

class MalariaModelApp:
    """Pure white Tkinter GUI for the Chapter 14 Malaria Modeling Project."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG_COLOR)

        self.fig_canvas = None
        self.current_fig = None
        self.current_result = None
        self.current_params = None

        self.seasonal_var = tk.BooleanVar(value=False)
        self.relapse_var = tk.BooleanVar(value=False)
        self.net_policy_var = tk.StringVar(value="none")

        self._build_header()
        self._build_input_panel()
        self._build_action_bar()
        self._build_kpi_banner()
        self._build_canvas_panel()

        self.run_model()

    def _build_header(self):
        """Build minimal clean header."""
        header_frame = tk.Frame(self.root, bg=BG_COLOR, pady=5)
        header_frame.pack(fill="x", padx=15)
        tk.Label(header_frame, text=APP_TITLE, font=("Arial", 13, "bold"), fg=FG_COLOR, bg=BG_COLOR).pack()
        tk.Label(header_frame, text=APP_SUBTITLE, font=("Arial", 8), fg=MUTED_COLOR, bg=BG_COLOR).pack(pady=(1, 0))

    def _build_input_panel(self):
        """Build aligned grid for Chapter 14 populations, rates, and optional extensions."""
        input_frame = tk.LabelFrame(
            self.root, text="Chapter 14 Model Parameters & Optional Extensions", padx=12, pady=8,
            font=("Arial", 9, "bold"), bg=BG_COLOR, fg=FG_COLOR, relief="solid", bd=1,
        )
        input_frame.pack(fill="x", padx=15, pady=4)

        for col_idx in range(6):
            input_frame.columnconfigure(col_idx, weight=(1 if col_idx % 2 == 0 else 0))

        self.entries = {}
        entry_style = dict(
            width=11, font=("Arial", 9), bg=BG_COLOR, fg=FG_COLOR,
            relief="solid", bd=1, highlightthickness=1,
            highlightbackground=BORDER_COLOR, highlightcolor=FG_COLOR,
        )

        # Rows 0-1: Initial populations & duration
        for idx, (label_text, default_val, key) in enumerate(POPULATION_FIELDS):
            row = idx // 3
            col = (idx % 3) * 2
            tk.Label(input_frame, text=label_text, anchor="w", font=("Arial", 8, "bold"), bg=BG_COLOR, fg=MUTED_COLOR).grid(
                row=row, column=col, sticky="w", padx=(6, 4), pady=2
            )
            ent = tk.Entry(input_frame, **entry_style)
            ent.insert(0, default_val)
            ent.grid(row=row, column=col + 1, sticky="w", padx=(0, 12), pady=2)
            self.entries[key] = ent

        # Rows 2-4: Empirical rates & Ghana economic parameters
        for idx, (label_text, default_val, key) in enumerate(RATE_FIELDS):
            row = 2 + (idx // 3)
            col = (idx % 3) * 2
            tk.Label(input_frame, text=label_text, anchor="w", font=("Arial", 8, "bold"), bg=BG_COLOR, fg=MUTED_COLOR).grid(
                row=row, column=col, sticky="w", padx=(6, 4), pady=2
            )
            ent = tk.Entry(input_frame, **entry_style)
            ent.insert(0, default_val)
            ent.grid(row=row, column=col + 1, sticky="w", padx=(0, 12), pady=2)
            self.entries[key] = ent

        # Divider
        tk.Frame(input_frame, height=1, bg=BORDER_COLOR).grid(row=5, column=0, columnspan=6, sticky="ew", pady=(6, 5))

        # Row 6: Optional Extensions Checkboxes & Ghana Net Policy Selector
        ext_bar = tk.Frame(input_frame, bg=BG_COLOR)
        ext_bar.grid(row=6, column=0, columnspan=6, sticky="ew")

        tk.Checkbutton(
            ext_bar, text="Opt 1: 17% Relapse", variable=self.relapse_var,
            font=("Arial", 8, "bold"), bg=BG_COLOR, fg=FG_COLOR, activebackground=BG_COLOR,
            selectcolor=BG_COLOR, cursor="hand2", command=self.run_model,
        ).pack(side="left", padx=(4, 12))

        tk.Checkbutton(
            ext_bar, text="Opt 2: Seasonal Mosquitoes (cos mbr/mdr)", variable=self.seasonal_var,
            font=("Arial", 8, "bold"), bg=BG_COLOR, fg=FG_COLOR, activebackground=BG_COLOR,
            selectcolor=BG_COLOR, cursor="hand2", command=self.run_model,
        ).pack(side="left", padx=(0, 16))

        tk.Label(ext_bar, text="Opt 3 (Ghana Nets):", font=("Arial", 8, "bold"), bg=BG_COLOR, fg=MUTED_COLOR).pack(side="left", padx=(4, 4))
        for text, val in [("None", "none"), ("Untreated ($2.50)", "untreated"), ("ITN ($3.50, -50%)", "itn")]:
            tk.Radiobutton(
                ext_bar, text=text, variable=self.net_policy_var, value=val,
                font=("Arial", 8), bg=BG_COLOR, fg=FG_COLOR, activebackground=BG_COLOR,
                selectcolor=BG_COLOR, cursor="hand2", command=self.run_model,
            ).pack(side="left", padx=3)

    def _build_action_bar(self):
        """Build consistent action bar with 200d/2000d presets, checks, sensitivity, and exports."""
        bar = tk.Frame(self.root, bg=BG_COLOR)
        bar.pack(fill="x", padx=15, pady=4)

        btn_style = dict(
            bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 8, "bold"),
            activebackground="#F3F4F6", relief="solid", bd=1, padx=8, pady=3, cursor="hand2",
        )

        tk.Button(bar, text="200d", command=lambda: self.set_days_and_run(200), **btn_style).pack(side="left")
        tk.Button(bar, text="2000d", command=lambda: self.set_days_and_run(2000), **btn_style).pack(side="left", padx=(5, 0))
        tk.Button(bar, text="Copy LaTeX", command=self.copy_latex_code, **btn_style).pack(side="left", padx=(5, 0))
        tk.Button(bar, text="Run checks", command=self.run_checks, **btn_style).pack(side="left", padx=(5, 0))
        tk.Button(bar, text="Sensitivity", command=self.run_sensitivity, **btn_style).pack(side="left", padx=(5, 0))
        tk.Button(bar, text="Export PNG", command=self.export_png, **btn_style).pack(side="left", padx=(5, 0))
        tk.Button(bar, text="Export CSV", command=self.export_table_csv, **btn_style).pack(side="left", padx=(5, 0))

        self.lbl_copy_status = tk.Label(bar, text="", font=("Arial", 8, "italic"), bg=BG_COLOR, fg=ACCENT_GREEN)
        self.lbl_copy_status.pack(side="left", padx=8)

        tk.Button(
            bar, text="▶ Calculate & Simulate", width=20, bg=FG_COLOR, fg="#FFFFFF",
            font=("Arial", 9, "bold"), activebackground="#374151", activeforeground="#FFFFFF",
            relief="solid", bd=1, pady=2, cursor="hand2", command=self.run_model,
        ).pack(side="right")

    def _build_kpi_banner(self):
        """Build 3-column horizontal KPI status banner."""
        self.banner_frame = tk.Frame(self.root, bg=CARD_BG, relief="solid", bd=1, padx=12, pady=6)
        self.banner_frame.pack(fill="x", padx=15, pady=4)
        for col in range(3):
            self.banner_frame.columnconfigure(col, weight=1)

        self.lbl_humans = tk.Label(self.banner_frame, text="Humans (H/S/I): --", font=("Arial", 8, "bold"), bg=CARD_BG, fg=FG_COLOR, anchor="w")
        self.lbl_humans.grid(row=0, column=0, sticky="w")

        self.lbl_peak = tk.Label(self.banner_frame, text="Peak Sick: --", font=("Arial", 8, "bold"), bg=CARD_BG, fg=ACCENT_RED, anchor="center")
        self.lbl_peak.grid(row=0, column=1, sticky="ew")

        self.lbl_mosq = tk.Label(self.banner_frame, text="Mosquitoes (MH/MI): --", font=("Arial", 8, "bold"), bg=CARD_BG, fg=MUTED_COLOR, anchor="e")
        self.lbl_mosq.grid(row=0, column=2, sticky="e")

    def _build_canvas_panel(self):
        """Build vertically scrollable container for the unified figure."""
        outer = tk.Frame(self.root, bg=BG_COLOR)
        outer.pack(fill="both", expand=True, padx=15, pady=4)

        self.scroll_canvas = tk.Canvas(outer, bg=BG_COLOR, highlightthickness=0)
        self.v_scrollbar = tk.Scrollbar(outer, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.v_scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.canvas_container = tk.Frame(self.scroll_canvas, bg=BG_COLOR)
        self.scroll_window = self.scroll_canvas.create_window((0, 0), window=self.canvas_container, anchor="nw")

        self.canvas_container.bind("<Configure>", lambda _e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.bind("<Configure>", lambda e: self.scroll_canvas.itemconfigure(self.scroll_window, width=e.width))
        self.scroll_canvas.bind_all("<MouseWheel>", lambda e: self.scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def set_days_and_run(self, days: int):
        """Quickly switch duration (200 or 2000 days) and re-run."""
        self.entries["days"].delete(0, tk.END)
        self.entries["days"].insert(0, str(days))
        self.run_model()

    def _parse_params(self) -> Optional[SimulationParameters]:
        """Parse and validate GUI inputs."""
        try:
            params = SimulationParameters(
                H0=float(self.entries["H0"].get()),
                S0=float(self.entries["S0"].get()),
                I0=float(self.entries["I0"].get()),
                MH0=float(self.entries["MH0"].get()),
                MI0=float(self.entries["MI0"].get()),
                days=int(self.entries["days"].get()),
                birth_rate=float(self.entries["birth_rate"].get()),
                death_rate=float(self.entries["death_rate"].get()),
                malaria_death_rate=float(self.entries["malaria_death_rate"].get()),
                immunity_rate=float(self.entries["immunity_rate"].get()),
                mosq_birth_rate=float(self.entries["mosq_birth_rate"].get()),
                mosq_death_rate=float(self.entries["mosq_death_rate"].get()),
                bite_rate=float(self.entries["bite_rate"].get()),
                ghana_coi=float(self.entries["ghana_coi"].get()),
                family_size=float(self.entries["family_size"].get()),
                use_seasonal=self.seasonal_var.get(),
                use_relapse=self.relapse_var.get(),
                net_policy=self.net_policy_var.get(),
            )
            if params.days <= 0 or (params.H0 + params.S0 + params.I0) <= 0:
                raise ValueError("Duration and total human population must be positive.")
            return params
        except ValueError as err:
            messagebox.showerror("Input Error", f"Please ensure all fields contain valid numbers.\n({err})")
            return None

    def copy_latex_code(self):
        """Copy LaTeX equations to clipboard."""
        if not self.current_result:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_result.latex_code)
        self.lbl_copy_status.config(text="✓ LaTeX copied!")
        self.root.after(2200, lambda: self.lbl_copy_status.config(text=""))

    def run_checks(self):
        """Run verify() suite and show pass/fail summary."""
        params = self._parse_params()
        if params is None:
            return
        ok, details = verify(params)
        if ok:
            messagebox.showinfo("Verification Checks: PASS", details)
        else:
            messagebox.showerror("Verification Checks: FAIL", details)

    def run_sensitivity(self):
        """Run immunity-rate and bite-rate sensitivity analysis and export CSV + PNG."""
        params = self._parse_params()
        if params is None:
            return
        out_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path, png_path = run_sensitivity_analysis(params, out_dir)
        messagebox.showinfo("Sensitivity Analysis Complete", f"Exported:\n• CSV: {csv_path}\n• PNG: {png_path}")

    def export_png(self):
        """Export current unified figure as PNG."""
        if self.current_fig is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")], initialfile="malaria_ch14_plot.png")
        if path:
            self.current_fig.savefig(path, dpi=150, facecolor=BG_COLOR)
            messagebox.showinfo("Export PNG", f"Saved figure to:\n{path}")

    def export_table_csv(self):
        """Export population breakdown at key days (0, 50, 100, 200, 500, 1000, 2000) to CSV."""
        params = self._parse_params()
        if params is None:
            return
        target_days = [0, 50, 100, 200, 500, 1000, 2000]
        p_full = SimulationParameters(**{**params.__dict__, "days": max(params.days, 2000)})
        res = simulate_malaria(p_full)

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")], initialfile="ch14_population_table.csv")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Day", "Healthy_Humans", "Sick_Humans", "Immune_Humans", "Total_Humans", "Healthy_Mosquitoes", "Infected_Mosquitoes", "Total_Mosquitoes"])
                for d in target_days:
                    tot_h = res.H[d] + res.S[d] + res.I[d]
                    tot_m = res.MH[d] + res.MI[d]
                    writer.writerow([d, f"{res.H[d]:.2f}", f"{res.S[d]:.2f}", f"{res.I[d]:.2f}", f"{tot_h:.2f}", f"{res.MH[d]:.2f}", f"{res.MI[d]:.2f}", f"{tot_m:.2f}"])
            messagebox.showinfo("Export CSV", f"Saved 2000-day population table to:\n{path}")

    def _clear_canvas(self):
        """Destroy current figure canvas and free memory."""
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()
            self.fig_canvas = None
        if self.current_fig:
            plt.close(self.current_fig)
            self.current_fig = None

    def run_model(self):
        """Validate inputs, run RK4 simulation, and render plots + vertical LaTeX card."""
        params = self._parse_params()
        if params is None:
            return

        self.current_params = params
        self.current_result = simulate_malaria(params)

        # Update KPI banner
        self.lbl_humans.config(
            text=f"Day {params.days} Humans — H: {self.current_result.H[-1]:.1f}, S: {self.current_result.S[-1]:.1f}, I: {self.current_result.I[-1]:.1f}"
        )
        self.lbl_peak.config(
            text=f"Peak Sick: {self.current_result.peak_sick:.1f} (Day {self.current_result.peak_sick_day}) | Cum. Cases: {self.current_result.cum_infections:.0f}"
        )
        self.lbl_mosq.config(
            text=f"Mosquitoes — MH: {self.current_result.MH[-1]:.1f}, MI: {self.current_result.MI[-1]:.1f}"
        )

        self._clear_canvas()
        self.current_fig = create_unified_figure(params, self.current_result)
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
