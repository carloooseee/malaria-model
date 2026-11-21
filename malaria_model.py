"""Malaria Transmission Simulator (Ross-Macdonald Mathematical Model).

A unified, self-contained single-file Python application implementing the classic
Ross-Macdonald epidemic model. It calculates the Basic Reproduction Number (R0),
simulates coupled host-vector difference equations over time, and presents a cohesive
visual interface where the dynamic transmission graph and the full LaTeX analytical
formulas (with exact user inputs substituted) are rendered together on a single canvas.

Requirements:
    pip install numpy matplotlib
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from dataclasses import dataclass
from typing import Optional
import tkinter as tk
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ==============================================================================

# Graphical User Interface (GUI) Settings (Pure white minimal theme)
APP_TITLE = "Malaria Model: Ross-Macdonald Transmission Simulator"
APP_SUBTITLE = "Host-Vector Mathematical Model, Numerical Dynamics & Analytical Solution"
WINDOW_SIZE = "940x940"
BG_COLOR = "#FFFFFF"
FG_COLOR = "#111827"
MUTED_COLOR = "#4B5563"
BORDER_COLOR = "#E5E7EB"
CARD_BG = "#FAFAFA"
ACCENT_GREEN = "#15803D"
ACCENT_RED = "#B91C1C"

# Default Model Parameters (based on Gordon & Guilfoos Chapter 14)
# Tuple format: (Label text, default value, parameter key)
MODEL_FIELDS = [
    ("Human Population (N):", "1000", "N"),
    ("Mosquito Population (M):", "5000", "M"),
    ("Biting Rate (a):", "0.25", "a"),
    ("Trans. Prob: M -> H (b):", "0.30", "b"),
    ("Trans. Prob: H -> M (c):", "0.15", "c"),
    ("Human Recovery (r):", "0.01", "r"),
    ("Mosquito Mortality (μ_m):", "0.10", "mu_m"),
    ("Duration (Days):", "180", "days"),
]

# Simulation Constants
INITIAL_INFECTED_HUMANS_RATIO = 0.01      # 1% of human population initially infected
INITIAL_INFECTED_MOSQUITOES_RATIO = 0.00  # 0% of mosquito population initially infected
TIME_STEP = 1.0                           # Discrete numerical integration step (1 day)


# ==============================================================================
# SECTION 2: MATHEMATICAL MODEL & SIMULATION LOGIC
# ==============================================================================

@dataclass
class SimulationParameters:
    """Input parameters for the Ross-Macdonald malaria transmission model."""
    N: float      # Total human population size
    M: float      # Total mosquito population size
    a: float      # Biting rate (average bites per mosquito per day)
    b: float      # Transmission efficiency: infectious mosquito to susceptible human
    c: float      # Transmission efficiency: infectious human to susceptible mosquito
    r: float      # Human recovery rate (1/r = duration of human infectiousness in days)
    mu_m: float   # Mosquito mortality rate (1/mu_m = average mosquito lifespan in days)
    days: int     # Total simulation duration in days


@dataclass
class SimulationResult:
    """Simulation output including trajectories, steady-state equilibria, and LaTeX expressions."""
    R0: float
    m: float
    t: np.ndarray
    X: np.ndarray             # Proportion of infected humans over time
    Y: np.ndarray             # Proportion of infected mosquitoes over time
    latex_code: str           # Ready-to-copy LaTeX code block
    days: int
    numerator: float          # Vector transmission numerator: m * a^2 * b * c
    denominator: float        # Loss denominator: r * mu_m
    x_star: Optional[float]   # Theoretical human endemic equilibrium
    y_star: Optional[float]   # Theoretical mosquito endemic equilibrium


def calculate_r0(m: float, a: float, b: float, c: float, r: float, mu_m: float) -> float:
    r"""Calculate the Basic Reproduction Number (R0).

    Mathematical Formulation:
        R_0 = \frac{m \cdot a^2 \cdot b \cdot c}{r \cdot \mu_m}

    Where:
        m: Mosquito-to-human ratio (M / N)
        a: Daily biting rate per mosquito
        b: Transmission probability: mosquito -> human per bite
        c: Transmission probability: human -> mosquito per bite
        r: Human recovery rate
        mu_m: Mosquito mortality rate
    """
    numerator = m * (a ** 2) * b * c
    denominator = r * mu_m
    return numerator / denominator


def simulate_transmission(params: SimulationParameters) -> SimulationResult:
    r"""Numerically integrate the Ross-Macdonald coupled difference equations.

    Governing Differential System:
        dX/dt = m * a * b * Y(t) * (1 - X(t)) - r * X(t)
        dY/dt = a * c * X(t) * (1 - Y(t)) - \mu_m * Y(t)
    """
    # Step 1: Compute mosquito-to-human ratio and Basic Reproduction Number
    m = params.M / params.N
    numerator = m * (params.a ** 2) * params.b * params.c
    denominator = params.r * params.mu_m
    R0 = numerator / denominator

    # Step 2: Compute theoretical steady-state endemic equilibrium (t -> infinity)
    if R0 > 1.0:
        ab_over_r = (params.a * params.b) / params.r
        x_star = (R0 - 1.0) / (R0 + ab_over_r)
        y_star = (params.a * params.c * x_star) / (params.a * params.c * x_star + params.mu_m)
    else:
        x_star = 0.0
        y_star = 0.0

    # Step 3: Run discrete time-step simulation over specified days
    dt = TIME_STEP
    steps = int(params.days / dt)
    t = np.linspace(0, params.days, steps + 1)

    X = np.zeros(steps + 1)
    Y = np.zeros(steps + 1)
    X[0] = INITIAL_INFECTED_HUMANS_RATIO
    Y[0] = INITIAL_INFECTED_MOSQUITOES_RATIO

    for step in range(steps):
        # Difference equations approximating dX/dt and dY/dt
        dX = (m * params.a * params.b * Y[step] * (1.0 - X[step]) - params.r * X[step]) * dt
        dY = (params.a * params.c * X[step] * (1.0 - Y[step]) - params.mu_m * Y[step]) * dt

        X[step + 1] = max(0.0, min(1.0, X[step] + dX))
        Y[step + 1] = max(0.0, min(1.0, Y[step] + dY))

    # Step 4: Format clean LaTeX source code block for export
    latex_code = (
        "% ==========================================================\n"
        "% Ross-Macdonald Malaria Model: Formulas & Substituted Solution\n"
        "% ==========================================================\n\n"
        "% 1. Model Inputs Provided:\n"
        f"% N = {params.N:.0f}, M = {params.M:.0f}, a = {params.a}, b = {params.b}, "
        f"c = {params.c}, r = {params.r}, mu_m = {params.mu_m}, Days = {params.days}\n\n"
        "% 2. Governing Differential Equations:\n"
        "\\begin{aligned}\n"
        "\\frac{dX}{dt} &= m \\cdot a \\cdot b \\cdot Y(t)\\,(1 - X(t)) - r \\cdot X(t) \\\\[6pt]\n"
        "\\frac{dY}{dt} &= a \\cdot c \\cdot X(t)\\,(1 - Y(t)) - \\mu_m \\cdot Y(t)\n"
        "\\end{aligned}\n\n"
        "% 3. Mosquito-to-Human Population Ratio:\n"
        f"m = \\frac{{M}}{{N}} = \\frac{{{params.M:.0f}}}{{{params.N:.0f}}} = {m:.4f}\n\n"
        "% 4. Basic Reproduction Number (R_0) & Analytical Evaluation:\n"
        "\\begin{aligned}\n"
        f"R_0 &= \\frac{{m \\cdot a^2 \\cdot b \\cdot c}}{{r \\cdot \\mu_m}} \\\\[6pt]\n"
        f"    &= \\frac{{{m:.4f} \\times ({params.a})^2 \\times {params.b} \\times {params.c}}}{{{params.r} \\times {params.mu_m}}} \\\\[6pt]\n"
        f"    &= \\frac{{{numerator:.7f}}}{{{denominator:.7f}}} \\\\[6pt]\n"
        f"    &= {R0:.4f} \\quad \\left(R_0 {' > 1 \\implies \\text{Endemic Persistence}' if R0 > 1 else ' \\leq 1 \\implies \\text{Disease Dies Out}'}\\right)\n"
        "\\end{aligned}\n\n"
        "% 5. Theoretical Steady-State Equilibrium:\n"
        "\\begin{aligned}\n"
        f"X^* &= \\frac{{R_0 - 1}}{{R_0 + \\frac{{a \\cdot b}}{{r}}}} = {x_star:.4f} \\quad ({x_star*100:.2f}\\%)\\\\[6pt]\n"
        f"Y^* &= \\frac{{a \\cdot c \\cdot X^*}}{{a \\cdot c \\cdot X^* + \\mu_m}} = {y_star:.4f} \\quad ({y_star*100:.2f}\\%)\n"
        "\\end{aligned}\n"
    )

    return SimulationResult(
        R0=R0,
        m=m,
        t=t,
        X=X,
        Y=Y,
        latex_code=latex_code,
        days=params.days,
        numerator=numerator,
        denominator=denominator,
        x_star=x_star,
        y_star=y_star,
    )


# ==============================================================================
# SECTION 3: VISUALIZATION & LATEX RENDERING (UNIFIED FIGURE)
# ==============================================================================

def create_unified_figure(params: SimulationParameters, result: SimulationResult) -> Figure:
    """Generate a single unified figure: transmission plot on top, LaTeX formulas with inputs below."""
    fig = plt.figure(figsize=(9.0, 6.2), dpi=100, facecolor=BG_COLOR)

    # --------------------------------------------------------------------------
    # TOP SUBPLOT: Dynamic Transmission Curves
    # --------------------------------------------------------------------------
    ax_plot = fig.add_axes([0.08, 0.54, 0.88, 0.39])
    ax_plot.set_facecolor(BG_COLOR)

    ax_plot.plot(result.t, result.X * 100, label="Infected Humans (%)", color="#D9381E", linewidth=2.2)
    ax_plot.plot(result.t, result.Y * 100, label="Infected Mosquitoes (%)", color="#1F77B4", linestyle="--", linewidth=2.2)

    ax_plot.set_title(
        f"Malaria Transmission Dynamics over {result.days} Days",
        fontsize=11, fontweight="bold", color=FG_COLOR, pad=6
    )
    ax_plot.set_xlabel("Time (Days)", fontsize=9, color=MUTED_COLOR)
    ax_plot.set_ylabel("Infected Proportion (%)", fontsize=9, color=MUTED_COLOR)
    ax_plot.grid(True, linestyle=":", alpha=0.6, color="#D1D5DB")
    ax_plot.legend(frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR, fontsize=8.5, loc="center right")
    ax_plot.tick_params(colors=MUTED_COLOR, labelsize=8.5)

    # --------------------------------------------------------------------------
    # BOTTOM SUBPLOT: LaTeX Formulas with Inputs Given
    # --------------------------------------------------------------------------
    ax_formula = fig.add_axes([0.03, 0.02, 0.94, 0.45])
    ax_formula.set_facecolor(BG_COLOR)
    ax_formula.axis("off")

    # Clean outer card border
    card_rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax_formula.add_patch(card_rect)

    # 1. Inputs Given
    ax_formula.text(0.03, 0.88, "1. Model Inputs Provided:", fontsize=8.8, fontweight="bold", color=FG_COLOR)
    inputs_math = (
        rf"$N = {params.N:.0f},\quad M = {params.M:.0f},\quad a = {params.a},\quad "
        rf"b = {params.b},\quad c = {params.c},\quad r = {params.r},\quad "
        rf"\mu_m = {params.mu_m},\quad \text{{Days}} = {params.days}$"
    )
    ax_formula.text(0.04, 0.74, inputs_math, fontsize=9.2, color="#1F2937")

    # 2. Governing ODEs
    ax_formula.text(0.03, 0.60, "2. Governing Ross-Macdonald Differential Equations:", fontsize=8.5, fontweight="bold", color=MUTED_COLOR)
    ax_formula.text(0.04, 0.46, r"$\frac{dX}{dt} = m \cdot a \cdot b \cdot Y(1 - X) - r \cdot X$", fontsize=9.5, color="#1E3A8A")
    ax_formula.text(0.52, 0.46, r"$\frac{dY}{dt} = a \cdot c \cdot X(1 - Y) - \mu_m \cdot Y$", fontsize=9.5, color="#1E3A8A")

    # 3. R0 Formula & Analytical Solution with Inputs
    ax_formula.text(
        0.03, 0.32,
        r"3. Basic Reproduction Number ($R_0$) & Analytical Derivation with Inputs Substituted:",
        fontsize=8.5, fontweight="bold", color=MUTED_COLOR
    )
    sol_math = (
        rf"$m = \frac{{M}}{{N}} = \frac{{{params.M:.0f}}}{{{params.N:.0f}}} = {result.m:.2f} \qquad "
        rf"R_0 = \frac{{m \cdot a^2 \cdot b \cdot c}}{{r \cdot \mu_m}} = "
        rf"\frac{{{result.m:.2f} \times ({params.a})^2 \times {params.b} \times {params.c}}}{{{params.r} \times {params.mu_m}}} = "
        rf"\frac{{{result.numerator:.6f}}}{{{result.denominator:.6f}}} = \mathbf{{{result.R0:.4f}}}$"
    )
    ax_formula.text(0.04, 0.18, sol_math, fontsize=8.8, color=FG_COLOR)

    # 4. Outcome & Equilibrium
    if result.R0 > 1.0 and result.x_star is not None and result.y_star is not None:
        outcome_str = (
            rf"4. Epidemiological Outcome: $R_0 = {result.R0:.4f} > 1$ (Endemic Persistence) $\quad \Longrightarrow \quad "
            rf"X^* = \frac{{R_0 - 1}}{{R_0 + \frac{{ab}}{{r}}}} = {result.x_star * 100:.1f}\%,\quad "
            rf"Y^* = {result.y_star * 100:.1f}\% \text{{ (Theoretical Steady State)}}$"
        )
        color = ACCENT_GREEN
    else:
        outcome_str = rf"4. Epidemiological Outcome: $R_0 = {result.R0:.4f} \leq 1$ (Disease Dies Out) $\quad \Longrightarrow \quad X^* = 0.00\%,\quad Y^* = 0.00\%$"
        color = ACCENT_RED

    ax_formula.text(0.03, 0.05, outcome_str, fontsize=8.8, fontweight="bold", color=color)

    return fig


# ==============================================================================
# SECTION 4: GRAPHICAL USER INTERFACE (TKINTER)
# ==============================================================================

class MalariaModelApp:
    """Consistent, pure white Tkinter GUI with unified graph and LaTeX formula display."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG_COLOR)

        self.fig_canvas = None
        self.current_fig = None
        self.current_result = None

        self._build_header()
        self._build_input_panel()
        self._build_action_bar()
        self._build_kpi_banner()
        self._build_canvas_panel()

        # Automatically execute initial simulation so plot and formulas are rendered on launch
        self.run_model()

    def _build_header(self):
        """Build minimal clean header."""
        header_frame = tk.Frame(self.root, bg=BG_COLOR, pady=6)
        header_frame.pack(fill="x", padx=15)

        lbl_title = tk.Label(
            header_frame,
            text=APP_TITLE,
            font=("Arial", 13, "bold"),
            fg=FG_COLOR,
            bg=BG_COLOR,
        )
        lbl_title.pack()

        lbl_sub = tk.Label(
            header_frame,
            text=APP_SUBTITLE,
            font=("Arial", 8),
            fg=MUTED_COLOR,
            bg=BG_COLOR,
        )
        lbl_sub.pack(pady=(2, 0))

    def _build_input_panel(self):
        """Build input parameter fields in a clean, consistent 4-column grid."""
        input_frame = tk.LabelFrame(
            self.root,
            text="Model Parameters",
            padx=12,
            pady=8,
            font=("Arial", 9, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
            relief="solid",
            bd=1,
        )
        input_frame.pack(fill="x", padx=15, pady=4)

        self.entries = {}
        for idx, (label_text, default_val, key) in enumerate(MODEL_FIELDS):
            row = idx // 4
            col = (idx % 4) * 2

            lbl = tk.Label(
                input_frame,
                text=label_text,
                anchor="w",
                font=("Arial", 8, "bold"),
                bg=BG_COLOR,
                fg=MUTED_COLOR,
            )
            lbl.grid(row=row, column=col, sticky="w", padx=6, pady=3)

            ent = tk.Entry(
                input_frame,
                width=10,
                font=("Arial", 9),
                bg=BG_COLOR,
                fg=FG_COLOR,
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground=BORDER_COLOR,
                highlightcolor=FG_COLOR,
            )
            ent.insert(0, default_val)
            ent.grid(row=row, column=col + 1, sticky="w", padx=4, pady=3)
            self.entries[key] = ent

    def _build_action_bar(self):
        """Build consistent action bar with buttons for Clear, Copy LaTeX, and Calculate."""
        bar_frame = tk.Frame(self.root, bg=BG_COLOR)
        bar_frame.pack(fill="x", padx=15, pady=4)

        btn_clear = tk.Button(
            bar_frame,
            text="Clear",
            width=10,
            bg=BG_COLOR,
            fg=FG_COLOR,
            font=("Arial", 8),
            activebackground="#F3F4F6",
            relief="solid",
            bd=1,
            cursor="hand2",
            command=self.clear_fields,
        )
        btn_clear.pack(side="left")

        btn_copy = tk.Button(
            bar_frame,
            text="📋 Copy LaTeX Code",
            font=("Arial", 8),
            bg=BG_COLOR,
            fg=FG_COLOR,
            activebackground="#F3F4F6",
            relief="solid",
            bd=1,
            padx=10,
            cursor="hand2",
            command=self.copy_latex_code,
        )
        btn_copy.pack(side="left", padx=10)

        self.lbl_copy_status = tk.Label(
            bar_frame,
            text="",
            font=("Arial", 8, "italic"),
            bg=BG_COLOR,
            fg=ACCENT_GREEN,
        )
        self.lbl_copy_status.pack(side="left")

        btn_calc = tk.Button(
            bar_frame,
            text="▶ Calculate & Simulate",
            width=22,
            bg=FG_COLOR,
            fg="#FFFFFF",
            font=("Arial", 9, "bold"),
            activebackground="#374151",
            activeforeground="#FFFFFF",
            relief="solid",
            bd=1,
            cursor="hand2",
            command=self.run_model,
        )
        btn_calc.pack(side="right")

    def _build_kpi_banner(self):
        """Build horizontal status banner displaying quick summary metrics."""
        self.banner_frame = tk.Frame(
            self.root,
            bg=CARD_BG,
            relief="solid",
            bd=1,
            padx=12,
            pady=5,
        )
        self.banner_frame.pack(fill="x", padx=15, pady=4)

        self.lbl_r0 = tk.Label(
            self.banner_frame,
            text="Basic Reproduction Number (R₀): --",
            font=("Arial", 9, "bold"),
            bg=CARD_BG,
            fg=FG_COLOR,
        )
        self.lbl_r0.pack(side="left", padx=(0, 16))

        self.lbl_status = tk.Label(
            self.banner_frame,
            text="Status: --",
            font=("Arial", 9, "bold"),
            bg=CARD_BG,
            fg=MUTED_COLOR,
        )
        self.lbl_status.pack(side="left", padx=(0, 16))

        self.lbl_eq = tk.Label(
            self.banner_frame,
            text="Theoretical Steady-State: --",
            font=("Arial", 8),
            bg=CARD_BG,
            fg=MUTED_COLOR,
        )
        self.lbl_eq.pack(side="right")

    def _build_canvas_panel(self):
        """Build container for the unified Matplotlib figure canvas (plot + formulas below)."""
        self.canvas_container = tk.Frame(self.root, bg=BG_COLOR)
        self.canvas_container.pack(fill="both", expand=True, padx=15, pady=4)

    def copy_latex_code(self):
        """Copy the full substituted LaTeX equations to the system clipboard."""
        if not self.current_result:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_result.latex_code)
        self.lbl_copy_status.config(text="✓ LaTeX code copied to clipboard!")
        self.root.after(2200, lambda: self.lbl_copy_status.config(text=""))

    def clear_fields(self):
        """Clear all inputs, reset status labels, and clear the canvas."""
        for ent in self.entries.values():
            ent.delete(0, tk.END)
        self.lbl_r0.config(text="Basic Reproduction Number (R₀): --")
        self.lbl_status.config(text="Status: --", fg=MUTED_COLOR)
        self.lbl_eq.config(text="Theoretical Steady-State: --")
        self.current_result = None
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
        """Validate inputs, run simulation, and render graph with formulas below it."""
        try:
            params = SimulationParameters(
                N=float(self.entries["N"].get()),
                M=float(self.entries["M"].get()),
                a=float(self.entries["a"].get()),
                b=float(self.entries["b"].get()),
                c=float(self.entries["c"].get()),
                r=float(self.entries["r"].get()),
                mu_m=float(self.entries["mu_m"].get()),
                days=int(self.entries["days"].get()),
            )

            if params.N <= 0 or params.M <= 0 or params.days <= 0:
                raise ValueError("Populations and days must be positive values.")

        except ValueError as err:
            messagebox.showerror(
                "Input Error",
                f"Please ensure all fields contain valid numerical values.\n({err})",
            )
            return

        # Run mathematical integration
        self.current_result = simulate_transmission(params)

        # Update KPI banner metrics
        self.lbl_r0.config(text=f"Basic Reproduction Number (R₀): {self.current_result.R0:.4f}")
        if self.current_result.R0 > 1.0:
            self.lbl_status.config(text="Status: Endemic Persistence (R₀ > 1)", fg=ACCENT_GREEN)
            self.lbl_eq.config(
                text=f"Theoretical Steady-State: Humans X* = {self.current_result.x_star * 100:.1f}%, Vectors Y* = {self.current_result.y_star * 100:.1f}%"
            )
        else:
            self.lbl_status.config(text="Status: Disease Dies Out (R₀ ≤ 1)", fg=ACCENT_RED)
            self.lbl_eq.config(text="Theoretical Steady-State: (X*, Y*) = (0.0%, 0.0%)")

        # Re-render unified canvas (plot on top, LaTeX formulas with inputs below)
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
