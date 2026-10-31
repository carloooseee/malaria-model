"""Mathematical model and simulation logic for Ross-Macdonald Malaria Transmission."""

from dataclasses import dataclass
import numpy as np
from config import INITIAL_INFECTED_HUMANS_RATIO, INITIAL_INFECTED_MOSQUITOES_RATIO, TIME_STEP


@dataclass
class SimulationParameters:
    N: float      # Human population
    M: float      # Mosquito population
    a: float      # Biting rate per mosquito
    b: float      # Transmission prob: mosquito -> human
    c: float      # Transmission prob: human -> mosquito
    r: float      # Human recovery rate
    mu_m: float   # Mosquito mortality rate
    days: int     # Simulation duration in days


@dataclass
class SimulationResult:
    R0: float
    m: float
    t: np.ndarray
    X: np.ndarray  # Fraction of infected humans over time
    Y: np.ndarray  # Fraction of infected mosquitoes over time
    summary: str
    days: int


def calculate_r0(m: float, a: float, b: float, c: float, r: float, mu_m: float) -> float:
    """Calculate the Basic Reproduction Number (R0).

    Formula: R0 = (m * a^2 * b * c) / (r * mu_m)
    """
    numerator = m * (a ** 2) * b * c
    denominator = r * mu_m
    return numerator / denominator


def simulate_transmission(params: SimulationParameters) -> SimulationResult:
    """Simulate the Ross-Macdonald difference equations over time."""
    # Mosquito-to-human ratio
    m = params.M / params.N
    R0 = calculate_r0(m, params.a, params.b, params.c, params.r, params.mu_m)

    dt = TIME_STEP
    steps = int(params.days / dt)
    t = np.linspace(0, params.days, steps + 1)

    # Proportions: X (infected human fraction), Y (infected mosquito fraction)
    X = np.zeros(steps + 1)
    Y = np.zeros(steps + 1)
    X[0] = INITIAL_INFECTED_HUMANS_RATIO
    Y[0] = INITIAL_INFECTED_MOSQUITOES_RATIO

    for step in range(steps):
        # Difference equations:
        # dX/dt = m * a * b * Y * (1 - X) - r * X
        # dY/dt = a * c * X * (1 - Y) - mu_m * Y
        dX = (m * params.a * params.b * Y[step] * (1 - X[step]) - params.r * X[step]) * dt
        dY = (params.a * params.c * X[step] * (1 - Y[step]) - params.mu_m * Y[step]) * dt

        X[step + 1] = max(0.0, min(1.0, X[step] + dX))
        Y[step + 1] = max(0.0, min(1.0, Y[step] + dY))

    # Generate step-by-step calculation report
    numerator = m * (params.a ** 2) * params.b * params.c
    denominator = params.r * params.mu_m
    status = "Persistence / Endemic (R0 > 1)" if R0 > 1 else "Disease Dies Out (R0 < 1)"

    summary = (
        f"Basic Reproduction Number (R0): {R0:.4f}\n"
        f"Epidemic Status: {status}\n\n"
        f"Calculation Steps:\n"
        f"1. Mosquito-to-human ratio: m = M / N = {params.M} / {params.N} = {m:.2f}\n"
        f"2. Formula: R0 = (m * a^2 * b * c) / (r * mu_m)\n"
        f"3. Numerator: {m:.2f} * ({params.a}^2) * {params.b} * {params.c} = {numerator:.6f}\n"
        f"4. Denominator: {params.r} * {params.mu_m} = {denominator:.6f}\n"
        f"5. Final Day ({params.days}) Infected Humans: {X[-1] * params.N:.1f} ({X[-1] * 100:.2f}%)\n"
        f"   Final Day ({params.days}) Infected Mosquitoes: {Y[-1] * params.M:.1f} ({Y[-1] * 100:.2f}%)\n"
    )

    return SimulationResult(
        R0=R0,
        m=m,
        t=t,
        X=X,
        Y=Y,
        summary=summary,
        days=params.days
    )
