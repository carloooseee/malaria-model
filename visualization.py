"""Plotting and visualization utilities for the Malaria Transmission Model."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def create_simulation_figure(t: np.ndarray, X: np.ndarray, Y: np.ndarray, days: int) -> Figure:
    """Generate a Matplotlib Figure depicting transmission dynamics over time on a clean white canvas."""
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100, facecolor="white")
    ax.set_facecolor("white")

    # Colors are retained here as this is the data visualization section where color is necessary to distinguish curves
    ax.plot(t, X * 100, label="Infected Humans (%)", color="#D9381E", linewidth=2)
    ax.plot(t, Y * 100, label="Infected Mosquitoes (%)", color="#1F77B4", linestyle="--", linewidth=2)

    ax.set_title(f"Malaria Transmission Dynamics over {days} Days", color="black", fontsize=11)
    ax.set_xlabel("Time (Days)", color="black")
    ax.set_ylabel("Infected Proportion (%)", color="black")
    ax.tick_params(colors="black")
    ax.grid(True, linestyle=":", alpha=0.5, color="#D0D0D0")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC")

    fig.tight_layout()
    return fig
