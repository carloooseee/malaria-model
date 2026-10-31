"""Graphical User Interface (GUI) for the Malaria Transmission Simulator.

Designed with a clean, pure white minimal style without unnecessary colors.
"""

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    APP_TITLE,
    WINDOW_SIZE,
    BG_COLOR,
    FG_COLOR,
    BORDER_COLOR,
    MODEL_FIELDS,
)
from simulation import SimulationParameters, simulate_transmission
from visualization import create_simulation_figure


class MalariaModelApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG_COLOR)

        self.current_fig = None
        self.canvas = None

        self._build_header()
        self._build_input_panel()
        self._build_action_buttons()
        self._build_results_panel()
        self._build_plot_panel()

    def _build_header(self):
        """Build clean white title header."""
        title_frame = tk.Frame(self.root, bg=BG_COLOR, pady=10)
        title_frame.pack(fill="x")
        lbl_title = tk.Label(
            title_frame,
            text=APP_TITLE,
            font=("Arial", 13, "bold"),
            fg=FG_COLOR,
            bg=BG_COLOR,
        )
        lbl_title.pack()

    def _build_input_panel(self):
        """Build input parameter fields with pure white style."""
        input_frame = tk.LabelFrame(
            self.root,
            text="Model Parameters",
            padx=12,
            pady=10,
            font=("Arial", 9, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
            relief="solid",
            bd=1,
        )
        input_frame.pack(fill="x", padx=15, pady=6)

        self.entries = {}
        for idx, (label_text, default_val, key) in enumerate(MODEL_FIELDS):
            row = idx // 2
            col = (idx % 2) * 2
            lbl = tk.Label(
                input_frame,
                text=label_text,
                anchor="w",
                font=("Arial", 9),
                bg=BG_COLOR,
                fg=FG_COLOR,
            )
            lbl.grid(row=row, column=col, sticky="w", padx=6, pady=4)

            ent = tk.Entry(
                input_frame,
                width=12,
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
            ent.grid(row=row, column=col + 1, sticky="w", padx=6, pady=4)
            self.entries[key] = ent

    def _build_action_buttons(self):
        """Build neutral action buttons with clean border."""
        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(fill="x", padx=15, pady=6)

        btn_clear = tk.Button(
            btn_frame,
            text="Clear",
            width=12,
            bg=BG_COLOR,
            fg=FG_COLOR,
            activebackground="#F0F0F0",
            relief="solid",
            bd=1,
            cursor="hand2",
            command=self.clear_fields,
        )
        btn_clear.pack(side="left")

        btn_calc = tk.Button(
            btn_frame,
            text="Calculate & Simulate",
            width=20,
            bg=BG_COLOR,
            fg=FG_COLOR,
            font=("Arial", 9, "bold"),
            activebackground="#F0F0F0",
            relief="solid",
            bd=1,
            cursor="hand2",
            command=self.run_model,
        )
        btn_calc.pack(side="right")

    def _build_results_panel(self):
        """Build clean white text area for results and calculation steps."""
        out_frame = tk.LabelFrame(
            self.root,
            text="Results & Calculations",
            padx=12,
            pady=8,
            font=("Arial", 9, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
            relief="solid",
            bd=1,
        )
        out_frame.pack(fill="x", padx=15, pady=6)

        self.txt_result = tk.Text(
            out_frame,
            height=9,
            font=("Consolas", 9),
            wrap="word",
            bg=BG_COLOR,
            fg=FG_COLOR,
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=FG_COLOR,
        )
        self.txt_result.pack(fill="both")

    def _build_plot_panel(self):
        """Build container for the Matplotlib figure canvas."""
        self.plot_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.plot_frame.pack(fill="both", expand=True, padx=15, pady=6)

    def clear_fields(self):
        """Clear all inputs, outputs, and active plot."""
        for ent in self.entries.values():
            ent.delete(0, tk.END)
        self.txt_result.delete("1.0", tk.END)
        self._clear_canvas()

    def _clear_canvas(self):
        """Destroy current canvas and free matplotlib figure memory."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.current_fig:
            plt.close(self.current_fig)
            self.current_fig = None

    def run_model(self):
        """Validate input parameters, run simulation, and display results."""
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

        # Run simulation logic
        result = simulate_transmission(params)

        # Update text output
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert("1.0", result.summary)

        # Render plot with white background
        self._clear_canvas()
        self.current_fig = create_simulation_figure(result.t, result.X, result.Y, result.days)
        self.canvas = FigureCanvasTkAgg(self.current_fig, master=self.plot_frame)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=BG_COLOR)
        widget.pack(fill="both", expand=True)
