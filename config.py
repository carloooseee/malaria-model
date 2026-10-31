"""Configuration and default parameters for the Malaria Transmission Model."""

# Window & UI Configuration (Pure white minimal theme)
APP_TITLE = "Malaria Model: Ross-Macdonald Transmission Calculator"
WINDOW_SIZE = "820x850"
BG_COLOR = "#FFFFFF"
FG_COLOR = "#000000"
BORDER_COLOR = "#CCCCCC"

# Default Model Parameters (based on Gordon & Guilfoos Chapter 14)
# Tuple format: (Label text, default value, parameter key)
MODEL_FIELDS = [
    ("Total Human Population (N):", "1000", "N"),
    ("Total Mosquito Population (M):", "5000", "M"),
    ("Biting Rate per Mosquito (a):", "0.25", "a"),
    ("Transmission Prob: Mosquito -> Human (b):", "0.30", "b"),
    ("Transmission Prob: Human -> Mosquito (c):", "0.15", "c"),
    ("Human Recovery Rate (r):", "0.01", "r"),
    ("Mosquito Mortality Rate (mu_m):", "0.10", "mu_m"),
    ("Simulation Time (Days):", "180", "days"),
]

# Simulation Constants
INITIAL_INFECTED_HUMANS_RATIO = 0.01   # 1% initial human infection
INITIAL_INFECTED_MOSQUITOES_RATIO = 0.00  # 0% initial mosquito infection
TIME_STEP = 1.0  # 1 day step
