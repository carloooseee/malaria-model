# Malaria Transmission Model (Ross-Macdonald)

A Python application implementing the Ross-Macdonald mathematical model to calculate the Basic Reproduction Number ($R_0$) and simulate malaria transmission dynamics between human and mosquito populations over time.

---

## Project Structure

The project has been split into dedicated modules following separation of concerns:

```
malaria-model/
├── config.py             # Model parameter metadata, default values, and UI styling
├── simulation.py         # Mathematical calculations (R0) and difference equation simulation
├── visualization.py      # Matplotlib figure and chart generation
├── gui.py                # Tkinter graphical user interface and event handlers
├── main.py               # Main application entry point
├── malaria_model.py      # Backward-compatible launcher
└── requirements.txt      # Python dependencies (numpy, matplotlib)
```

### Module Responsibilities:
- **[config.py](file:///c:/Programming/malaria-model/config.py)**: Contains all configuration parameters, default model values, colors, and window settings.
- **[simulation.py](file:///c:/Programming/malaria-model/simulation.py)**: Houses pure simulation and mathematical logic (`calculate_r0`, `simulate_transmission`). Decoupled from Tkinter, making it easy to test or use in command-line scripts.
- **[visualization.py](file:///c:/Programming/malaria-model/visualization.py)**: Handles rendering of the simulation graphs with Matplotlib.
- **[gui.py](file:///c:/Programming/malaria-model/gui.py)**: Manages window layout, text views, user interaction, error handling, and canvas embedding.
- **[main.py](file:///c:/Programming/malaria-model/main.py)**: Initializes and starts the application.

---

## How to Run

### 1. Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### 2. Run with One Command
You can run the entire application using either command:

```bash
python main.py
```
*(or `python malaria_model.py`)*
