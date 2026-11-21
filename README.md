# Malaria Transmission Model (Ross-Macdonald)

A unified, single-file Python application implementing the classic **Ross-Macdonald epidemic model** to calculate the Basic Reproduction Number ($R_0$) and simulate malaria transmission dynamics between human and mosquito populations over time.

Featuring a cohesive, pure-white design where the **transmission dynamics graph** and the **full LaTeX mathematical derivation (with user inputs substituted)** are presented together on a single unified canvas.

> 📖 **Full Technical & Mathematical Documentation:** See [`DOCUMENTATION.md`](file:///c:/Programming/malaria-model/DOCUMENTATION.md) for in-depth derivations, stability analysis, parameter references, and architecture details.

---

## How to Run

### 1. Prerequisites
Ensure you have **Python 3.8+** installed along with `numpy` and `matplotlib`:

```bash
pip install -r requirements.txt
```
*(or manually install via: `pip install numpy matplotlib`)*

> [!NOTE]
> No external LaTeX installation (such as MiKTeX or TeXLive) is required. Matplotlib's built-in **MathText** engine renders all mathematical notation natively.

### 2. Launch the Application
Run the standalone single-file script:

```bash
python malaria_model.py
```

---

## Unified Design & Interface

The application adopts a minimalist, pure-white theme with consistent typography, borders, and controls:

1. **Header**: Clean title and description of the Ross-Macdonald host-vector model.
2. **Model Parameters Panel**: Compact 4-column parameter grid with default values based on Gordon & Guilfoos.
3. **Action & Quick Status Bar**:
   - `Clear`: Resets all fields and clears plots.
   - `📋 Copy LaTeX Code`: Copies publication-ready LaTeX equations and substituted steps to the clipboard with visual confirmation.
   - `▶ Calculate & Simulate`: Computes trajectories and updates the display.
   - **KPI Status Banner**: Instant summary showing $R_0$, epidemic status (Endemic vs. Dies Out), and theoretical steady-state equilibria ($X^*, Y^*$).
4. **Unified Visual Canvas (No Tabs / No Separation)**:
   - **Top Subplot**: Interactive dynamic transmission curves over time ($X(t)$ infected humans % and $Y(t)$ infected mosquitoes %).
   - **Bottom Subplot**: Clean card rendering the LaTeX equations with the **exact user inputs substituted in**:
     - Given model inputs: $N, M, a, b, c, r, \mu_m, \text{Days}$
     - Governing Ross-Macdonald differential equations
     - Basic Reproduction Number ($R_0$) formulation and numerical substitution
     - Epidemiological outcome and theoretical endemic equilibrium

---

## Mathematical Formulation & Solution

### 1. Governing System of Differential Equations

The Ross-Macdonald model captures the cyclical vector-borne transmission of malaria between human hosts ($X$) and female *Anopheles* mosquitoes ($Y$):

$$\frac{dX}{dt} = m \cdot a \cdot b \cdot Y(t) \cdot (1 - X(t)) - r \cdot X(t)$$

$$\frac{dY}{dt} = a \cdot c \cdot X(t) \cdot (1 - Y(t)) - \mu_m \cdot Y(t)$$

Where:
- $X(t)$: Fraction of infected humans ($0 \leq X \leq 1$)
- $Y(t)$: Fraction of infected mosquitoes ($0 \leq Y \leq 1$)
- $m = \frac{M}{N}$: Ratio of female mosquitoes to humans
- $a$: Daily biting rate per mosquito
- $b$: Transmission probability: infectious mosquito $\to$ susceptible human per bite
- $c$: Transmission probability: infectious human $\to$ susceptible mosquito per bite
- $r$: Human recovery rate per day ($1/r$ is the average duration of human infectiousness)
- $\mu_m$: Mosquito mortality rate per day ($1/\mu_m$ is the average mosquito lifespan)

---

### 2. Basic Reproduction Number ($R_0$)

The Basic Reproduction Number ($R_0$) represents the expected number of secondary cases caused by one infected individual introduced into an entirely susceptible population:

$$R_0 = \frac{m \cdot a^2 \cdot b \cdot c}{r \cdot \mu_m}$$

#### Analytical Derivation with Substituted Inputs:
1. **Vector-to-Human Ratio**:
   $$m = \frac{M}{N}$$

2. **Numerator (Vectorial Transmission Potential)**:
   $$\text{Numerator} = m \cdot a^2 \cdot b \cdot c$$

3. **Denominator (Combined Loss & Clearance Rate)**:
   $$\text{Denominator} = r \cdot \mu_m$$

4. **Epidemiological Threshold Evaluation**:
   - If $R_0 > 1$: **Endemic Persistence**. Malaria persists in the host-vector system. Theoretical steady-state:
     $$X^* = \frac{R_0 - 1}{R_0 + \frac{a \cdot b}{r}}, \qquad Y^* = \frac{a \cdot c \cdot X^*}{a \cdot c \cdot X^* + \mu_m}$$
   - If $R_0 \leq 1$: **Disease Dies Out**. The disease-free equilibrium $(X^*, Y^*) = (0, 0)$ is stable.

---

## Default Parameters & Descriptions

| Parameter | Symbol | Default Value | Description |
| :--- | :---: | :---: | :--- |
| Human Population | $N$ | `1000` | Total human population size |
| Mosquito Population | $M$ | `5000` | Total female mosquito population size |
| Biting Rate | $a$ | `0.25` | Bites per mosquito per day (1 bite every 4 days) |
| Mosquito $\to$ Human Transmission | $b$ | `0.30` | Infection efficiency from mosquito to human |
| Human $\to$ Mosquito Transmission | $c$ | `0.15` | Infection efficiency from human to mosquito |
| Human Recovery Rate | $r$ | `0.01` | Daily recovery rate (mean infectious period $\approx 100$ days) |
| Mosquito Mortality Rate | $\mu_m$ | `0.10` | Daily mortality rate (mean mosquito lifespan $\approx 10$ days) |
| Simulation Duration | $T$ | `180` | Duration of dynamic integration in days |

---

## Code Sections in [`malaria_model.py`](file:///c:/Programming/malaria-model/malaria_model.py)

| Section | Description |
| :--- | :--- |
| **Section 1: Configuration & Constants** | Window dimensions, pure white minimal UI theme, default parameter definitions, and initial conditions. |
| **Section 2: Mathematical Model & Logic** | Dataclasses (`SimulationParameters`, `SimulationResult`), $R_0$ computation, coupled difference equation integration, and LaTeX export generation. |
| **Section 3: Visualization & LaTeX Rendering** | Single unified figure (`create_unified_figure`): transmission dynamics curves on top, LaTeX formulas with substituted inputs below. |
| **Section 4: Graphical User Interface** | Pure white Tkinter interface (`MalariaModelApp`), 4-column parameter grid, action bar, KPI banner, and unified canvas embedding. |
| **Section 5: Application Entry Point** | Standard `main()` function and execution guard. |
