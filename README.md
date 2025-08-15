# numerical-option-pricing-duke-energy

Methods and code to study **numerical option pricing** using **Duke Energy Corporation (DUK)** daily closing prices. We develop an end‑to‑end methodology to price European call options under the **Black–Scholes** framework, **validating GBM assumptions** before pricing. Besides, data provided contains closing prices of different companies which follow GBM assumptions.

> This repository accompanies an academic project. It focuses on methodology and reproducibility rather than business KPIs.

## Project Structure

```
numerical-option-pricing-duke-energy/
├─ code/
│ └─ matlab/
│ └─ Proyecto_EstocasticosII_AreizaParraSerna.mlx
├─ data/
│ └─ raw/
│ └─ datos.xlsx
├─ docs/
│ └─ Procesos_Estocasticos_II_Proyecto.pdf
├─ .github/
│ └─ workflows/ # (optional CI, empty by default)
├─ .gitignore
├─ LICENSE
├─ CITATION.cff
└─ README.md
```


## Goals
- Validate **GBM** assumptions on DUK closing prices.
- Implement and compare numerical methods for European call pricing:
  - Black–Scholes (closed form)
  - Monte Carlo simulation
  - Binomial tree
  - Explicit finite‑difference scheme
- Provide a clear, reproducible pipeline in **MATLAB Live Script** (`.mlx`).

## Data
- **Source**: Yahoo Finance
- **Range**: 2023‑05‑15 to 2025‑05‑09 (daily).
- `data/raw/datos.xlsx`: input dataset with closing prices of several companies.
> If you later fetch data via APIs, keep raw dumps in `data/raw/` and document the source here.

## Requirements
- **MATLAB R2024b** (Live Script `.mlx`).
  - Typical toolboxes: *Statistics and others*, *Financial Toolbox* (if available).
- Optional: Python/R for auxiliary checks.

## How to Run (MATLAB)
1. Open `code/matlab/Proyecto_EstocasticosII_AreizaParraSerna.mlx`.
2. Ensure `data/raw/datos.xlsx` is available.
3. Run the Live Script sections from top to bottom.

## Methodology (high‑level)
1. **Exploratory analysis** of prices/returns; highlight outliers and news events.
2. **Assumption checks for GBM** on returns:
   - Partial autocorrelation (PACF) — no significant autocorrelation.
     Hurst exponent and fractal dimension indicating trend‑like behavior.
   - Normality tests (Jarque–Bera, KS, Anderson–Darling, Lilliefors).
   - Homoskedasticity checks (ARCH and White tests).
   - Cumulative volatility profile.
3. **Parameter estimation** for GBM; **in‑sample forecasting** with 1,000 simulated paths and MAPE as accuracy criterion.
4. **Risk‑neutral valuation**:
   - Price European calls using **Black–Scholes**, **Monte Carlo**, **Binomial**, **Explicit Finite Differences**.
   - **Risk‑free rate**: r = 0.05. Initial spot: last observed price.
   - Maturities **T ∈ {1, 1/2, 1/3, 1/4}** (years).
   - Strikes **K1…K5** derived from simulated projections (see `docs`).

## Results (summary)
- **Model fit**: in‑sample **MAPE ≈ 1.2%** (good forecast accuracy).
- **Pricing**: methods broadly agree according to Black-Scholes equation.
- **Efficiency**: explicit finite differences is the most computationally expensive, especially for larger **T** and **K** (grid grows with domain size).
- **Caveat**: when the option’s reference price is extremely small (deep OTM), tiny absolute errors translate into large **relative** errors.

See tables and figures in `docs/Procesos_Estocasticos_II_Proyecto.pdf` for detailed numbers.

## Reproducibility notes
- MATLAB R2024b; some statistical tests used native functions.
- Hurst and fractal dimension are estimated with MATLAB self-implementations (documented briefly in the paper).
- Original experiments leveraged CPU parallelization; GPU not required.

## Documentation
- `docs/Procesos_Estocasticos_II_Proyecto.pdf` — full paper (Spanish), journal style. An English summary may be added in future updates.

## Contributing
Personal academic repo. Forks welcome; issues/PRs are optional.
