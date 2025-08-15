# numerical-option-pricing-duke-energy

Methods and code to study **numerical option pricing** using **Duke Energy Corporation** daily closing prices.

> This repository accompanies an academic project. It focuses on methodology and reproducible workflows rather than business KPIs.

## 📦 Project Structure

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

Mostrar siempre los detalles

## 🎯 Goals
- Implement and compare numerical methods for option pricing:
  - **Binomial/Trinomial trees**
  - **Black–Scholes (closed-form for European options)**
  - **Monte Carlo simulation**
  - *(optional)* **Finite-difference schemes** for PDEs
- Use Duke Energy (DUK) closing prices as underlying data.
- Provide a clear, reproducible pipeline in MATLAB Live Script (`.mlx`).

## 📑 Data
- `data/raw/datos.xlsx` — input dataset (closing prices). Replace or augment with your own series as needed.
- If you later fetch data via APIs, keep raw dumps in `data/raw/` and document the source in this section.

## 🛠️ Requirements
- **MATLAB R2021b+** (Live Scripts `.mlx`)
  - Toolboxes typically used: *Statistics and Machine Learning Toolbox*, *Financial Toolbox* (if available).
- Optional: Python/R for auxiliary analysis (not required).

## ▶️ How to Run (MATLAB)
1. Open `code/matlab/Proyecto_EstocasticosII_AreizaParraSerna.mlx` in MATLAB.
2. Adjust the path to the dataset if necessary (it expects `data/raw/datos.xlsx`).
3. Run the Live Script sections sequentially (from top to bottom).

## 📚 Methodology (high-level)
- **Preprocessing:** clean and align closing prices; compute log-returns if needed.
- **Volatility:** estimate historical or GARCH-style volatility (optional).
- **Pricing:** apply chosen numerical methods to European call/put options.
- **Validation:** sanity checks (no-arbitrage bounds, convergence when refining steps, MC variance reduction).

## 📄 Documentation
- `docs/Procesos_Estocasticos_II_Proyecto.pdf` contains the original project document (Spanish).

## 🤝 Contributing (personal use)
This is a personal academic repo. Feel free to fork. Issues and PRs are welcome but not expected.

## 📜 License
MIT — see `LICENSE`.

## 🙌 Acknowledgements
- Universidad EAFIT — coursework and guidance.
- Classic references in option pricing and stochastic calculus.
