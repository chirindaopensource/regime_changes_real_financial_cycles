# `README.md`


# Regime Changes and Real-Financial Cycles: Searching Minsky's Hypothesis in a Nonlinear Setting

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.04348-b31b1b.svg)](https://arxiv.org/abs/2511.04348)
[![Journal](https://img.shields.io/badge/Journal-arXiv%20Preprint-003366)](https://arxiv.org/abs/2511.04348)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/regime_changes_real_financial_cycles)
[![Discipline](https://img.shields.io/badge/Discipline-Macroeconomics%20%7C%20Financial%20Economics-00529B)](https://github.com/chirindaopensource/regime_changes_real_financial_cycles)
[![Data Sources](https://img.shields.io/badge/Data-OECD%20Statistics-lightgrey)](https://stats.oecd.org/)
[![Data Sources](https://img.shields.io/badge/Data-BIS%20Credit%20Statistics-lightgrey)](https://www.bis.org/statistics/totcredit.htm)
[![Data Sources](https://img.shields.io/badge/Data-FRED%20Economic%20Data-lightgrey)](https://fred.stlouisfed.org/)
[![Core Method](https://img.shields.io/badge/Method-MS--VAR%20%7C%20EM%20Algorithm%20%7C%20HP%20Filter-orange)](https://github.com/chirindaopensource/regime_changes_real_financial_cycles)
[![Analysis](https://img.shields.io/badge/Analysis-Nonlinear%20Dynamics%20%7C%20Minsky%20Cycles-red)](https://github.com/chirindaopensource/regime_changes_real_financial_cycles)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%2311557c.svg?style=flat&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-blue.svg)](https://www.statsmodels.org/)
[![PyYAML](https://img.shields.io/badge/PyYAML-gray?logo=yaml&logoColor=white)](https://pyyaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/regime_changes_real_financial_cycles`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Regime Changes and Real-Financial Cycles: Searching Minsky's Hypothesis in a Nonlinear Setting"** by:

*   Domenico Delli Gatti
*   Filippo Gusella
*   Giorgio Ricchiuti

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and spectral decomposition to the core nonlinear Markov-Switching Vector Autoregression (MS-VAR) estimation, Minsky regime classification, and Monte Carlo robustness analysis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_minsky_research_project`](#key-callable-execute_minsky_research_project)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Delli Gatti, Gusella, and Ricchiuti (2025). The core of this repository is the iPython Notebook `regime_changes_real_financial_cycles_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed to be a generalizable toolkit for detecting endogenous financial fragility and nonlinear regime shifts in macroeconomic data.

The paper investigates Minsky's Financial Instability Hypothesis (FIH) by extending linear models to a nonlinear setting. It captures local real-financial endogenous cycles where stability breeds instability. This codebase operationalizes the paper's framework, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Process raw macroeconomic time series, applying Hodrick-Prescott (HP) filtering to extract cyclical components.
-   Estimate a bivariate **Markov-Switching Vector Autoregression (MS-VAR)** model to distinguish between "Minsky" (interaction) and "No Minsky" (independence) regimes.
-   Empirically verify the mathematical conditions for endogenous oscillations (complex eigenvalues) and Minskyan feedback loops.
-   Trace the evolution of financial fragility over time using filtered and smoothed regime probabilities.
-   Validate the robustness of findings via parametric bootstrap Monte Carlo simulations.
-   Automatically generate all key tables and figures from the paper.

## Theoretical Background

The implemented methods are grounded in nonlinear dynamic macroeconomic modeling and time series econometrics.

**1. The Minsky Cycle Mechanism:**
The core hypothesis is that financial fragility builds endogenously during expansions. This is captured by a dynamic interaction between a real variable ($y_t$, e.g., GDP) and a financial variable ($f_t$, e.g., debt).
$$
\begin{bmatrix} y_t \\ f_t \end{bmatrix} = \mathbf{A}(s_t) \begin{bmatrix} y_{t-1} \\ f_{t-1} \end{bmatrix} + \boldsymbol{\epsilon}_t
$$
A "Minsky Regime" ($s_t=1$) is characterized by a specific sign pattern in the interaction matrix $\mathbf{A}_1$:
-   $\beta_1 > 0$: Economic expansion leads to rising debt/interest rates (pro-cyclical leverage).
-   $\alpha_2 < 0$: Rising financial fragility drags down real activity.
-   **Oscillation Condition:** The discriminant $\Delta = (\alpha_1 - \beta_2)^2 + 4\alpha_2\beta_1 < 0$, implying complex eigenvalues and endogenous cycles.

**2. Regime Switching:**
The economy transitions between the interaction regime and an independence regime ($s_t=2$) where real and financial variables decouple ($\mathbf{A}_2$ is diagonal). This switching is governed by a first-order Markov chain with transition probabilities $p_{ij}$.

**3. Estimation Strategy:**
The model is estimated using the **Expectation-Maximization (EM) Algorithm**, which iteratively maximizes the likelihood function. The **Hamilton Filter** is used for the Expectation step to infer the latent state probabilities, and the **Kim Smoother** provides the optimal inference using the full sample.

Below is a diagram which sums up the Inputs-Processes-Outputs of the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/regime_changes_real_financial_cycles/blob/main/regime_changes_real_financial_cycles_inputs_processes_outputs.png" alt="Minsky MS-VAR Process Flow" width="100%">
</div>


## Features

The provided iPython Notebook (`regime_changes_real_financial_cycles_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 24 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks the schema, temporal consistency, and domain constraints of all input data.
-   **Advanced Signal Extraction:** Implements a sparse-matrix version of the Hodrick-Prescott filter for efficient cycle extraction.
-   **Robust Econometric Engine:** A custom EM algorithm implementation with numerical safeguards (regularization, log-sum-exp) for stable MS-VAR estimation.
-   **Comprehensive Diagnostics:** Includes Ljung-Box residual tests and rigorous eigenvalue analysis for Minsky condition verification.
-   **Monte Carlo Robustness:** A fully integrated parametric bootstrap module to assess the stability of parameter estimates and regime classifications.
-   **Automated Reporting:** Generates publication-ready LaTeX tables and technical documentation.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Preprocessing (Tasks 1-4):** Ingests raw data, validates schemas, cleanses missing values via interpolation/truncation, and applies logarithmic transformations.
2.  **Signal Extraction (Task 5):** Applies the HP filter ($\lambda=100$) to extract stationary cyclical components from annual data.
3.  **Model Setup (Tasks 6-10):** Constructs bivariate systems, verifies stationarity via ADF tests, and initializes EM parameters using baseline VAR estimates.
4.  **Estimation (Tasks 11-15):** Executes the EM algorithm, utilizing the Hamilton Filter and Kim Smoother to estimate regime-dependent parameters and state probabilities.
5.  **Diagnostics & Analysis (Tasks 16-18):** Validates residuals, checks mathematical conditions for Minsky cycles (eigenvalues, signs), and analyzes regime dominance over time.
6.  **Robustness & Synthesis (Tasks 19-24):** Runs Monte Carlo simulations, cross-validates against paper benchmarks, and compiles the final report.

## Core Components (Notebook Structure)

The `regime_changes_real_financial_cycles_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 24 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `execute_minsky_research_project`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_minsky_research_project`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `pyyaml`, `scipy`, `statsmodels`, `matplotlib`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/regime_changes_real_financial_cycles.git
    cd regime_changes_real_financial_cycles
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy pyyaml scipy statsmodels matplotlib
    ```

## Input Data Structure

The pipeline requires a dictionary of DataFrames (`raw_datasets`) where keys are country names and values are DataFrames with a `DatetimeIndex` (annual frequency) and the following columns:
1.  **`real_gdp`**: Real Gross Domestic Product.
2.  **`nfcd`**: Non-financial Corporate Debt.
3.  **`household_debt`**: Household Debt.
4.  **`stir`**: Short-Term Interest Rate.

All other parameters are controlled by the `config.yaml` file.

## Usage

The `regime_changes_real_financial_cycles_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `execute_minsky_research_project` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    with open('config.yaml', 'r') as f:
        study_config = yaml.safe_load(f)
    
    # 2. Load or generate raw datasets (Example using synthetic generator)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    raw_datasets = load_country_data()
    
    # 3. Execute the entire replication study.
    final_results = execute_minsky_research_project(
        study_config=study_config,
        raw_datasets=raw_datasets
    )
    
    # 4. Access results
    print(final_results["master_results"]["analysis"]["minsky_conditions"])
```

## Output Structure

The pipeline returns a comprehensive dictionary containing all analytical artifacts, structured as follows:
-   **`master_results`**: The aggregated dictionary of all outputs.
    -   **`estimation`**: Contains estimated parameters ($A, \Sigma, P$) for all models.
    -   **`diagnostics`**: Ljung-Box test results.
    -   **`analysis`**: Minsky condition verification and regime probabilities.
    -   **`robustness`**: Monte Carlo simulation statistics.
    -   **`validation`**: Cross-validation report against paper benchmarks.
-   **`latex_tables`**: Ready-to-compile LaTeX code for parameter estimates and classification tables.
-   **`technical_appendix`**: A markdown summary of implementation choices.

## Project Structure

```
regime_changes_real_financial_cycles/
│
├── regime_changes_real_financial_cycles_draft.ipynb  # Main implementation notebook
├── config.yaml                                       # Master configuration file
├── requirements.txt                                  # Python package dependencies
│
├── data/                                             # Directory for raw data (optional)
│   ├── usa_macro_data.csv
│   └── ...
│
├── LICENSE                                           # MIT Project License File
└── README.md                                         # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as the HP filter lambda, convergence tolerances, Monte Carlo iterations, and statistical thresholds without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Filtering:** Implementing the Hamilton regression filter or Baxter-King filter as alternatives to HP.
-   **Model Selection:** Adding information criteria (AIC/BIC) to select the optimal number of regimes or lags.
-   **Time-Varying Transition Probabilities:** Extending the MS-VAR to allow transition probabilities to depend on exogenous variables (TVTP-MS-VAR).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{delligatti2025regime,
  title={Regime Changes and Real-Financial Cycles: Searching Minsky's Hypothesis in a Nonlinear Setting},
  author={Delli Gatti, Domenico and Gusella, Filippo and Ricchiuti, Giorgio},
  journal={arXiv preprint arXiv:2511.04348},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). Regime Changes and Real-Financial Cycles: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/regime_changes_real_financial_cycles
```

## Acknowledgments

-   Credit to **Domenico Delli Gatti, Filippo Gusella, and Giorgio Ricchiuti** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, Statsmodels, and Matplotlib**.

--

*This README was generated based on the structure and content of the `regime_changes_real_financial_cycles_draft.ipynb` notebook and follows best practices for research software documentation.*
```
