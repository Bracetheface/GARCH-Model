Volatility Modeling with a GARCH-like Model in Python

Project Objective
This project explores the core principles of financial volatility modeling, a critical component of risk management and derivatives pricing. The goal was to build a model from scratch that captures **volatility clustering**—the tendency for high-volatility periods and low-volatility periods to be grouped together in financial time series.

Due to the constraints of a browser-based environment (JupyterLite), this project uses a simplified but powerful GARCH-like model, the Exponentially Weighted Moving Average (EWMA), to demonstrate these core econometric concepts.

Methodology
1.  **Synthetic Data Generation**: A GARCH(1,1) process was used to generate a realistic financial time series exhibiting volatility clustering. This provided a controlled environment with a known "true" underlying volatility to test the model against.
2.  **Volatility Modeling (EWMA)**: An Exponentially Weighted Moving Average (EWMA) model was implemented to estimate the conditional volatility of the synthetic returns. The EWMA formula is:
    $$
    \sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2
    $$
    This model assumes that today's variance is a weighted average of yesterday's variance and yesterday's squared return (the volatility shock).
3.  **Model Evaluation**: The model's estimated volatility was plotted against the known true volatility of the generated data to visually assess its accuracy and tracking performance.
4.  **Forecasting**: A multi-step forecast was generated to demonstrate how the model projects future volatility, showing its expected reversion to a long-term average.

Key Visuals

Model Performance vs. True Volatility
This plot shows how well the simple EWMA model (blue line) tracks the true, underlying volatility (grey dashed line) of the asset.


Volatility Forecast
This plot shows the model's 30-day forecast, illustrating the expected decay of volatility over time.
