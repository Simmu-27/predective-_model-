 WeatherWise — Weather Prediction System

## Abstract
WeatherWise predicts a day's mean temperature (°C) from recent observations using Linear Regression (baseline) and Random Forest (comparison). It covers cleaning, EDA, chronological evaluation (MAE, RMSE, R²) and a Flask + Chart.js dashboard.

## Problem Statement
Given temperature history up to the day before, estimate the mean temperature of the next day.

## Objectives
Preprocess real weather data · explore trends and correlations · train and compare regression models · evaluate on unseen later dates · serve predictions through a web app.

## Dataset (you must supply it — none is bundled)
Put a real daily/hourly historical CSV at `data/weather.csv`. It needs a **date column** and a **temperature column in °C**. Sources:
- Kaggle: search "daily climate time series", "historical weather <your city>".
- NOAA: Climate Data Online (ncei.noaa.gov/cdo-web) or GHCN-Daily. Convert °F/tenths of °C to °C first.
Aim for 3+ years of data. Record in your report: source URL, location, period, units, missing-value handling. Hourly data is averaged to daily automatically.

## Requirements
Software: Windows 10/11, Python 3.10+, VS Code, a browser. Hardware: any laptop, 4 GB RAM.

## Setup (Windows, PowerShell)
```
cd weatherwise
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py --csv data/weather.csv --date-col date --target-col temperature
python app.py
```
Open http://127.0.0.1:5000. Change `--date-col/--target-col` to your CSV's column names. Chart.js loads from a CDN, so internet is needed for charts.

## Algorithms (simple language)
- **Linear Regression:** fits a straight-line formula: tomorrow ≈ a·yesterday + b·(2 days ago) + … It is fast and easy to explain.
- **Random Forest:** many decision trees vote on the answer. It captures non-linear patterns but can't extrapolate beyond temperatures seen in training.
- **Persistence baseline:** "tomorrow = today". A model is only useful if it beats this.

## Architecture
Browser (HTML/CSS/JS + Chart.js) → fetch JSON → Flask (`app.py`) → loads `models/weather_model.joblib` and `models/analytics.json` produced by `train_model.py` from `data/weather.csv`.

## Implementation steps
1. Load CSV, print shape/head/dtypes. 2. Parse dates, coerce numbers, drop missing/duplicate/impossible values (outside −90..60 °C). 3. Aggregate to daily, fill gaps of ≤3 days. 4. Features: temp 1/2/7 days earlier, previous-7-day mean, month, day of year (all known before the target day, so no leakage). 5. Chronological 80/20 split. 6. Train both models. 7. Evaluate, save with Joblib, export analytics.

## Evaluation methodology
Test set = the **latest** 20% of days, never seen in training. Metrics: MAE (average error, °C), RMSE (penalizes big errors), R² (variance explained). The persistence baseline is reported alongside. The best model is chosen by RMSE; read the actual numbers in the dashboard or terminal. Results are for one-day-ahead prediction at one location only.

## Using the prediction form
Enter the date you want to predict and the real observed temperatures for the days before it (1 day, 2 days, 7 days earlier, and the 7-day average). The app never asks for future humidity/pressure/wind.

## Testing
Verified with Flask's test client: `/api/health`, `/api/analytics`, `/api/model-metrics`, valid `/api/predict`, bad date, out-of-range value, missing fields (HTTP 400 + JSON error), and missing model (HTTP 503, no prediction returned). Pipeline was smoke-tested with temporary synthetic data, which is **not included**; your reported scores must come from your real dataset. Manually check that all three charts render and the Reset button clears the form.

## Limitations and future scope
One location, one-day-ahead, no pressure/humidity/cloud inputs, not an operational forecast. Future: multi-day forecasting, SARIMA/LSTM, extra weather variables, automatic fetch of recent observations from a weather API, hyperparameter tuning with `TimeSeriesSplit`.

## Project report outline
1 Title, certificate, acknowledgement · 2 Abstract · 3 Introduction & problem statement · 4 Literature survey · 5 Objectives & scope · 6 Dataset description · 7 System design (architecture, data-flow, API list) · 8 Methodology (cleaning, features, split, models) · 9 Implementation (screenshots, code snippets) · 10 Results (metrics table, charts, model comparison) · 11 Testing · 12 Limitations & future scope · 13 Conclusion · 14 References · Appendix (code).

## Viva questions
1. **What is predictive analytics?** Using historical data and models to estimate future or unknown values.
2. **Regression vs classification?** Regression predicts a number (temperature); classification predicts a category.
3. **Why Linear Regression first?** It is simple, fast, interpretable, and a good baseline.
4. **How does Random Forest work?** Many decision trees are trained on random samples; their predictions are averaged.
5. **Why a chronological split, not random?** Random mixing lets the model "see the future", giving unrealistically good scores.
6. **What is data leakage?** Using information at training that wouldn't exist at prediction time. We avoid it by using only past-day features.
7. **What are lag features?** Past values of the target (e.g. yesterday's temperature) used as inputs.
8. **Why month and day of year?** Temperature is seasonal.
9. **Explain MAE.** Average absolute difference between actual and predicted, in °C.
10. **Explain RMSE.** Square root of the mean squared error; punishes large errors more.
11. **Explain R².** Fraction of variance in temperature the model explains; 1 is perfect, 0 is no better than the mean.
12. **What is the persistence baseline and why use it?** Predicting "same as yesterday"; shows whether the model adds real value.
13. **How did you handle missing values?** Dropped invalid rows, interpolated gaps of ≤3 days only.
14. **Why Joblib?** Saves the trained model so Flask loads it without retraining.
15. **What does Flask do here?** Serves the page and JSON APIs, validates input, runs the model.
16. **Why not ask users for tomorrow's humidity?** It's unknown at prediction time; it would be leakage or guesswork.
17. **Can you claim this forecasts weather accurately?** Only as far as the time-based test shows: one day ahead, one location.
18. **What is overfitting?** Model memorizes training data and does poorly on new data; the test set reveals it.
19. **How to improve the model?** More features, more data, tuning, time-series models (SARIMA/LSTM).
20. **Which model won?** Answer from your own dashboard numbers, citing RMSE and the baseline.
