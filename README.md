# Capstone Project - Forecasting Hourly Bike-Share Demand (Capital Bikeshare, 2011-2012)

An end-to-end data science project: from problem definition and data preparation to statistical analysis, machine-learning forecasting, uncertainty quantification, error analysis and business recommendations.

**Problem.** Can hourly bike rental demand be forecast one day ahead accurately enough to support fleet rebalancing, maintenance and staffing, and what drives that demand?

**Data.** *Bike Sharing Dataset* (Fanaee-T and Gama, 2013), UCI Machine Learning Repository, DOI 10.24432/C5W894, CC BY 4.0: 17,379 hourly records of Capital Bikeshare (Washington D.C.) joined with weather and holiday data, 2011-2012. The CSV is included in `data/`; `src/features.py` can re-download it from a public mirror.

## Project structure
```
data/hour_raw.csv                         source data
notebooks/01_problem_data_eda.ipynb       problem definition, data quality, features, EDA, statistics
notebooks/02_modeling_evaluation_insights.ipynb   models, tuning, test evaluation, intervals, error analysis, recommendations
src/features.py                           cleaning, calendar restoration, feature engineering, chronological split
src/modeling.py                           metrics, baselines, level-adjusted Poisson model, bootstrap and conformal helpers
tests/                                    10 unit tests (incl. data-leakage tests)
models/bike_demand_model.joblib           final model;  interval_calibration.json
outputs/                                  13 figures, metrics tables, test predictions, recommendations, HTML notebook exports
docs/                                     project report (Word) and presentation (PowerPoint)
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/                                 # 10 tests, including leakage checks
jupyter lab                                             # run notebook 01, then notebook 02 (Run All)
```

## Method in brief
1. **Data preparation.** 165 hours of the two-year calendar are silently absent (113 at night = zero rentals, 52 in the daytime = extreme events such as Hurricane Sandy). The calendar is restored, unobserved hours are flagged and never used as targets.
2. **Features.** Calendar (hour, day type, rush hour), weather (real units), growth trend and demand-level features; all known at forecast time (day-ahead). A strict chronological split is used: train Jan 2011-Jun 2012, validation Jul-Sep 2012, test Oct-Dec 2012.
3. **Statistics.** Hypothesis tests and a Poisson regression quantify the effects of weather, holidays and growth.
4. **Models.** Four simple baselines, Ridge, random forest, HistGradientBoosting, XGBoost with and without lag features, and a **level-adjusted Poisson XGBoost** (it learns the rate relative to recent demand, so growth is handled by multiplication instead of tree extrapolation). Tuning with time-series cross-validation; selection on the validation period.
5. **Uncertainty.** A calibrated 80% prediction interval (normalised split-conformal).
6. **Error analysis, interpretation and robustness** (including imperfect weather forecasts).

## Key results (test period Oct-Dec 2012, never used for selection)
| | Final model | Best simple baseline |
|---|---|---|
| MAE (bikes/hour) | **42.8** | 53.3 |
| WAPE (total absolute error / total demand) | **19.5%** | 24.3% |
| R-squared | 0.889 | 0.810 |

- **19.8% lower MAE than the best simple baseline** (95% CI 13-26%) and 41% lower than "same hour last week". **The initial 30% target was not met** and was not re-defined afterwards.
- The nominal 80% interval covered **75%** of test hours (a naive Poisson interval: 33%).
- With realistic weather-forecast errors the MAE rises only 4% (42.8 to 44.5).
- Demand grew 63% per hour from 2011 to 2012. Working days have commuter peaks at 8:00 and 17:00; weekends a single midday peak. At the same hour, +10 C raises rentals about 28% and light rain/snow cuts them about 40%.
- Honest findings: lag features did not help; tuning improved cross-validation but not validation, so the default configuration was kept; the worst errors fall on public-holiday periods.

## Limitations
Perfect-weather assumption in evaluation (tested for robustness), observed rentals understate true demand, system-wide rather than station-level, two years in one city, rare holidays and extreme events are hard to learn. See the report for opportunities for improvement.
