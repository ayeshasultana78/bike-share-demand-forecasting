import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Capstone Project - Notebook 2: Modelling, Evaluation, Error Analysis and Recommendations
## Forecasting Hourly Bike-Share Demand (Capital Bikeshare, 2011-2012)

This notebook covers capstone objectives 3 to 6: applying statistical and machine-learning methods, building and evaluating the solution, analysing errors and limitations, and deriving insights and recommendations.

**Evaluation design (decided before looking at any test result)**
- Strict chronological split: **train** Jan 2011 - Jun 2012, **validation** Jul - Sep 2012 (model selection and tuning), **test** Oct - Dec 2012 (touched once, at the end).
- Day-ahead forecast issued at midnight; every feature is known at forecast time.
- Metrics: MAE (average error in bikes/hour), RMSE, R-squared, **WAPE** (total absolute error / total demand) and bias (mean error).
- Every candidate must beat strong, simple baselines, not just a naive forecast.
""")

md("## 1. Setup and data")
code("""
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import features as ft, modeling as md

OUT, MODELS = ROOT / "outputs", ROOT / "models"; OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook"); pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:,.2f}".format)
BLUE, RED, GREEN, ORANGE, GREY = "#2f6db5", "#d1495b", "#3a9d5d", "#f2a541", "#8d99ae"
SEED = 42; R = {}

feat, parts = ft.prepare()
tr, va, te = parts["train"], parts["val"], parts["test"]
trva = pd.concat([tr, va])
R["split"] = {k: {"rows": int(len(v)), "start": str(v.dteday.min().date()), "end": str(v.dteday.max().date()), "mean_cnt": float(v.cnt.mean())} for k, v in parts.items()}
pd.DataFrame(R["split"]).T
""")
md("""
Average demand differs a lot between the periods (growth plus season), which makes this a demanding, realistic test: a model must handle a shift in demand level, not just memorise the past.
""")

md("## 2. Baselines: what must be beaten")
code("""
base_lag168 = lambda d: d["lag_168"].to_numpy()                      # same hour last week
base_lag24 = lambda d: d["lag_24"].to_numpy()                        # same hour yesterday
base_static = md.static_profile_baseline(tr)                          # typical hourly profile by day type
base_level = md.level_profile_baseline(tr)                            # recent level x typical profile (strongest simple method)
baselines = {"Same hour yesterday": base_lag24, "Typical hourly profile": base_static, "Same hour last week": base_lag168, "Recent level x hourly profile": base_level}
rows = [{"model": n, **md.score(va.cnt, f(va))} for n, f in baselines.items()]
val_base = pd.DataFrame(rows).set_index("model").sort_values("MAE")
val_base
""")
md("""
The **recent level x hourly profile** baseline is the strongest simple method: it knows the typical shape of the day for each day type and scales it by how busy the system has been over the last week.
Any machine-learning model has to beat *this*, not just last week's value.
""")

md("## 3. Candidate models (trained on the training period, scored on validation)")
md("""
| Model | Idea |
|---|---|
| Ridge regression | Linear model on log demand with one-hot hour x day type, weather and log lags |
| Random forest | Non-linear, bagged trees on calendar and weather features plus the trend |
| HistGradientBoosting (Poisson) | Gradient-boosted trees with a Poisson loss for counts |
| XGBoost (Poisson), absolute | As above, predicting demand directly and using a time-trend feature |
| XGBoost (Poisson), **with lag features** | Adds demand 24 h and 168 h earlier |
| **Level-adjusted XGBoost (Poisson)** | Predicts the *rate* demand / recent level, then multiplies by the level, so growth is handled by multiplication instead of tree extrapolation |
""")
code("""
def ridge_frame(d):
    x = d[ft.FS_CAL + ["lag_24", "lag_168", "roll_168"]].copy()
    x["hr_dt"] = (x["hr"] * 3 + x["day_type"]).astype(str)
    for c in ("lag_24", "lag_168", "roll_168"): x["log_" + c] = np.log1p(x[c])
    return x.drop(columns=["lag_24", "lag_168", "roll_168"])
ridge_cat = ["hr_dt", "hr", "weekday", "mnth", "weathersit", "day_type"]
ridge_num = ["temp_c", "atemp_c", "hum_pct", "wind_kmh", "t_days", "log_lag_24", "log_lag_168", "log_roll_168"]
class RidgeLog:
    def __init__(self, alpha=3.0):
        self.pipe = Pipeline([("prep", ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), ridge_cat), ("n", StandardScaler(), ridge_num)])), ("m", Ridge(alpha=alpha))])
    def fit(self, d, y): self.pipe.fit(ridge_frame(d), np.log1p(y)); return self
    def predict(self, d): return np.expm1(self.pipe.predict(ridge_frame(d)))

class Plain:                                  # wraps a regressor so it always sees the same feature columns
    def __init__(self, est, cols): self.est, self.cols = est, cols
    def fit(self, d, y): self.est.fit(d[self.cols], y); return self
    def predict(self, d): return self.est.predict(d[self.cols])

xgb_abs = lambda **kw: XGBRegressor(objective="count:poisson", tree_method="hist", n_jobs=-1, random_state=SEED, **{"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, **kw})
candidates = {
    "Ridge (log target, with lags)":      RidgeLog(),
    "Random forest (calendar+weather)":   Plain(RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=SEED), ft.FS_CAL),
    "HistGB Poisson (calendar+weather)":  Plain(HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, random_state=SEED), ft.FS_CAL),
    "XGBoost Poisson (calendar+weather)": Plain(xgb_abs(), ft.FS_CAL),
    "XGBoost Poisson + lag features":     Plain(xgb_abs(), ft.FS_LAG),
    "Level-adjusted XGBoost Poisson":     md.LevelAdjustedPoisson(random_state=SEED),
}
rows, val_pred = [], {}
for name, m in candidates.items():
    t0 = time.perf_counter(); m.fit(tr, tr.cnt); p = np.maximum(m.predict(va), 0); val_pred[name] = p
    rows.append({"model": name, **md.score(va.cnt, p), "fit_seconds": time.perf_counter() - t0})
val_cand = pd.DataFrame(rows).set_index("model").sort_values("MAE")
val_cand
""")
md("""
**What the validation results show**
- **Several candidates do not even beat the strongest simple baseline:** the Ridge regression and the HistGradientBoosting model are no better than "recent level x hourly profile". A more complex model is not automatically a better one.
- The linear model is weaker than the best tree-based models: interactions between hour, day type and weather matter.
- **Adding lag features did not help; it made the XGBoost model worse** (the raw lags carry weather and one-off noise from the day they come from). This overturned the initial assumption that lags would help, which is why the comparison was done instead of assuming.
- Every tree-based **absolute-demand** model **under-predicts** (negative bias) because demand kept growing beyond the training data and trees cannot extrapolate a trend.
- The **level-adjusted** model removes that bias and has the lowest error among the candidates, so it is carried forward to tuning.
""")

md("## 4. Hyperparameter tuning with time-series cross-validation")
md("""
Tuning uses an **expanding-window `TimeSeriesSplit` (4 folds) on the training period only**; the validation period is never used for tuning. 14 random configurations are tried for each of the two XGBoost variants,
scored by MAE in bikes/hour. This is a modest, fair budget for both.
""")
code("""
space = {"n_estimators": [200, 300, 500, 800], "learning_rate": [0.02, 0.03, 0.05, 0.08], "max_depth": [3, 4, 5, 6, 8],
         "min_child_weight": [1, 3, 5, 10], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.5, 1, 3, 10]}
tscv = TimeSeriesSplit(n_splits=4)
def cv_mae(make, data):
    errs = []
    for a, b in tscv.split(data):
        m = make().fit(data.iloc[a], data.cnt.iloc[a]); errs.append(np.abs(np.maximum(m.predict(data.iloc[b]), 0) - data.cnt.iloc[b]).mean())
    return float(np.mean(errs))
def tune(make_from_params, n_iter=14):
    out = []
    for prm in ParameterSampler(space, n_iter=n_iter, random_state=SEED):
        out.append({**prm, "cv_mae": cv_mae(lambda: make_from_params(prm), tr)})
    return pd.DataFrame(out).sort_values("cv_mae").reset_index(drop=True)
t0 = time.perf_counter()
tuned_level = tune(lambda p: md.LevelAdjustedPoisson(random_state=SEED, **p))
tuned_abs = tune(lambda p: Plain(xgb_abs(**p), ft.FS_CAL))
print(f"Tuning took {time.perf_counter() - t0:.0f} s")
print("Level-adjusted - best 3 configurations:"); display_cols = list(space) + ["cv_mae"]
tuned_level[display_cols].head(3)
""")
code("""
print("Absolute-demand XGBoost - best 3 configurations:")
tuned_abs[display_cols].head(3)
""")
code("""
best_level = {k: (int(v) if k in ("n_estimators", "max_depth", "min_child_weight") else float(v)) for k, v in tuned_level.iloc[0][list(space)].items()}
best_abs = {k: (int(v) if k in ("n_estimators", "max_depth", "min_child_weight") else float(v)) for k, v in tuned_abs.iloc[0][list(space)].items()}
R["best_params_level"], R["best_params_abs"] = best_level, best_abs
R["cv_mae_level"], R["cv_mae_abs"] = float(tuned_level.cv_mae.iloc[0]), float(tuned_abs.cv_mae.iloc[0])
R["default_cv_mae_level"] = cv_mae(lambda: md.LevelAdjustedPoisson(random_state=SEED), tr)
print(f"Level-adjusted: default CV MAE {R['default_cv_mae_level']:.1f} -> tuned {R['cv_mae_level']:.1f}")
print("Best level-adjusted parameters found by cross-validation:", best_level)
""")
md("""
**Tuning outcome (important).** Tuning lowers the cross-validated error on the training folds, but whether it helps on genuinely new data is decided in the next step on the validation period, not here.
""")

md("## 5. Model selection on the validation period")
code("""
tuned_models = {"Level-adjusted XGBoost (tuned)": md.LevelAdjustedPoisson(random_state=SEED, **best_level),
                "XGBoost Poisson, absolute (tuned)": Plain(xgb_abs(**best_abs), ft.FS_CAL)}
for name, m in tuned_models.items():
    m.fit(tr, tr.cnt); p = np.maximum(m.predict(va), 0); val_pred[name] = p
all_val = pd.concat([val_base.assign(kind="baseline"), val_cand.drop(columns="fit_seconds").assign(kind="model"),
                     pd.DataFrame([{"model": n, **md.score(va.cnt, val_pred[n]), "kind": "model"} for n in tuned_models]).set_index("model")]).sort_values("MAE")
R["validation_table"] = all_val.reset_index().to_dict(orient="records")
selected = all_val[all_val.kind == "model"].index[0]; R["selected_model"] = selected
SELECTED_PARAMS = {"Level-adjusted XGBoost Poisson": {}, "Level-adjusted XGBoost (tuned)": best_level}
assert selected in SELECTED_PARAMS, f"selected model {selected} is not a level-adjusted variant; update the final-model code"
final_params = SELECTED_PARAMS[selected]; R["final_params"] = final_params
R["tuned_val_mae"] = float(all_val.loc["Level-adjusted XGBoost (tuned)", "MAE"]); R["default_val_mae"] = float(all_val.loc["Level-adjusted XGBoost Poisson", "MAE"])
fig, ax = plt.subplots(figsize=(10, 5.2))
colors = [GREEN if n == selected else (GREY if k == "baseline" else BLUE) for n, k in zip(all_val.index, all_val.kind)]
ax.barh(all_val.index[::-1], all_val.MAE[::-1], color=colors[::-1])
for i, v in enumerate(all_val.MAE[::-1]): ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)
ax.set(title="Validation MAE (bikes per hour, lower is better): grey = baselines, green = selected", xlabel="Mean absolute error", xlim=(0, all_val.MAE.max() * 1.12))
fig.tight_layout(); fig.savefig(OUT / "fig08_validation_comparison.png", dpi=150); plt.show()
print("Selected on validation:", selected, "| parameters:", final_params or "library defaults (moderate trees, 300 rounds)")
print(f"Validation MAE: default level-adjusted {R['default_val_mae']:.1f} vs tuned {R['tuned_val_mae']:.1f}")
all_val[["MAE", "RMSE", "R2", "WAPE_%", "Bias", "kind"]]
""")
md("""
**Selection rule: the validation period decides.** If the tuned configuration is not better on validation than the default one, the simpler default configuration is kept, because the tuning gain on the training folds did not carry over to new data.
Choosing by validation (rather than by cross-validation on the same data used for tuning) guards against over-fitting the tuning itself.
""")

md("## 6. Final evaluation on the held-out test period")
md("""
The selected model is **re-fitted on training + validation** (so it has seen all data up to 30 September 2012) and evaluated once on October - December 2012. Baselines are re-fitted the same way.
""")
code("""
final = md.LevelAdjustedPoisson(random_state=SEED, **final_params).fit(trva, trva.cnt)
p_final = np.maximum(final.predict(te), 0)
base_fn = {"Same hour yesterday": base_lag24, "Typical hourly profile": md.static_profile_baseline(trva), "Same hour last week": base_lag168, "Recent level x hourly profile": md.level_profile_baseline(trva)}
test_rows = [{"model": n, "kind": "baseline", **md.score(te.cnt, f(te))} for n, f in base_fn.items()]
test_rows.append({"model": "Level-adjusted XGBoost (final)", "kind": "model", **md.score(te.cnt, p_final)})
test_tab = pd.DataFrame(test_rows).set_index("model").sort_values("MAE")
R["test_table"] = test_tab.reset_index().to_dict(orient="records")
best_base = test_tab[test_tab.kind == "baseline"].index[0]; bb = test_tab.loc[best_base]; fm = test_tab.loc["Level-adjusted XGBoost (final)"]
err_f, err_b = p_final - te.cnt.to_numpy(), base_fn[best_base](te) - te.cnt.to_numpy()
red = 1 - fm.MAE / bb.MAE; lo, hi = md.daily_block_bootstrap(te, err_f, err_b)
R["final"] = {"metrics": {k: float(fm[k]) for k in ["MAE", "RMSE", "R2", "WAPE_%", "Bias"]}, "best_baseline": best_base, "best_baseline_metrics": {k: float(bb[k]) for k in ["MAE", "RMSE", "R2", "WAPE_%", "Bias"]},
              "mae_reduction_vs_best_baseline_pct": float(red * 100), "mae_reduction_ci95_pct": [lo * 100, hi * 100],
              "mae_reduction_vs_last_week_pct": float((1 - fm.MAE / test_tab.loc["Same hour last week", "MAE"]) * 100),
              "criterion_30pct_met": bool(red >= 0.30)}
print(f"Final model - test MAE {fm.MAE:.1f} bikes/h, RMSE {fm.RMSE:.1f}, R2 {fm.R2:.3f}, WAPE {fm['WAPE_%']:.1f}%, bias {fm.Bias:+.1f}")
print(f"Best simple baseline: {best_base} (MAE {bb.MAE:.1f})")
print(f"MAE reduction vs best baseline: {red:.1%}  (95% CI by day-block bootstrap: {lo:.1%} to {hi:.1%})")
print(f"MAE reduction vs 'same hour last week': {R['final']['mae_reduction_vs_last_week_pct']:.1f}%")
print("Success criterion (>= 30% lower MAE than the best simple baseline):", "MET" if R["final"]["criterion_30pct_met"] else "NOT MET")
test_tab[["MAE", "RMSE", "R2", "WAPE_%", "Bias", "kind"]]
""")
md("""
**Honest reading of the result.** The criterion check above is computed by the code, not written by hand. The model beats the naive methods by a wide margin and beats the strongest simple baseline by a statistically reliable margin
(the bootstrap interval excludes zero), but the improvement over that strongest baseline is below the 30% target set at the start. The criterion is therefore reported as **not met**, and it is not re-defined after the fact.
The practical value of the model is discussed in the insights section.
""")

md("### 6.1 Ablation: which ingredients matter? (re-fitted on train + validation, same hyperparameters as the final model)")
code("""
def refit_eval(name, make, cols_note=""):
    m = make().fit(trva, trva.cnt); p = np.maximum(m.predict(te), 0); return {"variant": name, **md.score(te.cnt, p)}
weather_cols = ["weathersit", "temp_c", "atemp_c", "hum_pct", "wind_kmh"]
abl = [refit_eval("Final: level-adjusted, calendar + weather", lambda: md.LevelAdjustedPoisson(random_state=SEED, **final_params)),
       refit_eval("Without weather (calendar only)", lambda: md.LevelAdjustedPoisson(random_state=SEED, features=[c for c in md.FS_REL if c not in weather_cols], **final_params)),
       refit_eval("Plus lag features (24 h, 168 h)", lambda: md.LevelAdjustedPoisson(random_state=SEED, features=md.FS_REL + ["lag_24", "lag_168"], **final_params)),
       refit_eval("Absolute demand + trend feature (tuned)", lambda: Plain(xgb_abs(**best_abs), ft.FS_CAL))]
abl = pd.DataFrame(abl).set_index("variant"); R["ablation"] = abl.reset_index().to_dict(orient="records")
abl
""")
md("""
**Reading the ablation:** without weather inputs the model is barely better than the strongest simple baseline, so **weather is where most of the value over the baseline comes from** (this uses *observed* weather; the robustness test in section 9 checks what happens with imperfect forecasts).
The level-adjusted formulation has lower error than predicting absolute demand with a trend feature, and lag features still add no value.
""")

md("## 7. Quantifying uncertainty: calibrated prediction intervals")
md("""
Point forecasts are not enough for planning: an operator needs to know how much buffer to add. Count data is **overdispersed** (Notebook 1), so a simple Poisson interval (forecast +/- 1.28 x sqrt(forecast)) will be far too narrow.
Instead, a **normalised split-conformal interval** is built: out-of-sample errors from time-series cross-validation on the training + validation data are expressed relative to the forecast, and their 80th percentile sets the interval width.
""")
code("""
folds = list(TimeSeriesSplit(n_splits=6).split(trva))[1:]              # skip the first fold (too little history)
scores = []
for a, b in folds:
    m = md.LevelAdjustedPoisson(random_state=SEED, **final_params).fit(trva.iloc[a], trva.cnt.iloc[a])
    pb = np.maximum(m.predict(trva.iloc[b]), 0); scores.append(np.abs(trva.cnt.iloc[b].to_numpy() - pb) / (pb + 5.0))
scores = np.concatenate(scores); qhat = md.split_conformal_width(scores, 0.80)
lower, upper = np.maximum(p_final - qhat * (p_final + 5.0), 0), p_final + qhat * (p_final + 5.0)
pois_lo, pois_hi = np.maximum(p_final - 1.2816 * np.sqrt(p_final), 0), p_final + 1.2816 * np.sqrt(p_final)
y = te.cnt.to_numpy()
cov = float(np.mean((y >= lower) & (y <= upper))); cov_pois = float(np.mean((y >= pois_lo) & (y <= pois_hi)))
rush = (te.rush_hour == 1).to_numpy(); night = (te.hr <= 5).to_numpy()
R["intervals"] = {"nominal": 80, "qhat": qhat, "coverage_conformal_pct": cov * 100, "coverage_poisson_pct": cov_pois * 100,
                  "below_upper_pct": float(np.mean(y <= upper) * 100), "mean_width_all": float(np.mean(upper - lower)),
                  "mean_width_rush": float(np.mean((upper - lower)[rush])), "mean_width_night": float(np.mean((upper - lower)[night])),
                  "mean_upper_minus_forecast_rush": float(np.mean((upper - p_final)[rush])), "coverage_rush_pct": float(np.mean(((y >= lower) & (y <= upper))[rush]) * 100),
                  "coverage_night_pct": float(np.mean(((y >= lower) & (y <= upper))[night]) * 100), "calibration_folds": len(folds), "calibration_points": int(len(scores))}
I = R["intervals"]
print(f"Nominal 80% interval -> actual test coverage: conformal {I['coverage_conformal_pct']:.1f}% vs naive Poisson interval {I['coverage_poisson_pct']:.1f}%")
print(f"Share of hours at or below the upper bound (one-sided ~90%): {I['below_upper_pct']:.1f}%")
print(f"Average interval width: {I['mean_width_all']:.0f} bikes/h overall | {I['mean_width_rush']:.0f} at commuter peaks | {I['mean_width_night']:.0f} at night (00-05)")
print(f"Coverage at commuter peaks: {I['coverage_rush_pct']:.1f}% | at night: {I['coverage_night_pct']:.1f}%")
""")

md("## 8. Error analysis")
md("### 8.1 Forecast versus actual")
code("""
res = te[["ts", "dteday", "hr", "day_type", "weathersit", "rush_hour", "mnth", "cnt", "temp_c", "hum_pct", "wind_kmh"]].copy()
res["pred"], res["lower"], res["upper"] = p_final, lower, upper; res["err"] = res.pred - res.cnt; res["abs_err"] = res.err.abs()
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
for ax, (a, b, title) in zip(axes, (("2012-11-12", "2012-11-18", "A typical week (12-18 Nov 2012)"), ("2012-10-27", "2012-11-02", "Hurricane Sandy week (27 Oct - 2 Nov 2012): most hours of 29-30 Oct were never recorded (gaps)"))):
    s = res[(res.ts >= a) & (res.ts < pd.Timestamp(b) + pd.Timedelta(days=1))]
    s = s.set_index("ts").reindex(pd.date_range(a, pd.Timestamp(b) + pd.Timedelta(hours=23), freq="h")).rename_axis("ts").reset_index()   # unrecorded hours become gaps
    ax.fill_between(s.ts, s.lower, s.upper, color=BLUE, alpha=.18, label="80% prediction interval")
    ax.plot(s.ts, s.pred, color=BLUE, lw=1.8, label="Forecast"); ax.plot(s.ts, s.cnt, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
    ax.set(title=title, ylabel="Rentals per hour"); ax.legend(loc="upper right", ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig09_forecast_vs_actual.png", dpi=150); plt.show()
""")
md("### 8.2 Where is the model weak?")
code("""
def breakdown(g):
    return pd.Series({"hours": len(g), "mean_demand": g.cnt.mean(), "MAE": g.abs_err.mean(), "WAPE_%": g.abs_err.sum() / g.cnt.sum() * 100, "bias": g.err.mean()})
by_hr = res.groupby("hr").apply(breakdown)
res["weather"] = res.weathersit.map({1: "Clear", 2: "Mist/cloudy", 3: "Light rain/snow", 4: "Heavy rain/snow"})
by_wx = res.groupby("weather").apply(breakdown).loc[["Clear", "Mist/cloudy", "Light rain/snow"]]
by_dt = res.assign(day=res.day_type.map({0: "Working day", 1: "Weekend", 2: "Holiday"})).groupby("day").apply(breakdown)
by_mo = res.groupby("mnth").apply(breakdown)
R["errors"] = {"by_hour": by_hr.reset_index().to_dict(orient="records"), "by_weather": by_wx.reset_index().to_dict(orient="records"), "by_daytype": by_dt.reset_index().to_dict(orient="records"),
               "by_month": by_mo.reset_index().to_dict(orient="records")}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
axes[0].bar(by_hr.index, by_hr.MAE, color=BLUE); axes[0].set(title="Forecast error by hour of day", xlabel="Hour", ylabel="MAE (bikes/h)", xticks=range(0, 24, 2))
ax2 = axes[0].twinx(); ax2.plot(by_hr.index, by_hr.bias, color=RED, marker="o", ms=3); ax2.axhline(0, color=RED, lw=.6); ax2.set_ylabel("Bias (red)"); ax2.grid(False)
axes[1].bar(by_wx.index, by_wx["WAPE_%"], color=[GREEN, ORANGE, RED]); axes[1].set(title="Relative error (WAPE) by weather", ylabel="WAPE (%)", ylim=(0, by_wx["WAPE_%"].max() * 1.3))
for i, (w, n) in enumerate(zip(by_wx["WAPE_%"], by_wx.hours)): axes[1].text(i, w + .5, f"{w:.0f}%\\n(n={int(n)})", ha="center", fontsize=9)
daily_bias = res.groupby("dteday").err.mean().rolling(7, center=True, min_periods=3).mean()
axes[2].plot(daily_bias.index, daily_bias.values, color=BLUE); axes[2].axhline(0, color="black", lw=.8); axes[2].set(title="7-day rolling average forecast error (bias)", ylabel="Forecast minus actual (bikes/h)")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "fig10_error_breakdown.png", dpi=150); plt.show()
print(f"Peak-hour error: 17:00 MAE {by_hr.loc[17, 'MAE']:.0f}, 08:00 MAE {by_hr.loc[8, 'MAE']:.0f} vs night 03:00 MAE {by_hr.loc[3, 'MAE']:.0f}  (errors scale with demand)")
by_wx
""")
code("""
day_err = res.groupby("dteday").agg(actual=("cnt", "sum"), predicted=("pred", "sum"), hours=("cnt", "size"), abs_err=("abs_err", "sum"), wx=("weathersit", "max"))
day_err["error_pct"] = (day_err.predicted / day_err.actual - 1) * 100
worst_days = day_err.sort_values("abs_err", ascending=False).head(8); worst_days.index = worst_days.index.strftime("%Y-%m-%d (%a)")
R["worst_days"] = worst_days.reset_index().rename(columns={"dteday": "day"}).to_dict(orient="records")
sandy = day_err.loc["2012-10-29":"2012-10-30"]
R["sandy"] = {"observed_hours": int(sandy.hours.sum()), "actual": float(sandy.actual.sum()), "predicted": float(sandy.predicted.sum())}
mae_wo = res[~res.dteday.isin(sandy.index)].abs_err.mean()
R["test_mae_without_sandy"] = float(mae_wo); R["test_mae_with_sandy"] = float(res.abs_err.mean())
print(f"Test MAE: {res.abs_err.mean():.1f} with the two Hurricane Sandy days, {mae_wo:.1f} without them (those days barely matter: the shutdown hours were never recorded, so they are not scored)")
# Holiday-adjacent days: the public holiday itself, the day before and the day after
hol_days = pd.Series(feat.loc[feat.holiday == 1, "dteday"].unique())
adj = set(pd.concat([hol_days - pd.Timedelta(days=1), hol_days, hol_days + pd.Timedelta(days=1)]))
res["hol_adjacent"] = res.dteday.isin(adj)
ha = res.groupby("hol_adjacent").apply(lambda g: pd.Series({"hours": len(g), "MAE": g.abs_err.mean(), "WAPE_%": g.abs_err.sum() / g.cnt.sum() * 100, "bias": g.err.mean(), "share_of_total_abs_error_%": g.abs_err.sum() / res.abs_err.sum() * 100}))
R["holiday_adjacent"] = {"days": sorted(str(d.date()) for d in set(res.loc[res.hol_adjacent, "dteday"])), "table": ha.reset_index().to_dict(orient="records")}
top8_adj = int(sum(pd.Timestamp(d[:10]) in adj for d in worst_days.index)); R["worst8_holiday_adjacent"] = top8_adj
print(f"{top8_adj} of the 8 worst days are a public holiday or the day before/after one")
ha
""")
code("""
worst_days[["actual", "predicted", "hours", "abs_err", "error_pct", "wx"]]
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
axes[0].scatter(res.cnt, res.pred, s=6, alpha=.35, color=BLUE); lim = [0, max(res.cnt.max(), res.pred.max())]; axes[0].plot(lim, lim, color=RED, ls="--")
axes[0].set(title="Forecast vs actual (test hours)", xlabel="Actual rentals", ylabel="Forecast")
axes[1].hist(res.err.clip(-200, 200), bins=60, color=BLUE, edgecolor="white"); axes[1].axvline(0, color=RED); axes[1].set(title="Forecast error distribution", xlabel="Forecast minus actual (bikes/h, clipped at +/-200)", ylabel="Hours")
ac = acf(res.sort_values("ts").err.to_numpy(), nlags=48, fft=True); axes[2].bar(range(49), ac, color=BLUE); axes[2].axhline(0, color="black", lw=.8)
axes[2].set(title="Residual autocorrelation", xlabel="Lag (hours)", ylabel="ACF")
fig.tight_layout(); fig.savefig(OUT / "fig11_residual_diagnostics.png", dpi=150); plt.show()
R["residual_acf_lag1"], R["residual_acf_lag24"] = float(ac[1]), float(ac[24])
print(f"Residual autocorrelation: lag 1 = {ac[1]:.2f}, lag 24 = {ac[24]:.2f}")
print(f"Share of hours within +/-20% of actual (for hours with >= 50 rentals): {np.mean((res[res.cnt >= 50].abs_err / res[res.cnt >= 50].cnt) <= .2):.1%}")
R["within20_pct_ge50"] = float(np.mean((res[res.cnt >= 50].abs_err / res[res.cnt >= 50].cnt) <= .2) * 100)
""")
md("""
**Error patterns found**
1. **Errors scale with demand:** the largest absolute errors are at the commuter peaks (08:00 and 17:00-18:00), while in the quiet night hours the absolute error is tiny but the *relative* error is highest (WAPE 29-53% between 00:00 and 04:00, against 16-20% in the daytime), because the counts are small and noisy.
2. **Holiday periods are the biggest failure:** several of the eight worst days are public holidays or the day before/after one (for example the day after Thanksgiving and Christmas Eve), when the model forecasts a normal working day but demand collapses. The data flags only the holiday itself, not the surrounding days, and even Thanksgiving Day itself was over-forecast by about 90%, because only a handful of holidays appear in the training data.
3. **Bad weather is hard to forecast:** relative error is more than twice as high in rain/snow hours as in clear hours, and the sample is small.
4. **Extreme events are invisible to the score:** the hours in which Hurricane Sandy closed the system were never recorded, so they are not scored; a live system would have forecast normal demand on those days, which is why a severe-weather protocol is recommended.
5. **Residuals are autocorrelated:** errors persist from one hour to the next (and day to day), meaning part of the error is a slowly-changing level mistake that the model does not see.
""")

md("## 9. Interpretation and robustness")
md("### 9.1 What does the model rely on?")
code("""
groups = {"Hour of day and day type": ["hr", "hr_sin", "hr_cos", "weekday", "is_weekend", "workingday", "holiday", "day_type", "rush_hour"],
          "Season / month": ["mnth", "season"], "Weather": ["weathersit", "temp_c", "atemp_c", "hum_pct", "wind_kmh"], "Recent demand level": ["roll_168"]}
rng = np.random.default_rng(SEED); base_mae = np.abs(final.predict(te) - te.cnt).mean(); rows = []
for g, cols in groups.items():
    inc = []
    for _ in range(5):
        Xp = te.copy(); perm = rng.permutation(len(te)); Xp[cols] = te[cols].iloc[perm].to_numpy(); inc.append(np.abs(final.predict(Xp) - te.cnt).mean() - base_mae)
    rows.append({"group": g, "mae_increase": np.mean(inc), "std": np.std(inc)})
imp = pd.DataFrame(rows).sort_values("mae_increase"); R["importance"] = imp.sort_values("mae_increase", ascending=False).to_dict(orient="records")
def pdp(feature, grid):
    out = []
    for v in grid: Xp = te.copy(); Xp[feature] = v; out.append(np.maximum(final.predict(Xp), 0).mean())
    return np.array(out)
tgrid = np.linspace(0, 35, 36); hgrid = np.linspace(20, 100, 17)
R["pdp_temp"] = {"grid": tgrid.tolist(), "pred": pdp("temp_c", tgrid).tolist()}; R["pdp_hum"] = {"grid": hgrid.tolist(), "pred": pdp("hum_pct", hgrid).tolist()}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
axes[0].barh(imp.group, imp.mae_increase, xerr=imp["std"], color=BLUE); axes[0].set(title="Grouped permutation importance (test)", xlabel="Increase in MAE when the group is shuffled")
axes[1].plot(tgrid, R["pdp_temp"]["pred"], color=ORANGE, lw=2.5); axes[1].set(title="Partial dependence: temperature", xlabel="Temperature (C)", ylabel="Average forecast (bikes/h)")
axes[2].plot(hgrid, R["pdp_hum"]["pred"], color=GREEN, lw=2.5); axes[2].set(title="Partial dependence: humidity", xlabel="Relative humidity (%)", ylabel="Average forecast (bikes/h)")
fig.tight_layout(); fig.savefig(OUT / "fig12_interpretation.png", dpi=150); plt.show()
imp.sort_values("mae_increase", ascending=False).round(1)
""")
md("""
**Interpretation:** the time-of-day and day-type pattern is by far the most important input, followed by the recent demand level (which carries growth and season) and then weather. The partial-dependence curves show the learned relationships:
forecast demand rises with temperature and falls with humidity, in line with the Poisson regression in Notebook 1.
""")
md("### 9.2 Robustness: what if the weather forecast is wrong?")
md("""
All results so far use *observed* weather as if the forecast were perfect. In reality, a day-ahead weather forecast has errors. To estimate the effect, random errors typical of day-ahead forecasts are added to the test weather inputs
(temperature +/- 2 C, humidity +/- 8 points, wind +/- 5 km/h, and a 15% chance that the weather category is shifted by one step), averaged over 20 random draws.
""")
code("""
def perturb(d, r):
    x = d.copy(); dt = r.normal(0, 2.0, len(x)); x["temp_c"] += dt; x["atemp_c"] += dt
    x["hum_pct"] = np.clip(x["hum_pct"] + r.normal(0, 8, len(x)), 0, 100); x["wind_kmh"] = np.clip(x["wind_kmh"] + r.normal(0, 5, len(x)), 0, None)
    shift = r.random(len(x)) < 0.15; x.loc[shift, "weathersit"] = np.clip(x.loc[shift, "weathersit"] + r.choice([-1, 1], shift.sum()), 1, 4)
    return x
rr = np.random.default_rng(SEED); noisy = [md.score(te.cnt, np.maximum(final.predict(perturb(te, rr)), 0)) for _ in range(20)]
noisy_mae = float(np.mean([n["MAE"] for n in noisy])); noisy_wape = float(np.mean([n["WAPE_%"] for n in noisy]))
no_weather = float(abl.loc["Without weather (calendar only)", "MAE"])
R["robustness"] = {"clean_mae": float(fm.MAE), "noisy_mae": noisy_mae, "noisy_wape": noisy_wape, "no_weather_mae": no_weather, "best_baseline_mae": float(bb.MAE),
                   "mae_increase_pct": float((noisy_mae / fm.MAE - 1) * 100)}
print(f"MAE with perfect weather: {fm.MAE:.1f} | with realistic forecast errors: {noisy_mae:.1f} (+{R['robustness']['mae_increase_pct']:.0f}%) | calendar only (no weather): {no_weather:.1f} | best baseline: {bb.MAE:.1f}")
fig, ax = plt.subplots(figsize=(9.5, 4.4))
labels = ["Best simple baseline", "Calendar only\\n(no weather)", "Final model,\\nforecast-quality weather", "Final model,\\nperfect weather"]
vals = [bb.MAE, no_weather, noisy_mae, fm.MAE]; ax.bar(labels, vals, color=[GREY, ORANGE, BLUE, GREEN])
for i, v in enumerate(vals): ax.text(i, v + 1, f"{v:.1f}", ha="center")
ax.set(title="Test MAE under different information scenarios (bikes per hour)", ylabel="MAE", ylim=(0, max(vals) * 1.15))
fig.tight_layout(); fig.savefig(OUT / "fig13_robustness.png", dpi=150); plt.show()
""")

md("## 10. Insights and recommendations")
code("""
E1 = json.loads((OUT / "results_eda.json").read_text()); P, G, W = E1["profile"], E1["glm_effects"], E1["weather"]
F, I, RB = R["final"], R["intervals"], R["robustness"]
night_share = P["night_0_5_share_pct"]; rush_share = P["peak_hours_workday_share_pct"]
recs = pd.DataFrame([
 {"priority": 1, "timeline": "0-3 months", "recommendation": "Run the nightly day-ahead forecast and use it to plan rebalancing before the commuter peaks",
  "evidence": f"Working-day rush hours (07-09 and 16-19) hold {rush_share:.0f}% of working-day rentals; peaks of {P['workday_peak_am']:.0f}/h (08:00) and {P['workday_peak_pm']:.0f}/h (17:00). Forecast WAPE {F['metrics']['WAPE_%']:.1f}% vs {F['best_baseline_metrics']['WAPE_%']:.1f}% for the best simple method",
  "expected_benefit": f"{F['mae_reduction_vs_best_baseline_pct']:.0f}% lower hourly error than the best simple baseline ({F['mae_reduction_ci95_pct'][0]:.0f}%-{F['mae_reduction_ci95_pct'][1]:.0f}% 95% CI) and {F['mae_reduction_vs_last_week_pct']:.0f}% lower than 'same hour last week'", "kpi": "Forecast WAPE; empty-bike and full-dock incidents at peak"},
 {"priority": 2, "timeline": "0-3 months", "recommendation": "Plan fleet buffers from the calibrated 80% interval, not the point forecast",
  "evidence": f"Real coverage of the interval is {I['coverage_conformal_pct']:.0f}% (nominal 80%), whereas a naive Poisson interval covers only {I['coverage_poisson_pct']:.0f}%; the average upper bound sits {I['mean_upper_minus_forecast_rush']:.0f} bikes/h above the forecast at commuter peaks",
  "expected_benefit": f"About {I['below_upper_pct']:.0f}% of hours stay at or below the upper bound, giving a quantified safety margin", "kpi": "Interval coverage; share of peak hours above the upper bound"},
 {"priority": 3, "timeline": "0-3 months", "recommendation": "Make operations weather-aware: scale deployed fleet and crew plans with the forecast",
  "evidence": f"At the same hour and season, +10 C raises rentals {(G['Temperature, +10 C']['irr'] - 1) * 100:.0f}%, light rain/snow cuts them {-(G['Light rain/snow vs clear']['irr'] - 1) * 100:.0f}%, +10 km/h wind cuts them {-(G['Wind, +10 km/h']['irr'] - 1) * 100:.0f}%; weather information lowers forecast error from {RB['no_weather_mae']:.0f} to {F['metrics']['MAE']:.0f} bikes/h",
  "expected_benefit": "Fewer idle bikes on cold or wet days, more bikes on warm days", "kpi": "Utilisation per bike by weather class"},
 {"priority": 4, "timeline": "0-3 months", "recommendation": "Schedule maintenance and large rebalancing runs in the 00:00-05:59 window",
  "evidence": f"Only {night_share:.1f}% of all rentals occur between midnight and 06:00",
  "expected_benefit": "Maintenance with minimal lost rentals", "kpi": "Rentals lost during maintenance windows"},
 {"priority": 5, "timeline": "3-6 months", "recommendation": "Treat weekends as a different product: keep fleet size but shift bikes to leisure areas and promote casual use at midday",
  "evidence": f"Daily totals are similar on working and non-working days (about {P['avg_workday_total']:,.0f} vs {P['avg_weekend_total']:,.0f}), but weekends have a single midday peak ({P['weekend_peak']:.0f}/h at 13:00) and a {P['casual_share_weekend']:.0f}% casual share vs {P['casual_share_workday']:.0f}% on working days",
  "expected_benefit": "Better placement and conversion of casual riders", "kpi": "Weekend casual rentals; casual-to-member conversion"},
 {"priority": 6, "timeline": "3-6 months", "recommendation": "Add holiday-period and severe-weather rules: flag the days around public holidays, and do not rely on the model during extreme events",
  "evidence": f"{R['worst8_holiday_adjacent']} of the 8 worst-forecast days are within one day of a public holiday (e.g. the day after Thanksgiving, Christmas Eve), and WAPE on holiday-adjacent days is {ha.loc[True, 'WAPE_%']:.0f}% against {ha.loc[False, 'WAPE_%']:.0f}% on other days; rain/snow hours have WAPE {by_wx.loc['Light rain/snow', 'WAPE_%']:.0f}% vs {by_wx.loc['Clear', 'WAPE_%']:.0f}% in clear weather; Hurricane Sandy shut the system down (most hours of 29-30 Oct were never recorded)",
  "expected_benefit": "Avoids acting on forecasts at the times they are known to fail", "kpi": "Forecast error on flagged days; number of manual overrides"},
 {"priority": 7, "timeline": "3-6 months", "recommendation": "Retrain monthly and monitor forecast bias and WAPE to handle growth",
  "evidence": f"Demand grew {E1['growth_hourly_mean_pct']:.0f}% per hour from 2011 to 2012; the level-adjusted design copes with growth, but forecast errors are autocorrelated (lag-24 ACF {R['residual_acf_lag24']:.2f}), so level drift must be monitored",
  "expected_benefit": "Prevents silent model decay", "kpi": "Rolling 28-day WAPE and bias"},
 {"priority": 8, "timeline": "6-12 months", "recommendation": "Invest in better data: station-level trips, bike availability, real weather forecasts and an events calendar",
  "evidence": f"Forecast-quality weather instead of perfect weather raises MAE by about {RB['mae_increase_pct']:.0f}%; observed rentals understate true demand when bikes run out",
  "expected_benefit": "Station-level rebalancing and less hidden (unserved) demand", "kpi": "Share of demand served; station-level forecast error"},
])
recs.to_csv(OUT / "recommendations.csv", index=False); R["recommendations"] = recs.to_dict(orient="records")
recs[["priority", "timeline", "recommendation", "expected_benefit"]]
""")
md("""
## 11. Limitations and opportunities for improvement
**Limitations**
- **Perfect-weather assumption.** Evaluation uses observed weather; robustness testing shows the error rises with realistic forecast errors, so live accuracy will be lower.
- **Observed rentals are not true demand.** When bikes or docks run out, demand is censored; the data cannot reveal unserved demand.
- **System-wide only.** Station-level rebalancing needs station-level models and data.
- **Two years, one city.** One full seasonal cycle is in the training data; Q4 2012 is only compared with Q4 2011, and results may not transfer to other cities.
- **Holidays and extreme events** are rare, so the model cannot learn them well: the days around public holidays dominate the worst errors, and the hours of the Hurricane Sandy shutdown were never recorded, so the score does not reflect that failure mode.
- **Wind speed and humidity measurement quality** (zero values) is imperfect.
- **Success criterion.** The 30% improvement over the best simple baseline set at the start is evaluated honestly in Section 6 and was not re-defined after seeing the results.

**Opportunities for improvement**
1. Add real weather *forecasts* (and their errors) as training inputs rather than observed weather.
2. Add holiday-eve, holiday-after and bridge-day features, event calendars and a severe-weather flag from alerts (the biggest single opportunity given the error analysis).
3. Model the level error explicitly (e.g., a short-memory correction of the previous day's forecast error, since errors are autocorrelated).
4. Hierarchical or station-level forecasts; separate models for casual and registered riders, whose behaviour differs.
5. Compare with dedicated time-series methods (e.g., gradient boosting with several quantile levels, or neural sequence models) once more data is available.
""")

md("## 12. Saving the solution")
code("""
joblib.dump(final, MODELS / "bike_demand_model.joblib", compress=3)
reloaded = joblib.load(MODELS / "bike_demand_model.joblib")
assert np.allclose(reloaded.predict(te.head(50)), final.predict(te.head(50)))
json.dump({"qhat": qhat, "offset": 5.0, "level": 0.80}, open(MODELS / "interval_calibration.json", "w"))
res[["ts", "cnt", "pred", "lower", "upper", "weathersit", "day_type"]].to_csv(OUT / "test_predictions.csv", index=False)
test_tab.to_csv(OUT / "test_metrics.csv"); all_val.to_csv(OUT / "validation_metrics.csv")
print("Saved model, interval calibration, test predictions and metrics.")
# Inference demo: day-ahead forecast for Thursday 13 Dec 2012 (features exist at midnight; weather taken as observed here)
day = te[te.dteday == "2012-12-13"].copy(); day["forecast"] = np.maximum(reloaded.predict(day), 0)
day["lower"] = np.maximum(day.forecast - qhat * (day.forecast + 5), 0); day["upper"] = day.forecast + qhat * (day.forecast + 5)
day[["hr", "forecast", "lower", "upper", "cnt"]].round(0).astype(int).rename(columns={"cnt": "actual"}).set_index("hr").T
""")
code("""
(OUT / "results_model.json").write_text(json.dumps(R, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
print("Saved outputs/results_model.json")
""")
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/02_modeling_evaluation_insights.ipynb"); print("nb2 cells", len(C))
