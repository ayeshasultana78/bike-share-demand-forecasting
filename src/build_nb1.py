import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Capstone Project - Notebook 1: Problem Definition, Data Preparation and Exploratory Analysis
## Forecasting Hourly Bike-Share Demand (Capital Bikeshare, Washington D.C., 2011-2012)

This notebook covers capstone objectives 1 and 2: defining the problem, collecting and preparing the data, feature engineering, exploratory data analysis and statistical analysis.
Notebook 2 builds and evaluates the forecasting models and turns the results into insights and recommendations.
""")

md("""
## 1. Problem definition

**Context.** Bike-sharing systems are a core part of urban mobility. Their operators must decide, every day, how many bikes to place where and when, when to schedule maintenance, and how to staff rebalancing crews.
If supply is too low, riders are turned away; if it is too high, bikes sit idle and cost money. Demand is strongly affected by time of day, weekday, season and weather, and it is growing.

**Problem statement.** *Can hourly bike rental demand be forecast accurately enough (one day ahead) to support fleet-rebalancing, maintenance and staffing decisions, and which factors drive that demand?*

**Objectives**
1. Understand and quantify the main drivers of hourly demand (time, calendar, weather, growth).
2. Build a day-ahead hourly demand forecast that clearly beats simple benchmark methods.
3. Quantify forecast uncertainty so that planners can add a sensible safety buffer.
4. Identify where and why the model fails, and what that means for real-world use.
5. Turn the findings into practical, data-driven recommendations.

**Scope.** System-wide hourly rentals (not individual stations), 1 January 2011 to 31 December 2012. Forecast horizon: the next 24 hours, issued at midnight.
Out of scope: station-level demand, bike availability, trip routes and prices (not in the data).

**Requirements.** Strictly chronological evaluation (no information from the future); features must be available at forecast time; results must be reproducible and documented.

**Expected outcomes and success criteria**
| Criterion | Target |
|---|---|
| Accuracy | At least 30% lower mean absolute error than the best simple baseline on a held-out future period |
| Uncertainty | A prediction interval whose real coverage is close to its nominal level (80%) |
| Insight | Clear, quantified demand drivers and at least five actionable recommendations |
| Reproducibility | One command re-creates all results; leakage tests pass |
""")

md("## 2. Setup")
code("""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import features as ft

OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:,.2f}".format)
BLUE, RED, GREEN, ORANGE, GREY = "#2f6db5", "#d1495b", "#3a9d5d", "#f2a541", "#8d99ae"
R = {}
""")

md("""
## 3. Data collection
**Source:** *Bike Sharing Dataset*, Fanaee-T, H. and Gama, J. (2013), UCI Machine Learning Repository, DOI 10.24432/C5W894, licence CC BY 4.0.
The data combines the public Capital Bikeshare trip logs (Washington D.C.) with hourly weather and holiday information.
The file `hour.csv` was obtained from a public GitHub mirror of the UCI file, and its integrity is verified below against the published description (17,379 hourly records, two years).
""")
code("""
raw = ft.load_raw()
print("Rows x columns:", raw.shape, "| period:", raw.dteday.min().date(), "to", raw.dteday.max().date())
# Integrity checks against the published dataset description
checks = {"17,379 hourly records": len(raw) == 17379, "casual + registered = cnt for every row": bool((raw.casual + raw.registered == raw.cnt).all()),
          "record index unique": bool(raw.instant.is_unique), "no exact duplicates": int(raw.duplicated().sum()) == 0, "no null values": int(raw.isna().sum().sum()) == 0}
R["integrity"] = {k: bool(v) for k, v in checks.items()}
R["total_rentals"] = int(raw.cnt.sum()); R["mean_hourly"] = float(raw.cnt.mean())
pd.Series(checks, name="passed").to_frame()
""")
code("""
raw.head()
""")
md("""
**Data dictionary (key fields).** `cnt` = total rentals in the hour (target) = `casual` + `registered`; `hr` hour 0-23; `weekday`, `workingday`, `holiday`, `season`, `mnth`, `yr` calendar fields;
`weathersit` 1 clear, 2 mist/cloudy, 3 light rain/snow, 4 heavy rain/snow; `temp`, `atemp`, `hum`, `windspeed` normalised weather values (converted back to degrees, percent and km/h below).
""")

md("## 4. Data understanding and quality audit")
code("""
raw.describe().T[["count", "mean", "std", "min", "50%", "max"]]
""")
md("### 4.1 Hidden missing data: the calendar has gaps")
md("""
The file has no null values, but a complete two-year hourly calendar has 17,544 hours and the file has only 17,379. The missing 165 hours are **absent rows**, not blank cells. Understanding them matters, because they are not random.
""")
code("""
grid = ft.make_grid(raw)
missing = grid[~grid.observed].copy()
night = missing[(missing.hr >= 23) | (missing.hr <= 5)]
day = missing[(missing.hr >= 6) & (missing.hr <= 22)]
R["calendar_hours"] = int(len(grid)); R["missing_hours"] = int(len(missing)); R["missing_night"] = int(len(night)); R["missing_daytime"] = int(len(day))
print(f"Complete calendar: {len(grid):,} hours | present: {int(grid.observed.sum()):,} | absent: {len(missing)}")
print(f"  absent at night (23:00-05:59): {len(night)}  -> hours with zero rentals are simply not recorded (3-4 am averages only {raw[raw.hr.isin([3, 4])].cnt.mean():.0f} rentals)")
print(f"  absent in daytime (06:00-22:59): {len(day)}")
by_day = day.groupby("dteday").size().sort_values(ascending=False)
events = by_day[by_day >= 3]
R["event_days"] = {str(k.date()): int(v) for k, v in events.items()}
events.to_frame("daytime hours missing")
""")
md("""
**Finding:** all daytime gaps cluster on a handful of days: 18 and 26-27 January 2011 (winter weather; the data records heavy rain/snow/ice on 26 January), 27 August 2011 (the arrival of Hurricane Irene) and **29-30 October 2012 (Hurricane Sandy)**.
On those days the system largely shut down, so the gaps mean *no rentals*, not randomly lost data. Treatment: the missing hours are restored to a complete calendar, flagged `observed = False`,
counted as zero rentals when computing lag features, and **never used as prediction targets**.
""")
md("### 4.2 Other quality checks")
code("""
q = {}
q["humidity = 0 (impossible)"] = int((raw.hum == 0).sum()); q["humidity = 0 on days"] = sorted({str(d.date()) for d in raw.loc[raw.hum == 0, "dteday"]})
q["wind speed = 0"] = int((raw.windspeed == 0).sum()); q["wind speed = 0 (%)"] = round((raw.windspeed == 0).mean() * 100, 1)
q["workingday inconsistent with weekday/holiday"] = int(((raw.workingday == 1) & (raw.weekday.isin([0, 6]) | (raw.holiday == 1))).sum())
q["max hourly rentals"] = int(raw.cnt.max()); q["rows above Tukey upper fence"] = int((raw.cnt > raw.cnt.quantile(.75) + 1.5 * (raw.cnt.quantile(.75) - raw.cnt.quantile(.25))).sum())
R["quality"] = q
pd.Series(q, name="result").to_frame()
""")
md("""
**Decisions**
| Finding | Decision |
|---|---|
| Humidity = 0 on a single day (sensor fault) | Kept; affects a handful of rows, tree-based models are robust to it |
| Wind speed = 0 for about 12% of hours | Kept; the measurement is coarse and zero most likely means "calm or not recorded"; documented as a limitation |
| High rental counts flagged as outliers | **Kept.** They are genuine commuter peaks, exactly what the model must predict |
| Calendar logic (workingday vs weekday/holiday) | Consistent, no fixes needed |
""")

md("## 5. Preprocessing and feature engineering")
md("""
**Principles:** (1) every feature must be known at forecast time; (2) lag features use data that is at least 24 hours old (a day-ahead forecast issued at midnight); (3) the target hour's own value never enters its own features (verified by unit tests in `tests/test_features.py`).

| Feature | Description | Why |
|---|---|---|
| `hr_sin`, `hr_cos` | Cyclical encoding of hour | Hour 23 and hour 0 are neighbours |
| `is_weekend`, `day_type`, `rush_hour` | Day type (working / weekend / holiday) and commuter-peak flag | Working days and weekends have very different profiles |
| `temp_c`, `atemp_c`, `hum_pct`, `wind_kmh` | Weather converted to real units | Interpretability |
| `weathersit` | Weather category | Rain and snow suppress riding |
| `t_days` | Days since 1 Jan 2011 | Captures system growth |
| `lag_24`, `lag_168` | Rentals 1 day and 1 week earlier (same hour) | Recent level and weekly pattern |
| `roll_168` | Average hourly rentals over the 7 days ending 24 h earlier | Current demand level (growth, season) |
""")
code("""
feat = ft.add_features(grid)
parts = ft.split(feat)
R["split"] = {k: {"rows": int(len(v)), "start": str(v.dteday.min().date()), "end": str(v.dteday.max().date()), "mean_cnt": float(v.cnt.mean())} for k, v in parts.items()}
print("Rows excluded: first", ft.LAG_WARMUP_HOURS, "hours (no lag history) and the", int((~feat.observed).sum()), "unobserved hours")
pd.DataFrame(R["split"]).T
""")
code("""
feat.loc[feat.usable, ["ts", "cnt", "hr", "day_type", "rush_hour", "temp_c", "hum_pct", "weathersit", "lag_24", "lag_168", "roll_168"]].head(5)
""")
md("""
**Chronological split.** Training: Jan 2011 - Jun 2012. Validation (model selection and tuning): Jul - Sep 2012. **Test (touched once at the end): Oct - Dec 2012**, which also contains Hurricane Sandy, a realistic stress test.
Random splitting would leak future information into training and overstate accuracy, so it is not used.
""")
code("""
feat.to_parquet(ROOT / "data" / "features.parquet", index=False)
print("Saved data/features.parquet", feat.shape)
""")

md("## 6. Exploratory data analysis")
md("### 6.1 The target: hourly rentals")
code("""
y = raw.cnt
R["target"] = {"mean": float(y.mean()), "median": float(y.median()), "std": float(y.std()), "max": int(y.max()), "skew": float(stats.skew(y)),
               "share_hours_under_50": float((y < 50).mean() * 100)}
# Overdispersion: variance-to-mean ratio inside homogeneous cells (same hour, day type, year-month)
cells = feat[feat.observed].groupby(["hr", "day_type", "yr", "mnth"]).cnt.agg(["mean", "var", "size"])
cells = cells[(cells["size"] >= 5) & (cells["mean"] > 0)]
R["median_var_mean_ratio"] = float((cells["var"] / cells["mean"]).median())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].hist(y, bins=60, color=BLUE, edgecolor="white"); axes[0].axvline(y.mean(), color=RED, ls="--", label=f"mean {y.mean():.0f}"); axes[0].axvline(y.median(), color=GREEN, ls="--", label=f"median {y.median():.0f}")
axes[0].set(title="Hourly rentals are right-skewed", xlabel="Rentals per hour", ylabel="Hours"); axes[0].legend()
axes[1].scatter(cells["mean"], cells["var"], s=8, alpha=.4, color=BLUE); lim = [1, cells["var"].max()]
axes[1].plot(lim, lim, color=RED, label="variance = mean (Poisson)"); axes[1].set(xscale="log", yscale="log", title="Variance far exceeds the mean (overdispersion)", xlabel="Mean rentals (per hour/day-type/month cell)", ylabel="Variance"); axes[1].legend()
fig.tight_layout(); fig.savefig(OUT / "fig01_target_distribution.png", dpi=150); plt.show()
print(f"Mean {y.mean():.0f}, median {y.median():.0f}, max {y.max()}; {R['target']['share_hours_under_50']:.0f}% of hours have fewer than 50 rentals")
print(f"Median variance-to-mean ratio inside comparable cells: {R['median_var_mean_ratio']:.1f} (a Poisson count would give 1)")
""")
md("**Implication:** demand is a skewed count with variance well above its mean, so errors should be judged with scale-aware metrics (MAE, RMSE, weighted error) and the models should respect non-negativity (Poisson-type loss is a natural choice).")

md("### 6.2 Growth, trend and seasonality")
code("""
daily = feat[feat.observed].groupby("dteday").cnt.sum().asfreq("D").fillna(0)
stl = STL(daily, period=7, robust=True).fit()
yearly = feat[feat.observed].groupby("yr").cnt.agg(["sum", "mean"]); growth = yearly.loc[1, "sum"] / yearly.loc[0, "sum"] - 1
growth_h = yearly.loc[1, "mean"] / yearly.loc[0, "mean"] - 1
mon = feat[feat.observed].groupby(["yr", "mnth"]).cnt.mean().unstack(0)
R["growth_2012_vs_2011_pct"] = float(growth * 100); R["growth_hourly_mean_pct"] = float(growth_h * 100); R["yearly_mean_hourly"] = {"2011": float(yearly.loc[0, "mean"]), "2012": float(yearly.loc[1, "mean"])}
R["monthly_mean"] = {int(m): {"2011": float(mon.loc[m, 0]), "2012": float(mon.loc[m, 1])} for m in mon.index}
R["peak_month_2012"] = int(mon[1].idxmax()); R["low_month_2012"] = int(mon[1].idxmin())
seasonal_amp = stl.seasonal.groupby(stl.seasonal.index.dayofweek).mean()
R["weekday_seasonal_effect"] = {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]: float(v) for i, v in seasonal_amp.items()}
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
axes[0, 0].plot(daily.index, daily.values, color=GREY, lw=.8, label="daily rentals"); axes[0, 0].plot(stl.trend.index, stl.trend.values, color=RED, lw=2.2, label="STL trend")
axes[0, 0].set(title="Daily rentals and trend (strong growth, annual cycle)", ylabel="Rentals per day"); axes[0, 0].legend()
axes[0, 1].plot(mon.index, mon[0], marker="o", color=GREY, label="2011"); axes[0, 1].plot(mon.index, mon[1], marker="o", color=BLUE, label="2012")
axes[0, 1].set(title=f"Average hourly rentals by month (2012 is {growth_h:.0%} higher per hour)", xlabel="Month", ylabel="Rentals per hour", xticks=range(1, 13)); axes[0, 1].legend()
axes[1, 0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], seasonal_amp.values, color=[BLUE] * 5 + [ORANGE] * 2); axes[1, 0].axhline(0, color="black", lw=.8)
axes[1, 0].set(title="Weekly pattern (STL seasonal component)", ylabel="Rentals vs typical day")
axes[1, 1].plot(stl.resid.index, stl.resid.values, color=BLUE, lw=.7); axes[1, 1].axhline(0, color="black", lw=.8)
worst = stl.resid.abs().sort_values(ascending=False).head(3).index
for d in worst: axes[1, 1].annotate(str(d.date()), (d, stl.resid[d]), fontsize=8, color=RED)
axes[1, 1].set(title="Residual after trend and weekly pattern: weather and events", ylabel="Rentals per day")
fig.tight_layout(); fig.savefig(OUT / "fig02_growth_seasonality.png", dpi=150); plt.show()
print(f"2012 vs 2011: total rentals +{growth:.1%} | average rentals per recorded hour {yearly.loc[0, 'mean']:.0f} -> {yearly.loc[1, 'mean']:.0f} (+{growth_h:.1%})")
print("Largest residual days:", [str(d.date()) for d in worst])
""")

md("### 6.3 Time of day: two different systems")
code("""
obs = feat[feat.observed].copy()
obs["day_label"] = obs.day_type.map({0: "Working day", 1: "Weekend", 2: "Holiday"})
prof = obs.groupby(["day_label", "hr"]).agg(cnt=("cnt", "mean"), casual=("casual", "mean"), registered=("registered", "mean")).reset_index()
wd = prof[prof.day_label == "Working day"].set_index("hr"); we = prof[prof.day_label == "Weekend"].set_index("hr")
sh = obs.groupby("day_label")[["casual", "registered", "cnt"]].sum(); sh["casual_share_pct"] = sh.casual / sh.cnt * 100
R["profile"] = {"workday_peak_hr_am": int(wd.cnt.loc[6:10].idxmax()), "workday_peak_am": float(wd.cnt.loc[6:10].max()), "workday_peak_hr_pm": int(wd.cnt.loc[15:20].idxmax()),
                "workday_peak_pm": float(wd.cnt.loc[15:20].max()), "weekend_peak_hr": int(we.cnt.idxmax()), "weekend_peak": float(we.cnt.max()),
                "casual_share_workday": float(sh.loc["Working day", "casual_share_pct"]), "casual_share_weekend": float(sh.loc["Weekend", "casual_share_pct"]),
                "night_0_5_share_pct": float(obs[obs.hr <= 5].cnt.sum() / obs.cnt.sum() * 100),
                "top3_hours_share_pct": float(obs.groupby("hr").cnt.sum().nlargest(3).sum() / obs.cnt.sum() * 100),
                "peak_hours_workday_share_pct": float(obs[(obs.day_type == 0) & obs.hr.isin([7, 8, 9, 16, 17, 18, 19])].cnt.sum() / obs[obs.day_type == 0].cnt.sum() * 100),
                "avg_workday_total": float(obs[obs.day_type == 0].groupby("dteday").cnt.sum().mean()), "avg_weekend_total": float(obs[obs.day_type == 1].groupby("dteday").cnt.sum().mean())}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for lab, col in (("Working day", BLUE), ("Weekend", ORANGE), ("Holiday", GREEN)):
    s = prof[prof.day_label == lab]; axes[0].plot(s.hr, s.cnt, marker="o", ms=4, lw=2.2, color=col, label=lab)
axes[0].annotate(f"{wd.cnt.loc[6:10].idxmax()}:00 peak\\n{wd.cnt.loc[6:10].max():.0f}", (wd.cnt.loc[6:10].idxmax(), wd.cnt.loc[6:10].max()), xytext=(2, 430), arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[0].annotate(f"{wd.cnt.loc[15:20].idxmax()}:00 peak\\n{wd.cnt.loc[15:20].max():.0f}", (wd.cnt.loc[15:20].idxmax(), wd.cnt.loc[15:20].max()), xytext=(19.5, 520), arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[0].set(title="Average rentals by hour: commuter peaks vs leisure plateau", xlabel="Hour of day", ylabel="Average rentals per hour", xticks=range(0, 24, 2)); axes[0].legend()
s = prof[prof.day_label == "Working day"]; axes[1].stackplot(s.hr, s.registered, s.casual, colors=[BLUE, ORANGE], labels=["Registered (commuters)", "Casual"])
axes[1].set(title="Working days: registered riders drive the peaks", xlabel="Hour of day", ylabel="Average rentals per hour", xticks=range(0, 24, 2)); axes[1].legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig03_hourly_profile.png", dpi=150); plt.show()
sh[["casual", "registered", "cnt", "casual_share_pct"]]
""")
code("""
P = R["profile"]
print(f"Working days: peaks at {P['workday_peak_hr_am']}:00 ({P['workday_peak_am']:.0f}/h) and {P['workday_peak_hr_pm']}:00 ({P['workday_peak_pm']:.0f}/h); casual share {P['casual_share_workday']:.0f}%")
print(f"Weekends: single midday plateau peaking at {P['weekend_peak_hr']}:00 ({P['weekend_peak']:.0f}/h); casual share {P['casual_share_weekend']:.0f}%")
print(f"Rush-hour windows hold {P['peak_hours_workday_share_pct']:.0f}% of working-day rentals; the 00:00-05:59 window holds only {P['night_0_5_share_pct']:.1f}% of all rentals")
""")
code("""
dow_names = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
hm = obs.pivot_table(index="weekday", columns="hr", values="cnt", aggfunc="mean").rename(index=dow_names).loc[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
fig, ax = plt.subplots(figsize=(13, 3.8))
sns.heatmap(hm, cmap="YlOrRd", annot=False, cbar_kws={"label": "Average rentals per hour"}, ax=ax)
ax.set(title="Demand heatmap: weekday x hour (two commuter peaks Mon-Fri, one leisure peak at weekends)", xlabel="Hour of day", ylabel="")
fig.tight_layout(); fig.savefig(OUT / "fig04_heatmap_weekday_hour.png", dpi=150); plt.show()
""")

md("### 6.4 Weather")
code("""
w = obs.copy()
w["temp_band"] = pd.cut(w.temp_c, [-10, 5, 10, 15, 20, 25, 30, 45], labels=["<5", "5-10", "10-15", "15-20", "20-25", "25-30", ">30"])
w["hum_band"] = pd.cut(w.hum_pct, [-1, 30, 45, 60, 75, 90, 101], labels=["<30", "30-45", "45-60", "60-75", "75-90", ">90"])
w["wind_band"] = pd.cut(w.wind_kmh, [-1, 0.1, 10, 20, 30, 100], labels=["calm/0", "0-10", "10-20", "20-30", ">30"])
lab = {1: "Clear / few clouds", 2: "Mist / cloudy", 3: "Light rain / snow", 4: "Heavy rain / snow"}
wt = w.groupby("weathersit").cnt.agg(["mean", "count"])
tb = w.groupby("temp_band", observed=True).cnt.agg(["mean", "count"])
R["weather"] = {"by_weathersit": {lab[k]: {"mean": float(v["mean"]), "n": int(v["count"])} for k, v in wt.iterrows()},
                "rain_vs_clear_pct": float((wt.loc[3, "mean"] / wt.loc[1, "mean"] - 1) * 100), "mist_vs_clear_pct": float((wt.loc[2, "mean"] / wt.loc[1, "mean"] - 1) * 100),
                "by_temp_band": {str(k): {"mean": float(v["mean"]), "n": int(v["count"])} for k, v in tb.iterrows()}}
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
axes[0, 0].bar([lab[i].replace(" / ", "/\\n") for i in wt.index[:3]], wt["mean"].iloc[:3], color=[GREEN, ORANGE, RED])
for i, (m, n) in enumerate(zip(wt["mean"].iloc[:3], wt["count"].iloc[:3])): axes[0, 0].text(i, m + 3, f"{m:.0f}\\n(n={n:,})", ha="center", fontsize=9)
axes[0, 0].set(title="Rentals by weather category (heavy rain: only 3 hours, not shown)", ylabel="Average rentals per hour", ylim=(0, 250))
for ax, col, title, xl in ((axes[0, 1], "temp_band", "Warmer is busier (up to 30+ C)", "Temperature (C)"), (axes[1, 0], "hum_band", "Humid air suppresses riding", "Relative humidity (%)"), (axes[1, 1], "wind_band", "Wind has a weaker effect", "Wind speed (km/h)")):
    g = w.groupby(col, observed=True).cnt.agg(["mean", "sem"]); ax.errorbar(range(len(g)), g["mean"], yerr=1.96 * g["sem"], marker="o", color=BLUE, capsize=4, lw=2)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index.astype(str)); ax.set(title=title, xlabel=xl, ylabel="Average rentals per hour")
fig.tight_layout(); fig.savefig(OUT / "fig05_weather_effects.png", dpi=150); plt.show()
print(f"Light rain/snow hours: {R['weather']['rain_vs_clear_pct']:.0f}% vs clear hours | mist/cloudy: {R['weather']['mist_vs_clear_pct']:.0f}%")
tb.T
""")
md("""
**Caution:** these are raw averages. Weather and time of day are entangled (it is colder at night, when few people ride), so the *net* weather effect needs a model that holds hour, day type, season and year constant. Section 7 provides that.
""")
code("""
num = obs[["temp_c", "hum_pct", "wind_kmh", "hr", "workingday", "t_days", "cnt", "casual", "registered"]]
corr = num.corr(method="spearman")
R["spearman_with_cnt"] = corr["cnt"].drop("cnt").round(3).to_dict()
fig, ax = plt.subplots(figsize=(7.5, 5.8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax); ax.set_title("Spearman correlation matrix")
fig.tight_layout(); fig.savefig(OUT / "fig06_correlation.png", dpi=150); plt.show()
""")

md("## 7. Statistical analysis")
md("### 7.1 Hypothesis tests")
code("""
tests = []
dd = obs.groupby("dteday").agg(total=("cnt", "sum"), yr=("yr", "first"), workingday=("workingday", "first"))
a, b = dd[dd.yr == 1].total, dd[dd.yr == 0].total
t, p = stats.ttest_ind(a, b, equal_var=False); tests.append(("Daily rentals 2012 vs 2011 (Welch t-test)", f"{a.mean():,.0f} vs {b.mean():,.0f}", p))
a, b = dd[dd.workingday == 1].total, dd[dd.workingday == 0].total
u, p = stats.mannwhitneyu(a, b); tests.append(("Daily rentals working vs non-working day (Mann-Whitney)", f"{a.mean():,.0f} vs {b.mean():,.0f}", p))
g = [obs.loc[obs.weathersit == k, "cnt"] for k in (1, 2, 3)]
h, p = stats.kruskal(*g); tests.append(("Hourly rentals across weather categories 1-3 (Kruskal-Wallis)", f"H = {h:,.0f}", p))
rho, p = stats.spearmanr(obs.temp_c, obs.cnt); tests.append(("Temperature vs rentals, all hours (Spearman)", f"rho = {rho:.2f}", p))
s17 = obs[(obs.hr == 17) & (obs.workingday == 1)]; rho17, p = stats.spearmanr(s17.temp_c, s17.cnt); tests.append(("Temperature vs rentals, working-day 17:00 only (Spearman)", f"rho = {rho17:.2f}", p))
res = pd.DataFrame(tests, columns=["test", "result", "p_value"]); res["p_value"] = res["p_value"].map(lambda v: f"{v:.2e}")
R["tests"] = res.to_dict(orient="records")
res
""")
md("""
**Interpretation of the tests**
- Growth is real and large: average daily rentals rose from about 3,400 to 5,600 (p < 0.001).
- Weather category matters (p < 0.001), and temperature is positively associated with demand, more strongly once the hour is held fixed (rho 0.60 at the 17:00 commute peak vs 0.42 overall).
- **Working and non-working days have similar *daily totals*** (about 4,600 vs 4,300, p = 0.12), so the weekday/weekend difference is in the **shape of the day** (commuter peaks vs a midday plateau), not in total volume. This is why hour-by-day-type interactions are essential in any model.
""")
md("### 7.2 What drives demand once everything else is held constant? (Poisson regression)")
md("""
A log-linear **Poisson regression with robust standard errors** (a standard model for counts, with quasi-Poisson-style correction for overdispersion) estimates each factor's effect while controlling for hour-by-day-type pattern, month and year.
Coefficients are shown as **incidence rate ratios** (IRR): 1.10 means +10% rentals.
""")
code("""
d = obs.copy(); d["wx"] = d.weathersit.clip(upper=3).astype(str)
glm = smf.glm("cnt ~ C(hr) * C(workingday) + C(mnth) + yr + holiday + C(wx) + temp_c + hum_pct + wind_kmh", data=d, family=sm.families.Poisson()).fit(cov_type="HC1")
def irr(name, scale=1):
    b, ci = glm.params[name] * scale, glm.conf_int().loc[name] * scale
    return {"irr": float(np.exp(b)), "low": float(np.exp(ci[0])), "high": float(np.exp(ci[1])), "p": float(glm.pvalues[name])}
eff = {"Temperature, +10 C": irr("temp_c", 10), "Humidity, +10 points": irr("hum_pct", 10), "Wind, +10 km/h": irr("wind_kmh", 10),
       "Mist/cloudy vs clear": irr("C(wx)[T.2]"), "Light rain/snow vs clear": irr("C(wx)[T.3]"), "Public holiday": irr("holiday"), "Year 2012 vs 2011": irr("yr")}
R["glm_effects"] = eff
tab = pd.DataFrame(eff).T.assign(effect_pct=lambda t: (t.irr - 1) * 100)[["irr", "low", "high", "effect_pct", "p"]]
tab["p"] = tab["p"].map(lambda v: f"{v:.1e}")
print(f"Pseudo R-squared (deviance based): {1 - glm.deviance / glm.null_deviance:.3f}")
R["glm_pseudo_r2"] = float(1 - glm.deviance / glm.null_deviance)
tab
""")
code("""
fig, ax = plt.subplots(figsize=(8.5, 4.2))
names = list(eff)[::-1]; vals = [(eff[n]["irr"] - 1) * 100 for n in names]
lo = [(eff[n]["low"] - 1) * 100 for n in names]; hi = [(eff[n]["high"] - 1) * 100 for n in names]
ax.barh(names, vals, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], color=[BLUE if v > 0 else RED for v in vals], capsize=4)
ax.axvline(0, color="black", lw=.8); ax.set(title="Effect on hourly rentals, other factors held constant (95% CI)", xlabel="Change in rentals (%)")
fig.tight_layout(); fig.savefig(OUT / "fig07_glm_effects.png", dpi=150); plt.show()
""")
md("""
**Interpretation:** at the same hour, day type, month and year, each +10 C raises rentals by about 28%, light rain/snow cuts them by about 40%, and humidity, wind and public holidays reduce them; the year effect (+60%) captures the system's growth.
Note how the controlled effect of mist (about -5%) is much smaller than its raw average (-14%): misty hours occur disproportionately at night and early morning, when few people ride anyway, so the raw difference was mostly a time-of-day effect.
These are *associations* estimated from observational data, but the signs and magnitudes are consistent with the raw patterns and with common sense, which gives confidence in the features used in the forecasting model.
""")
code("""
(OUT / "results_eda.json").write_text(json.dumps(R, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
print("Saved outputs/results_eda.json")
""")

md("""
## 8. Summary of Notebook 1
- **Data:** 17,379 hourly records (2011-2012) verified against the published description; 165 hours are silently absent (113 at night = zero rentals, 52 in daytime = extreme events such as Hurricane Sandy).
- **Preparation:** complete hourly calendar restored, unobserved hours flagged and excluded as targets, engineered calendar, weather, trend and lag features, strict chronological split, leakage tests.
- **Patterns:** strong growth (+65% total rentals, +63% per hour, in 2012), a pronounced annual cycle, two commuter peaks on working days vs one midday plateau at weekends (similar daily totals, very different shapes), and clear weather effects.
- **Statistics:** growth and weather effects are highly significant; daily totals do not differ between working and non-working days; a Poisson regression quantifies the weather, holiday and growth effects after controlling for time.
- **Next:** Notebook 2 builds baselines and machine-learning forecasts, quantifies uncertainty, analyses errors and derives recommendations.
""")
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/01_problem_data_eda.ipynb"); print("nb1 cells", len(C))
