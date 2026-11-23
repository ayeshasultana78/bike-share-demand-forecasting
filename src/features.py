"""Data loading, cleaning and feature engineering for the Capital Bikeshare demand-forecasting capstone."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "hour_raw.csv"
URL = "https://raw.githubusercontent.com/muditp19/UCI_Bike-sharing-dataset/master/hour.csv"  # mirror of the UCI file (CC BY 4.0)

TRAIN_END = pd.Timestamp("2012-06-30")      # train: 2011-01-01 .. 2012-06-30
VAL_END = pd.Timestamp("2012-09-30")        # validation: 2012-07-01 .. 2012-09-30; test: 2012-10-01 .. 2012-12-31
LAG_WARMUP_HOURS = 24 + 168                 # first rows have no complete lag history

FS_CAL = ["hr", "hr_sin", "hr_cos", "weekday", "is_weekend", "mnth", "season", "workingday", "holiday", "day_type",
          "rush_hour", "weathersit", "temp_c", "atemp_c", "hum_pct", "wind_kmh", "t_days"]
FS_LAG = FS_CAL + ["lag_24", "lag_168", "roll_168"]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read hour.csv; download it once from the public mirror if it is not cached locally."""
    path = path or RAW_PATH
    if not path.exists():
        import urllib.request
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return pd.read_csv(path, parse_dates=["dteday"])


def make_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete hourly calendar.

    The source silently omits hours with no rentals (mostly at night) and hours during extreme events
    (snowstorm, hurricanes). We add those rows back, flag them with ``observed = False`` and treat their
    rental count as 0 when building lag features. Rows that were not observed are never used as targets.
    """
    r = raw.copy()
    r["ts"] = r["dteday"] + pd.to_timedelta(r["hr"], unit="h")
    full = pd.date_range(r["ts"].min(), r["ts"].max().normalize() + pd.Timedelta(hours=23), freq="h")
    g = r.set_index("ts").reindex(full)
    g.index.name = "ts"
    g["observed"] = g["cnt"].notna()
    g["dteday"] = g.index.normalize()
    g["hr"] = g.index.hour
    day_cols = ["season", "yr", "mnth", "holiday", "weekday", "workingday"]      # constant within a day
    g[day_cols] = g.groupby("dteday")[day_cols].transform(lambda s: s.ffill().bfill())
    weather = ["weathersit", "temp", "atemp", "hum", "windspeed"]
    g[weather] = g[weather].ffill().bfill()                                       # only affects rows that are never scored
    g[day_cols + ["weathersit"]] = g[day_cols + ["weathersit"]].astype(int)
    g["cnt_filled"] = g["cnt"].fillna(0.0)
    return g.reset_index()


def add_features(g: pd.DataFrame) -> pd.DataFrame:
    """Engineer calendar, weather, trend and lag features.

    All lag features use information that is available 24 hours ahead (a day-ahead forecast issued at midnight),
    so they never contain the value being predicted.
    """
    f = g.copy()
    f["hr_sin"] = np.sin(2 * np.pi * f["hr"] / 24)
    f["hr_cos"] = np.cos(2 * np.pi * f["hr"] / 24)
    f["is_weekend"] = f["weekday"].isin([0, 6]).astype(int)
    f["day_type"] = np.select([f["holiday"] == 1, f["workingday"] == 1], [2, 0], default=1)   # 0 working, 1 weekend, 2 holiday
    f["rush_hour"] = ((f["workingday"] == 1) & f["hr"].isin([7, 8, 9, 16, 17, 18, 19])).astype(int)
    f["temp_c"] = f["temp"] * 41.0            # UCI normalisation of temperature; converted back to degrees Celsius
    f["atemp_c"] = f["atemp"] * 50.0          # feels-like temperature, converted back to degrees Celsius
    f["hum_pct"] = f["hum"] * 100.0
    f["wind_kmh"] = f["windspeed"] * 67.0
    f["t_days"] = (f["ts"] - f["ts"].min()).dt.total_seconds() / 86400.0
    c = f["cnt_filled"]
    f["lag_24"] = c.shift(24)
    f["lag_168"] = c.shift(168)
    f["roll_168"] = c.shift(24).rolling(168, min_periods=168).mean()      # mean demand over the 7 days ending 24 h ago
    f["usable"] = f["observed"] & (np.arange(len(f)) >= LAG_WARMUP_HOURS)
    return f


def split(f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Strict chronological split. Only observed rows with a complete lag history are used."""
    u = f[f["usable"]]
    d = u["dteday"]
    return {"train": u[d <= TRAIN_END], "val": u[(d > TRAIN_END) & (d <= VAL_END)], "test": u[d > VAL_END]}


def prepare() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    f = add_features(make_grid(load_raw()))
    return f, split(f)
