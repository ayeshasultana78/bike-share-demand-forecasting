import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import features as ft  # noqa: E402


def _toy(n_days=20, drop=((3, 2), (5, 14))):
    """Hourly toy data (day index, hour) with a couple of hours removed, like the real source file."""
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2011-01-01") + pd.Timedelta(days=d)
        for h in range(24):
            if (d, h) in drop:
                continue
            rows.append({"instant": len(rows) + 1, "dteday": day, "season": 1, "yr": 0, "mnth": 1, "hr": h, "holiday": 0,
                         "weekday": day.dayofweek, "workingday": int(day.dayofweek < 5), "weathersit": 1, "temp": 0.3,
                         "atemp": 0.3, "hum": 0.5, "windspeed": 0.1, "casual": 1, "registered": 9, "cnt": 10 + d + h})
    return pd.DataFrame(rows)


def test_grid_restores_missing_hours():
    g = ft.make_grid(_toy())
    assert len(g) == 20 * 24
    assert (~g["observed"]).sum() == 2
    assert g["ts"].is_monotonic_increasing and g["ts"].is_unique
    assert g.loc[~g["observed"], "cnt_filled"].eq(0).all()


def test_lag_features_use_past_only():
    f = ft.add_features(ft.make_grid(_toy()))
    i = 300
    assert f.loc[i, "lag_24"] == f.loc[i - 24, "cnt_filled"]
    assert f.loc[i, "lag_168"] == f.loc[i - 168, "cnt_filled"]
    expected = f["cnt_filled"].iloc[i - 24 - 167: i - 24 + 1].mean()
    assert f.loc[i, "roll_168"] == pytest.approx(expected)


def test_changing_the_target_does_not_change_its_own_features():
    g = ft.make_grid(_toy())
    base = ft.add_features(g)
    g2 = g.copy(); g2.loc[250, "cnt_filled"] += 1000          # alter one hour's own demand
    changed = ft.add_features(g2)
    cols = ["lag_24", "lag_168", "roll_168"]
    assert changed.loc[250, cols].equals(base.loc[250, cols])


def test_split_is_chronological_and_disjoint():
    days = pd.date_range("2011-01-01", "2012-12-31")
    g = pd.DataFrame({"ts": days, "dteday": days, "usable": True})
    parts = ft.split(g)
    assert parts["train"]["dteday"].max() < parts["val"]["dteday"].min() <= parts["val"]["dteday"].max() < parts["test"]["dteday"].min()
    assert sum(len(p) for p in parts.values()) == len(g)


def test_calendar_features():
    f = ft.add_features(ft.make_grid(_toy()))
    row = f[f["hr"] == 6].iloc[0]
    assert row["hr_sin"] == pytest.approx(1.0) and row["hr_cos"] == pytest.approx(0.0, abs=1e-9)
    assert f["is_weekend"].isin([0, 1]).all() and f["day_type"].isin([0, 1, 2]).all()
