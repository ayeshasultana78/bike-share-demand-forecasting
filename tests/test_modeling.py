import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import modeling as md  # noqa: E402


def test_score_perfect_and_bias():
    s = md.score([10, 20, 30], [10, 20, 30])
    assert s["MAE"] == 0 and s["RMSE"] == 0 and s["R2"] == pytest.approx(1.0) and s["WAPE_%"] == 0
    s = md.score([10, 20], [12, 18])
    assert s["MAE"] == pytest.approx(2.0) and s["Bias"] == pytest.approx(0.0) and s["WAPE_%"] == pytest.approx(4 / 30 * 100)


def test_score_clips_negative_predictions():
    assert md.score([5], [-3])["MAE"] == pytest.approx(5.0)


def test_level_adjusted_model_scales_with_level():
    rng = np.random.default_rng(0)
    n = 600
    hr = rng.integers(0, 24, n)
    level = rng.uniform(50, 300, n)
    rate = 0.5 + (hr >= 8) * 1.5                                  # demand = level x rate (hour-dependent)
    X = pd.DataFrame({"hr": hr, "roll_168": level})
    y = rng.poisson(level * rate)
    m = md.LevelAdjustedPoisson(features=["hr"], n_estimators=80, max_depth=3).fit(X, y)
    lo = m.predict(pd.DataFrame({"hr": [12], "roll_168": [100.0]}))[0]
    hi = m.predict(pd.DataFrame({"hr": [12], "roll_168": [200.0]}))[0]
    assert hi / lo == pytest.approx(2.0, rel=1e-6)                # forecast is exactly proportional to the level
    assert lo == pytest.approx(100 * 2.0, rel=0.1)                # and recovers the true rate


def test_split_conformal_width_monotone():
    s = np.linspace(0, 1, 100)
    assert md.split_conformal_width(s, 0.8) < md.split_conformal_width(s, 0.95)


def test_bootstrap_identical_models_gives_zero_reduction():
    df = pd.DataFrame({"dteday": np.repeat(pd.date_range("2012-01-01", periods=30), 24)})
    e = np.random.default_rng(0).normal(size=len(df))
    lo, hi = md.daily_block_bootstrap(df, e, e, n_boot=200)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)
