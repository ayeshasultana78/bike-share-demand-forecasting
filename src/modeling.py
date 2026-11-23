"""Models, metrics and helpers for the bike-share demand-forecasting capstone."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import features as ft

LEVEL = "roll_168"                                           # recent demand level (mean of the 7 days ending 24 h earlier)
FS_REL = [c for c in ft.FS_CAL if c != "t_days"]             # features of the level-adjusted model (growth enters via the level)


def score(y_true, y_pred) -> dict:
    """Accuracy metrics for count forecasts. WAPE = total absolute error / total demand."""
    y_true, y_pred = np.asarray(y_true, float), np.maximum(np.asarray(y_pred, float), 0.0)
    err = y_pred - y_true
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)), "WAPE_%": float(np.abs(err).sum() / y_true.sum() * 100),
            "Bias": float(err.mean())}


class LevelAdjustedPoisson(BaseEstimator, RegressorMixin):
    """Poisson gradient boosting with an exposure term.

    The booster learns the *rate* = demand / recent level (a Poisson model with exposure, fitted by weighting each row by
    its level). The forecast is rate x level, so growth in overall demand is handled by multiplication instead of
    requiring the trees to extrapolate a trend they have never seen.
    """

    def __init__(self, features=None, level=LEVEL, n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1,
                 subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, random_state=0):
        self.features = features
        self.level = level
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def _xgb(self):
        from xgboost import XGBRegressor
        return XGBRegressor(objective="count:poisson", tree_method="hist", n_jobs=-1, n_estimators=self.n_estimators,
                            learning_rate=self.learning_rate, max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                            subsample=self.subsample, colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                            random_state=self.random_state)

    def fit(self, X: pd.DataFrame, y):
        cols = list(self.features) if self.features is not None else list(FS_REL)
        self.features_ = cols
        lvl = np.maximum(X[self.level].to_numpy(float), 1.0)
        self.model_ = self._xgb().fit(X[cols], np.asarray(y, float) / lvl, sample_weight=lvl)
        return self

    def predict(self, X: pd.DataFrame):
        lvl = np.maximum(X[self.level].to_numpy(float), 1.0)
        return self.model_.predict(X[self.features_]) * lvl


def level_profile_baseline(train: pd.DataFrame):
    """'Recent level x typical hourly profile': the strongest simple benchmark. Returns a predictor function."""
    ratio = (train["cnt"] / train[LEVEL]).groupby([train["day_type"], train["hr"]]).mean()

    def predict(df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["day_type"], df["hr"]])
        return df[LEVEL].to_numpy() * ratio.reindex(idx).to_numpy()
    return predict


def static_profile_baseline(train: pd.DataFrame):
    prof = train.groupby(["day_type", "hr"])["cnt"].mean()

    def predict(df: pd.DataFrame) -> np.ndarray:
        return prof.reindex(pd.MultiIndex.from_arrays([df["day_type"], df["hr"]])).to_numpy()
    return predict


def daily_block_bootstrap(df: pd.DataFrame, err_a, err_b, n_boot: int = 2000, seed: int = 0):
    """Bootstrap (resampling whole days) the relative MAE reduction of model A versus model B."""
    d = pd.DataFrame({"day": df["dteday"].to_numpy(), "a": np.abs(err_a), "b": np.abs(err_b)})
    g = d.groupby("day").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    a, b = g["a"].to_numpy()[idx].sum(1), g["b"].to_numpy()[idx].sum(1)
    red = 1 - a / b
    return float(np.percentile(red, 2.5)), float(np.percentile(red, 97.5))


def split_conformal_width(scores: np.ndarray, level: float = 0.8) -> float:
    """Finite-sample-corrected quantile of normalised absolute residuals."""
    n = len(scores)
    k = int(np.ceil((n + 1) * level))
    return float(np.sort(scores)[min(k, n) - 1])
