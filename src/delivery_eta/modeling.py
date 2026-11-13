"""Models, cross-validation and the delivery-promise analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, NUMERIC

SEED = 42


def build_preprocessor() -> ColumnTransformer:
    """Impute and one-hot encode. Fitted inside each fold, so nothing leaks.

    Numeric gaps take the training-fold median; categorical gaps become their
    own "Unknown" level, because a missing traffic or weather reading is
    information in itself, not a value to guess.
    """
    numeric = SimpleImputer(strategy="median")
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Unknown"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def _pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def make_models() -> dict[str, Pipeline]:
    """Baseline, linear model, two tree ensembles and their average.

    Trees do not need feature scaling, so there is no scaler.
    """
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    )
    gb = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=SEED
    )
    return {
        "Baseline (always the average)": _pipe(DummyRegressor(strategy="mean")),
        "Linear regression": _pipe(LinearRegression()),
        "Random forest": _pipe(clone(rf)),
        "Gradient boosting": _pipe(clone(gb)),
        "RF + GB average": _pipe(VotingRegressor([("rf", rf), ("gb", gb)])),
    }


def cv_scores(model, X, y, cv, groups=None) -> np.ndarray:
    """MAE on every held-out fold."""
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[test_idx])
        scores.append(mean_absolute_error(y.iloc[test_idx], pred))
    return np.array(scores)


def compare_models(models: dict, X, y, cv) -> pd.DataFrame:
    """Mean and spread of fold MAE for each model, best first."""
    rows = []
    for name, model in models.items():
        s = cv_scores(model, X, y, cv)
        rows.append({"model": name, "MAE_min": s.mean(), "MAE_std": s.std()})
    return pd.DataFrame(rows).sort_values("MAE_min").reset_index(drop=True)


def oof_predictions(model, X, y, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions: every order is predicted by a model that never saw it."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv)


def error_by_segment(df: pd.DataFrame, y_true, y_pred, column: str) -> pd.DataFrame:
    """MAE, average actual time and order count for each level of a column."""
    frame = pd.DataFrame(
        {
            column: df[column].fillna("Unknown").values,
            "actual": np.asarray(y_true),
            "abs_error": np.abs(np.asarray(y_true) - np.asarray(y_pred)),
        }
    )
    out = frame.groupby(column).agg(
        orders=("actual", "size"),
        avg_actual_min=("actual", "mean"),
        MAE_min=("abs_error", "mean"),
    )
    return out.sort_values("MAE_min", ascending=False)


def grouped_permutation_importance(model, X, y, n_splits: int = 5, n_repeats: int = 10):
    """How much MAE rises when one original column is shuffled.

    Computed on held-out folds and averaged, on the raw columns (before
    one-hot encoding), so each bar is a feature a person recognises.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    per_fold = []
    for train_idx, test_idx in cv.split(X):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            fitted,
            X.iloc[test_idx],
            y.iloc[test_idx],
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats,
            random_state=SEED,
            n_jobs=-1,
        )
        per_fold.append(result.importances_mean)
    imp = np.vstack(per_fold)
    return (
        pd.DataFrame(
            {"feature": X.columns, "MAE_increase_min": imp.mean(0), "std": imp.std(0)}
        )
        .sort_values("MAE_increase_min", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# From a single number to a delivery promise
# ---------------------------------------------------------------------------

def linear_effects(fitted_pipeline) -> pd.Series:
    """Coefficients of a fitted linear pipeline, in minutes, by encoded column."""
    names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
    return pd.Series(fitted_pipeline.named_steps["model"].coef_, index=names)


def promise_models(point_model) -> dict[str, Pipeline]:
    """Three ways to set the promised delivery time.

    Every strategy is calibrated the same way (see `calibrated_promises`), so
    all of them aim at the same on-time rate and differ only in how long the
    customer is told to wait.
    """
    q_model = QuantileRegressor(quantile=0.9, alpha=0.0, solver="highs")
    return {
        "Same promise for every order": _pipe(DummyRegressor(strategy="mean")),
        "Point estimate + flat buffer": clone(point_model),
        "Quantile model (90th percentile)": _pipe(q_model),
    }


def calibrated_promises(model, X, y, target: float = 0.9,
                        n_splits: int = 5, inner_splits: int = 3) -> np.ndarray:
    """Out-of-fold promised times aimed at `target` on-time rate.

    For each outer fold: fit on the training part, then find the smallest
    extra minutes that would have made `target` of the training orders on time,
    using inner out-of-fold predictions (never in-sample ones). The test orders
    get prediction + that offset. The test fold is never used to set anything.
    """
    outer = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    promises = np.empty(len(y))
    for train_idx, test_idx in outer.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        inner = KFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
        inner_pred = cross_val_predict(clone(model), X_tr, y_tr, cv=inner)
        offset = np.quantile(y_tr.to_numpy() - inner_pred, target)
        fitted = clone(model).fit(X_tr, y_tr)
        promises[test_idx] = fitted.predict(X.iloc[test_idx]) + offset
    return promises


def promise_report(y_true, promises: dict[str, np.ndarray]) -> pd.DataFrame:
    """On-time rate, average promise, and how late the late orders are.

    Lateness uses the median: a handful of very long deliveries would
    dominate an average.
    """
    y = np.asarray(y_true)
    rows = []
    for name, p in promises.items():
        late = y > p
        rows.append(
            {
                "strategy": name,
                "on_time_%": 100 * (1 - late.mean()),
                "avg_promised_min": p.mean(),
                "median_min_late_when_late": float(np.median(y[late] - p[late])) if late.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)
