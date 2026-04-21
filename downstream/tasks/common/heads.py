"""Head trainers for downstream-task eval.

Two flavors, swappable via ``head_type``:

    "mlp"       sklearn ``MLPClassifier`` / ``MLPRegressor`` inside a
                ``StandardScaler`` pipeline — fast, reproducible, matches the
                baselines we already have in scripts/.

    "autogluon" AutoGluon ``TabularPredictor`` — matches the GRAM-DTI paper's
                original repo (they use AutoGluon over learned features).
                Lazy-imported so the dependency is optional.

Protein embeddings (ESM-2) and molecule embeddings live on different scales;
the sklearn path z-scores on train only. AutoGluon does its own preprocessing.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HeadType = Literal["mlp", "autogluon", "linear"]
HEAD_CHOICES: Tuple[str, ...] = ("mlp", "autogluon", "linear")

TaskType = Literal["classification", "regression"]


class _AutoGluonWrapper:
    """Minimal predict_proba / predict facade over TabularPredictor.

    Exposes the same methods sklearn does so ``train_head`` can return a
    uniform object regardless of backend.
    """
    def __init__(self, predictor: Any, task: TaskType, feature_names: list[str]):
        self._predictor = predictor
        self._task = task
        self._feature_names = feature_names

    def _to_df(self, X: np.ndarray):
        import pandas as pd
        return pd.DataFrame(X, columns=self._feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._task != "classification":
            raise TypeError("predict_proba is only available for classification heads.")
        proba = self._predictor.predict_proba(self._to_df(X))
        # AutoGluon may return a DataFrame with class columns; enforce 2-col order [neg, pos].
        import pandas as pd
        if isinstance(proba, pd.DataFrame):
            if 1 in proba.columns and 0 in proba.columns:
                return proba[[0, 1]].values
            return proba.values
        return np.asarray(proba)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self._predictor.predict(self._to_df(X))
        return np.asarray(pred)


def _train_sklearn(
    X_tr: np.ndarray, y_tr: np.ndarray, task: TaskType, head_type: str,
    *, random_state: int, mlp_hidden: Tuple[int, ...], mlp_max_iter: int,
) -> Pipeline:
    if head_type == "linear":
        if task == "classification":
            estimator = LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )
        else:
            estimator = Ridge(alpha=1.0, random_state=random_state)
    elif head_type == "mlp":
        if task == "classification":
            estimator = MLPClassifier(
                hidden_layer_sizes=mlp_hidden, max_iter=mlp_max_iter,
                early_stopping=True, validation_fraction=0.1,
                random_state=random_state,
            )
        else:
            estimator = MLPRegressor(
                hidden_layer_sizes=mlp_hidden, max_iter=mlp_max_iter,
                early_stopping=True, validation_fraction=0.1,
                random_state=random_state,
            )
    else:
        raise ValueError(f"Unsupported sklearn head_type {head_type!r}")
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    pipeline.fit(X_tr, y_tr)
    return pipeline


def _train_autogluon(
    X_tr: np.ndarray, y_tr: np.ndarray, task: TaskType,
    *, time_limit: int, preset: str, save_dir: Optional[Path],
) -> _AutoGluonWrapper:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as e:
        raise ImportError(
            "AutoGluon not installed. `pip install autogluon.tabular` (heavy; pulls "
            "lightgbm/xgboost/catboost). Alternatively use --head mlp."
        ) from e

    import pandas as pd

    feature_names = [f"f{i}" for i in range(X_tr.shape[1])]
    train_df = pd.DataFrame(X_tr, columns=feature_names)
    train_df["label"] = y_tr

    problem_type = "binary" if task == "classification" else "regression"
    eval_metric = "average_precision" if task == "classification" else "root_mean_squared_error"

    path = save_dir or Path(tempfile.mkdtemp(prefix="autogluon_"))
    predictor = TabularPredictor(
        label="label", problem_type=problem_type, eval_metric=eval_metric,
        path=str(path), verbosity=1,
    ).fit(train_df, time_limit=time_limit, presets=preset)
    return _AutoGluonWrapper(predictor, task, feature_names)


def train_head(
    X_tr: np.ndarray, y_tr: np.ndarray, task: TaskType, head_type: HeadType,
    *,
    random_state: int = 0,
    mlp_hidden: Tuple[int, ...] = (256, 256),
    mlp_max_iter: int = 200,
    autogluon_time_limit: int = 600,
    autogluon_preset: str = "medium_quality",
    autogluon_save_dir: Optional[Path] = None,
):
    """Train a downstream head. Returns an object exposing ``predict`` (and
    ``predict_proba`` for classification)."""
    if head_type in ("mlp", "linear"):
        return _train_sklearn(
            X_tr, y_tr, task, head_type,
            random_state=random_state,
            mlp_hidden=mlp_hidden, mlp_max_iter=mlp_max_iter,
        )
    if head_type == "autogluon":
        return _train_autogluon(
            X_tr, y_tr, task,
            time_limit=autogluon_time_limit, preset=autogluon_preset,
            save_dir=autogluon_save_dir,
        )
    raise ValueError(f"Unknown head_type {head_type!r}. Choose from {HEAD_CHOICES}.")


def predict_proba_positive(clf, X: np.ndarray) -> np.ndarray:
    """Return shape-(N,) probability of the positive class, robust to single-class train."""
    proba = clf.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    # sklearn exposes .classes_; our AutoGluon wrapper already orders [neg, pos].
    classes = list(getattr(clf, "classes_", []))
    if classes and 1 in classes:
        return proba[:, classes.index(1)]
    if proba.shape[1] == 2:
        return proba[:, 1]
    return np.zeros(len(X), dtype=np.float32)
