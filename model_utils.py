from __future__ import annotations

from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    stacking_cv: int = 5
    calibration_cv: int = 3


def build_model(config: ModelConfig | None = None) -> Pipeline:
    if config is None:
        config = ModelConfig()

    base_estimators = [
        (
            "hist_gb",
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.08,
                max_iter=300,
                random_state=config.random_state,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                random_state=config.random_state,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=2,
                random_state=config.random_state,
                n_jobs=-1,
            ),
        ),
    ]

    meta_learner = LogisticRegression(
        max_iter=800,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.15,
        n_jobs=-1,
    )

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=config.stacking_cv,
        passthrough=True,
        n_jobs=-1,
    )

    calibrated = CalibratedClassifierCV(
        stacking,
        method="isotonic",
        cv=config.calibration_cv,
    )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", calibrated),
        ]
    )
