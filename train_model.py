from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from data_utils import load_dataset, split_features_target
from model_utils import build_model


def evaluate_model(model, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    return metrics


def cross_validate_model(model, x, y, cv_splits: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring=["accuracy", "roc_auc", "neg_log_loss"],
        n_jobs=-1,
        return_train_score=True,
    )
    return {
        "train_accuracy": float(np.mean(scores["train_accuracy"])),
        "train_roc_auc": float(np.mean(scores["train_roc_auc"])),
        "train_log_loss": float(-np.mean(scores["train_neg_log_loss"])),
        "val_accuracy": float(np.mean(scores["test_accuracy"])),
        "val_roc_auc": float(np.mean(scores["test_roc_auc"])),
        "val_log_loss": float(-np.mean(scores["test_neg_log_loss"])),
    }


def save_artifacts(
    artifacts_dir: Path,
    model,
    metrics: dict,
    cv_metrics: dict,
    feature_importance: pd.DataFrame,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "stacked_model.joblib")
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"test": metrics, "cross_validation": cv_metrics}, handle, indent=2)
    feature_importance.to_csv(artifacts_dir / "permutation_importance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the advanced League of Legends winrate model.")
    parser.add_argument("--refresh-data", action="store_true", help="Download the dataset again.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Output directory for model artifacts.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data for holdout test.")
    args = parser.parse_args()

    df = load_dataset(refresh=args.refresh_data)
    df = df.dropna()
    features, target = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=42,
        stratify=target,
    )

    model = build_model()
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    cv_metrics = cross_validate_model(model, features, target)

    perm_importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": perm_importance.importances_mean,
                "importance_std": perm_importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    save_artifacts(Path(args.artifacts_dir), model, metrics, cv_metrics, importance_df)


if __name__ == "__main__":
    main()
