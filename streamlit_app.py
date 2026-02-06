from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import load_dataset, split_features_target
from model_utils import build_model

ARTIFACTS_DIR = Path("artifacts")


@st.cache_data(show_spinner=False)
def get_dataset(refresh: bool) -> pd.DataFrame:
    return load_dataset(refresh=refresh)


def get_metrics() -> dict | None:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_model():
    model_path = ARTIFACTS_DIR / "stacked_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def train_and_cache_model(df: pd.DataFrame):
    features, target = split_features_target(df)
    model = build_model()
    model.fit(features, target)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "stacked_model.joblib")
    return model


def main() -> None:
    st.set_page_config(page_title="LoL Winrate Intelligence Lab", layout="wide")
    st.title("League of Legends Winrate Intelligence Lab")
    st.markdown(
        "An end-to-end research dashboard featuring stacked ensembles, calibration, "
        "permutation importance, and interactive scouting tools."
    )

    with st.sidebar:
        st.header("Data Controls")
        refresh = st.checkbox("Refresh dataset", value=False)
        if st.button("Train model (fast, cached)"):
            df = get_dataset(refresh=refresh)
            with st.spinner("Training stacked ensemble..."):
                train_and_cache_model(df)
            st.success("Model trained and cached.")

    df = get_dataset(refresh=refresh)
    features, target = split_features_target(df)

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Matches", f"{df.shape[0]:,}")
    col2.metric("Features", f"{features.shape[1]:,}")
    col3.metric("Blue Win Rate", f"{target.mean():.2%}")

    with st.expander("Preview raw data"):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Feature Correlations (top 20 by absolute correlation with target)")
    correlation = df.corr(numeric_only=True)["blueWins"].drop("blueWins").abs().sort_values(ascending=False)
    top_corr_features = correlation.head(20).index
    corr_matrix = df[list(top_corr_features) + ["blueWins"]].corr(numeric_only=True)
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Model Insights")
    model = get_model()
    metrics = get_metrics()
    if model is None:
        st.info("Train the model from the sidebar to unlock performance metrics and predictions.")
    if metrics is not None:
        test_metrics = metrics["test"]
        cv_metrics = metrics["cross_validation"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Holdout Accuracy", f"{test_metrics['accuracy']:.3f}")
        metric_cols[1].metric("Holdout ROC AUC", f"{test_metrics['roc_auc']:.3f}")
        metric_cols[2].metric("Holdout Log Loss", f"{test_metrics['log_loss']:.3f}")
        metric_cols[3].metric("Holdout Brier", f"{test_metrics['brier_score']:.3f}")

        st.markdown("**Cross-validation (5-fold mean)**")
        cv_cols = st.columns(3)
        cv_cols[0].metric("Accuracy", f"{cv_metrics['val_accuracy']:.3f}")
        cv_cols[1].metric("ROC AUC", f"{cv_metrics['val_roc_auc']:.3f}")
        cv_cols[2].metric("Log Loss", f"{cv_metrics['val_log_loss']:.3f}")

    importance_path = ARTIFACTS_DIR / "permutation_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).head(15)
        fig_importance = px.bar(
            importance_df,
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title="Permutation Importance (Top 15)",
        )
        fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Match Scouting Simulator")
    match_index = st.slider("Select a match row to simulate", 0, len(df) - 1, 0)
    selected_row = features.iloc[[match_index]]
    st.dataframe(selected_row, use_container_width=True)

    if model is not None:
        win_prob = model.predict_proba(selected_row)[:, 1][0]
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=win_prob * 100,
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f77b4"}},
                title={"text": "Predicted Blue Win Probability (%)"},
            )
        )
        st.plotly_chart(gauge, use_container_width=True)


if __name__ == "__main__":
    main()
