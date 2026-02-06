# Machine Learning Project: League of Legends Match Prediction (Research-Grade)

## Overview

Welcome to a research-grade League of Legends Match Prediction project. This repo now ships an advanced stacked-ensemble pipeline, probabilistic calibration, permutation-importance analysis, and an interactive Streamlit intelligence dashboard to explore and simulate match outcomes.

## Dataset

We automatically download a dataset containing high-diamond ranked matches with essential features that may influence match results. The dataset includes data points such as wards placed, kills, deaths, assists, objectives secured, and many other factors. The dataset is downloaded from:

[high_diamond_ranked_10min.csv](https://raw.githubusercontent.com/trevorkarn/MLCamp2022/main/high_diamond_ranked_10min.csv)

## Libraries Used

We use several Python libraries for data manipulation, machine learning, evaluation, and visualization:
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Models Explored

1. **Stacked Ensemble (HistGradientBoosting + ExtraTrees + RandomForest):** A layered ensemble that combines strong non-linear learners.
2. **Elastic-Net Logistic Meta-Learner:** Blends base estimators while controlling for overfitting.
3. **Isotonic Calibration:** Produces well-calibrated win probabilities for probabilistic decision-making.
4. **Permutation Importance:** Quantifies feature influence with uncertainty estimates.

## Cross-validation and Model Evaluation

We perform stratified cross-validation to estimate generalization. Evaluation includes accuracy, ROC AUC, log loss, Brier score, and confusion matrices. Metrics and artifacts are stored in `artifacts/` for reproducible analysis.

## Instructions

To replicate or further explore our project, follow these steps:

1. Clone or download this repository to your local machine.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the research-grade model and generate artifacts:

   ```bash
   python train_model.py
   ```

4. Launch the Streamlit dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

## Note

Model accuracy and performance may vary depending on the dataset, feature engineering, and hyperparameters. Feel free to extend the modeling stack, adjust calibration strategies, or integrate additional esports telemetry.

## Acknowledgments

We would like to express our gratitude to the developers of the datasets used in this project and the creators of the Python libraries that made this analysis possible.

Let's dive into the world of League of Legends match prediction! 🎮🏆
