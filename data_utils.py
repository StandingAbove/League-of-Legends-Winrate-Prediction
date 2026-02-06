from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/trevorkarn/MLCamp2022/main/"
    "high_diamond_ranked_10min.csv"
)
DEFAULT_CACHE_PATH = Path("data/high_diamond_ranked_10min.csv")


def load_dataset(
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        df = pd.read_csv(DATA_URL)
        df.to_csv(cache_path, index=False)
        return df
    return pd.read_csv(cache_path)


def split_features_target(df: pd.DataFrame, target: str = "blueWins") -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    features = df.drop(columns=[target])
    return features, df[target]
