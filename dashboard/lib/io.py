import glob
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).parent.parent.parent
REPORTS_PATH = Path(os.getenv("REPORTS_PATH", str(BASE_DIR / "reports")))
DELTA_PATH = Path(os.getenv("DELTA_PATH", str(BASE_DIR / "data" / "delta")))


def _eda_path(name: str) -> Path:
    return REPORTS_PATH / "eda_outputs" / name


def _ml_path(name: str) -> Path:
    return REPORTS_PATH / "ml_outputs" / name


def _fig_path(name: str) -> Path:
    return REPORTS_PATH / "figures" / name


@st.cache_data
def read_eda_csv(name: str) -> pd.DataFrame | None:
    path = _eda_path(name)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def read_ml_csv(name: str) -> pd.DataFrame | None:
    path = _ml_path(name)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def read_figure(name: str) -> Image.Image | None:
    path = _fig_path(name)
    if not path.exists():
        return None
    return Image.open(path)


def read_best_model_summary() -> dict:
    path = _ml_path("best_model_summary.txt")
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def ml_file_exists(name: str) -> bool:
    return _ml_path(name).exists()


@st.cache_data(show_spinner="Silver verisi okunuyor...")
def read_silver_sample() -> pd.DataFrame | None:
    """Reads Silver parquet files, returns sampled DataFrame with app/sentiment/votes/text_len."""
    silver_path = DELTA_PATH / "silver" / "steam_reviews_clean"
    if not silver_path.exists():
        return None
    parquet_files = glob.glob(str(silver_path / "*.parquet"))
    if not parquet_files:
        return None
    try:
        import pyarrow.parquet as pq
        dfs = []
        for f in parquet_files:
            table = pq.read_table(f, columns=["app_name", "sentiment", "review_votes", "review_text"])
            df_part = table.to_pandas()
            # Sample up to 75k rows per file to keep memory manageable
            if len(df_part) > 75_000:
                df_part = df_part.sample(n=75_000, random_state=42)
            dfs.append(df_part)
        combined = pd.concat(dfs, ignore_index=True)
        combined["text_len"] = combined["review_text"].str.len()
        return combined[["app_name", "sentiment", "review_votes", "text_len"]]
    except Exception:
        return None
