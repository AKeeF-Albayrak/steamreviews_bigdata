import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/steam_reviews.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset bulunamadı: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, nrows=10)

print("Kolonlar:")
print(df.columns.tolist())

print("\nİlk 10 satır:")
print(df.head())

print("\nVeri tipleri:")
print(df.dtypes)