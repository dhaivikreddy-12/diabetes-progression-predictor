"""Load the real Diabetes dataset (442 patients, 10 baseline features).

Source: sklearn.datasets.load_diabetes, a well-known regression benchmark
derived from clinical measurements. Cached to data/ for offline reruns.
"""
import os
import pandas as pd

CSV_PATH = "data/diabetes.csv"


def load():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes()
    df = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
    df["disease_progression"] = bunch.target
    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} patients -> {CSV_PATH}")
    print(df.head())
