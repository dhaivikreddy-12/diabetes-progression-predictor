"""CLI to estimate a patient's disease progression score."""
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.load_data import load

df = load()
TARGET = "disease_progression"
FEATURES = [c for c in df.columns if c != TARGET]

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(df[FEATURES], df[TARGET])


def main():
    parser = argparse.ArgumentParser(
        description="Predict diabetes disease progression",
        epilog="Pass the 10 standardised features using --key value pairs, e.g. --age 0.03 --bmi 0.05",
    )
    for f in FEATURES:
        parser.add_argument(f"--{f}", type=float, required=False)
    args = parser.parse_args()

    missing = [f for f in FEATURES if getattr(args, f) is None]
    if missing:
        print(f"Missing required features: {', '.join(missing)}")
        parser.print_help()
        return

    row = {f: getattr(args, f) for f in FEATURES}
    pred = model.predict(pd.DataFrame([row]))[0]
    print(f"\nEstimated disease progression: {pred:.1f}")
    print("(Target is one year after baseline, in a scaled units)")


if __name__ == "__main__":
    main()
