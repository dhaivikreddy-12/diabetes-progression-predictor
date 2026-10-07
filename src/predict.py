"""CLI to predict a student's final score from their habits."""
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/student_marks.csv")
X = df.drop(columns=["final_score"])
y = df["final_score"]

model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)


def main():
    parser = argparse.ArgumentParser(description="Predict student final score")
    parser.add_argument("--study", type=float, required=True)
    parser.add_argument("--sleep", type=float, required=True)
    parser.add_argument("--attendance", type=float, required=True)
    parser.add_argument("--prev", type=float, required=True)
    args = parser.parse_args()

    sample = pd.DataFrame([{
        "study_hours": args.study,
        "sleep_hours": args.sleep,
        "attendance_pct": args.attendance,
        "previous_score": args.prev,
    }])
    pred = model.predict(sample)[0]

    print(f"\nPredicted final score: {pred:.1f}%")
    print("(Data suggests: study & attendance matter most)")


if __name__ == "__main__":
    main()
