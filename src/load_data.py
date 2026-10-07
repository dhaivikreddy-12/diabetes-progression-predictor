"""Generate a synthetic but realistic student marks dataset."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

n = 300
study_hours = np.clip(rng.normal(4.5, 1.8, n), 0.5, 12).round(1)
sleep_hours = np.clip(rng.normal(7, 1.2, n), 3, 10).round(1)
attendance_pct = np.clip(rng.normal(85, 12, n), 40, 100).round(1)
previous_score = np.clip(rng.normal(70, 15, n), 20, 100).round(1)

score = (
    2.5 * (study_hours * 2.0)
    + 0.30 * attendance_pct
    + 0.22 * previous_score
    + 1.0 * sleep_hours
    + rng.normal(0, 3.0, n)
)
final_score = np.clip(score, 5, 100).round(1)

df = pd.DataFrame({
    "study_hours": study_hours,
    "sleep_hours": sleep_hours,
    "attendance_pct": attendance_pct,
    "previous_score": previous_score,
    "final_score": final_score,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/student_marks.csv", index=False)
print(f"Generated {len(df)} students -> data/student_marks.csv")
print(df.head())
