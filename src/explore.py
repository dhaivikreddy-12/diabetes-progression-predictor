"""Explore relationships in the student marks data."""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/student_marks.csv")
os.makedirs("plots", exist_ok=True)

sns.scatterplot(data=df, x="study_hours", y="final_score", alpha=0.6)
plt.title("Study Hours vs Final Score")
plt.tight_layout()
plt.savefig("plots/study_vs_score.png", dpi=120)
plt.close()

sns.scatterplot(data=df, x="attendance_pct", y="final_score", alpha=0.6)
plt.title("Attendance vs Final Score")
plt.tight_layout()
plt.savefig("plots/attendance_vs_score.png", dpi=120)
plt.close()

print("Saved plots to plots/")
