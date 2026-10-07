"""Exploratory analysis on the real Diabetes dataset."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load

df = load()
os.makedirs("plots", exist_ok=True)

TARGET = "disease_progression"
FEATURES = [c for c in df.columns if c != TARGET]

corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
print("\n--- Correlation with disease progression ---")
for feat, val in corr.items():
    print(f"{feat:<14} {val:+.3f}")

plt.figure(figsize=(8, 5))
sns.barplot(x=corr.values, y=corr.index, color="#4c72b0")
plt.xlabel("Pearson correlation")
plt.title(f"Feature correlation with {TARGET}")
plt.tight_layout()
plt.savefig("plots/correlation.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 6))
sns.heatmap(df[FEATURES + [TARGET]].corr(), cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("plots/corr_heatmap.png", dpi=120)
plt.close()

print("Saved plots to plots/")
