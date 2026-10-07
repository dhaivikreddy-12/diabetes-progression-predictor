"""Train models on the real Diabetes dataset and report feature importance."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.load_data import load

df = load()
print(f"Loaded {len(df)} patients, columns: {list(df.columns)}")

TARGET = "disease_progression"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

print("\n--- Model comparison (test split) ---")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = (model, r2)
    print(f"{name:<18} MAE {mae:>6.2f} | RMSE {rmse:>6.2f} | R2 {r2:.4f}")

best_name = max(results, key=lambda k: results[k][1])
best_model, best_r2 = results[best_name]
print(f"\nBest model: {best_name} (R2 {best_r2:.4f})")

importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n--- Feature importance ---")
for feat, val in importance.items():
    print(f"{feat:<14} {val:.3f}")

importance.plot(kind="barh", color="#4c72b0")
plt.xlabel("Importance")
plt.title(f"Feature Importance - Diabetes Progression ({best_name})")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("importance.png", dpi=120)
print("\nSaved plot to importance.png")
