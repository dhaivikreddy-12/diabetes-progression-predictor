"""Train a Random Forest regressor on student marks and show feature importance."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/student_marks.csv")
print(f"Loaded {len(df)} students")
print(df.describe().round(2))

X = df.drop(columns=["final_score"])
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation ---")
print(f"MAE: {mae:.2f} points")
print(f"R2 : {r2:.4f}")

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Feature importance ---")
for feat, val in importance.items():
    print(f"{feat:<18} {val:.3f}")

import matplotlib.pyplot as plt
importance.plot(kind="barh", color="#4c72b0")
plt.xlabel("Importance")
plt.title("Feature Importance - Student Marks")
plt.tight_layout()
plt.savefig("importance.png", dpi=120)
print("\nSaved plot to importance.png")
