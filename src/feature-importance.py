"""
House Price Prediction Project
Feature Importance Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load Model
model = joblib.load("models/house_price_model.pkl")

# Load Dataset
X = pd.read_csv("data/X_train_processed.csv")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

top20 = importance.head(20)

print(top20)

plt.figure(figsize=(12,8))

plt.barh(top20["Feature"], top20["Importance"])

plt.gca().invert_yaxis()

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.show()

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("Feature Importance saved successfully!")