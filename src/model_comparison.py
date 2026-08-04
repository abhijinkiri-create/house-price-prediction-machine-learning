"""
House Price Prediction Project
Model Comparison
"""

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==============================
# Load Dataset
# ==============================

X = pd.read_csv("data/X_train_processed.csv")

y = pd.read_csv("data/y_train.csv").squeeze()

# ==============================
# Split Dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==============================
# Models
# ==============================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

}

# ==============================
# Compare Models
# ==============================

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    mse = mean_squared_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        prediction
    )

    results.append({

        "Model": name,

        "MAE": round(mae,2),

        "MSE": round(mse,2),

        "RMSE": round(rmse,2),

        "R2 Score": round(r2,4)

    })

# ==============================
# Result DataFrame
# ==============================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\nModel Comparison\n")

print(results_df)

# ==============================
# Save Results
# ==============================

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nComparison Report Saved Successfully!")