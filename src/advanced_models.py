"""
House Price Prediction
Advanced Regression Models
"""

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

# ==========================
# Load Dataset
# ==========================

X = pd.read_csv("data/X_train_processed.csv")

y = pd.read_csv("data/y_train.csv").squeeze()

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Models
# ==========================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )

}

results = []

# ==========================
# Train Models
# ==========================

for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)

    rmse = np.sqrt(mean_squared_error(y_test, prediction))

    r2 = r2_score(y_test, prediction)

    results.append({

        "Model": name,

        "MAE": round(mae,2),

        "RMSE": round(rmse,2),

        "R2 Score": round(r2,4)

    })

# ==========================
# Result Table
# ==========================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\nModel Comparison\n")

print(results_df)

results_df.to_csv(
    "outputs/advanced_model_comparison.csv",
    index=False
)

print("\nComparison Saved Successfully!")