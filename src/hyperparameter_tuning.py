"""
House Price Prediction Project
Hyperparameter Tuning
"""

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

import numpy as np

# ==============================
# Load Dataset
# ==============================

X = pd.read_csv("data/X_train_processed.csv")

y = pd.read_csv("data/y_train.csv").squeeze()

# ==============================
# Train Validation Split
# ==============================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==============================
# Base Model
# ==============================

rf = RandomForestRegressor(random_state=42)

# ==============================
# Hyperparameter Grid
# ==============================

param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [10, 20, None],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2]

}

# ==============================
# Grid Search
# ==============================

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=5,

    scoring="r2",

    n_jobs=-1,

    verbose=2

)

grid_search.fit(X_train, y_train)

# ==============================
# Best Model
# ==============================

best_model = grid_search.best_estimator_

print("\nBest Parameters")

print(grid_search.best_params_)

# ==============================
# Prediction
# ==============================

predictions = best_model.predict(X_valid)

# ==============================
# Evaluation
# ==============================

mae = mean_absolute_error(y_valid, predictions)

rmse = np.sqrt(mean_squared_error(y_valid, predictions))

r2 = r2_score(y_valid, predictions)

print("\nEvaluation")

print("MAE :", round(mae,2))

print("RMSE :", round(rmse,2))

print("R2 :", round(r2,4))

# ==============================
# Save Model
# ==============================

joblib.dump(

    best_model,

    "models/house_price_model.pkl"

)

print("\nOptimized Model Saved Successfully!")