"""
House Price Prediction
Hyperparameter Tuning using RandomizedSearchCV
"""

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================
# Load Dataset
# ==========================

X = pd.read_csv("data/X_train_processed.csv")

y = pd.read_csv("data/y_train.csv").squeeze()

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Base Model
# ==========================

rf = RandomForestRegressor(
    random_state=42
)

# ==========================
# Parameter Distribution
# ==========================

param_dist = {

    "n_estimators": [100, 200, 300, 500],

    "max_depth": [10, 20, 30, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2", None]

}

# ==========================
# Random Search
# ==========================

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_dist,

    n_iter=20,

    scoring="r2",

    cv=5,

    random_state=42,

    n_jobs=-1

)

print("Training Started...")

random_search.fit(X_train, y_train)

print("Training Completed!")

# ==========================
# Best Parameters
# ==========================

print("\nBest Parameters")

print(random_search.best_params_)

# ==========================
# Best Model
# ==========================

best_model = random_search.best_estimator_

prediction = best_model.predict(X_test)

# ==========================
# Evaluation
# ==========================

mae = mean_absolute_error(y_test, prediction)

rmse = np.sqrt(mean_squared_error(y_test, prediction))

r2 = r2_score(y_test, prediction)

print("\nEvaluation")

print("MAE :", round(mae,2))

print("RMSE :", round(rmse,2))

print("R2 Score :", round(r2,4))

# ==========================
# Save Model
# ==========================

joblib.dump(

    best_model,

    "models/random_search_model.pkl"

)

print("\nOptimized Model Saved Successfully!")