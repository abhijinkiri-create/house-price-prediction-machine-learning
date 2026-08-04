"""
House Price Prediction Project
Feature Selection & Model Retraining
"""

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split

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
# Load Best Model
# ==============================

model = joblib.load("models/house_price_model.pkl")

# ==============================
# Feature Importance
# ==============================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

# ==============================
# Select Top 20 Features
# ==============================

selected_features = importance.head(20)["Feature"].tolist()

print("\nSelected Features:\n")

print(selected_features)

# ==============================
# Keep Only Selected Features
# ==============================

X_selected = X[selected_features]

# ==============================
# Train Validation Split
# ==============================

X_train, X_valid, y_train, y_valid = train_test_split(

    X_selected,

    y,

    test_size=0.20,

    random_state=42

)

# ==============================
# Train Model
# ==============================

model = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

model.fit(

    X_train,

    y_train

)

# ==============================
# Prediction
# ==============================

predictions = model.predict(X_valid)

# ==============================
# Evaluation
# ==============================

mae = mean_absolute_error(

    y_valid,

    predictions

)

rmse = np.sqrt(

    mean_squared_error(

        y_valid,

        predictions

    )

)

r2 = r2_score(

    y_valid,

    predictions

)

print("\nModel Performance")

print("MAE :", round(mae,2))

print("RMSE :", round(rmse,2))

print("R2 Score :", round(r2,4))

# ==============================
# Save Model
# ==============================

joblib.dump(

    model,

    "models/house_price_top20.pkl"

)

# ==============================
# Save Selected Features
# ==============================

joblib.dump(

    selected_features,

    "models/selected_features.pkl"

)

print("\nTop-20 Model Saved Successfully!")

print("Selected Features Saved Successfully!")