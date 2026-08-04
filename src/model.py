import pandas as pd
import numpy as np
import sklearn  
import matplotlib.pyplot as plt
import seaborn as sns

print("all libraries are imported successfully!")
print("numPy version:", np.__version__)
print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__) 


"""
House Price Prediction Project
Model Training
"""

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import numpy as np
import joblib

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
# Load Processed Dataset
# ==============================

X = pd.read_csv("data/X_train_processed.csv")

y = pd.read_csv("data/y_train.csv").squeeze()

print("Dataset Loaded Successfully!")

# ==============================
# Split Dataset
# ==============================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Validation Shape :", X_valid.shape)

# ==============================
# Create Models
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
# Store Results
# ==============================

results = []

best_model = None

best_r2 = -999999

# ==============================
# Train Models
# ==============================

for name, model in models.items():

    print("\n" + "=" * 60)

    print("Training :", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

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

    results.append([

        name,

        mae,

        rmse,

        r2

    ])

    print("MAE :", round(mae,2))

    print("RMSE :", round(rmse,2))

    print("R2 Score :", round(r2,4))

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

# ==============================
# Result Table
# ==============================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "MAE",

        "RMSE",

        "R2 Score"

    ]

)

print("\n")

print(results_df)

# ==============================
# Best Model
# ==============================

best_model_name = results_df.sort_values(

    by="R2 Score",

    ascending=False

).iloc[0]["Model"]

print("\nBest Model :", best_model_name)

# ==============================
# Retrain Best Model
# ==============================

best_model.fit(X, y)

# ==============================
# Save Model
# ==============================

joblib.dump(

    best_model,

    "models/house_price_model.pkl"

)

print("\nModel Saved Successfully!")

# ==============================
# Feature Importance
# ==============================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": best_model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    print("\nTop 20 Important Features\n")

    print(importance.head(20))

print("\nProject Completed Successfully!")