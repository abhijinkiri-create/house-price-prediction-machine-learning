"""
House Price Prediction
Cross Validation
"""

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

# ==========================
# Load Dataset
# ==========================

X = pd.read_csv("data/X_train_processed.csv")

y = pd.read_csv("data/y_train.csv").squeeze()

# ==========================
# Model
# ==========================

model = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

# ==========================
# Cross Validation
# ==========================

scores = cross_val_score(

    model,

    X,

    y,

    cv=5,

    scoring="r2",

    n_jobs=-1

)

# ==========================
# Results
# ==========================

print("\nCross Validation Scores\n")

print(scores)

print("\nAverage R2 Score :", round(scores.mean(),4))

print("Standard Deviation :", round(scores.std(),4))

results = pd.DataFrame({

    "Fold": range(1, 6),

    "R2 Score": scores

})

results.loc[len(results)] = [

    "Average",

    scores.mean()

]

results.to_csv(

    "outputs/cross_validation_results.csv",

    index=False

)

print("\nCross-validation results saved successfully!")